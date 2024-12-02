import argparse
import logging
import random
import time
import os

from datasets import Dataset
import numpy as np
import torch
from accelerate import Accelerator
#from peft import get_peft_model, AdaLoraConfig, TaskType
from torch.optim import AdamW
from transformers import DataCollatorForLanguageModeling, AutoModelForCausalLM, AutoTokenizer, get_scheduler

# Set random seeds for reproducibility
SEED = 8888
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def create_dataloader_from_parquet(tokenizer, parquet_file, batch_size=4, shuffle=True, max_length=512):
    """
    Create a DataLoader from a Parquet file containing retain or forget data.

    Args:
        tokenizer: Tokenizer.
        parquet_file: Path to a Parquet file with 'input' and 'output' columns.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the DataLoader.
        max_length: Maximum sequence length for tokenization.

    Returns:
        DataLoader of retain or forget set.
    """
    def preprocess(examples):
        full_texts = [
            f"### Input: {inp}\n ### Output: {outp}" 
            for inp, outp in zip(examples['input'], examples['output'])
        ]
        
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        
        # Calculate start locations
        start_locs = [
            len(tokenizer(f"### Input: {inp}\n ### Output: ", truncation=True, max_length=max_length)["input_ids"])
            for inp in examples['input']
        ]
        
        # Return as a dictionary
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "start_locs": start_locs,
            "labels": tokenized["input_ids"],  # Labels are the same as input_ids
        }
    
    # Load Parquet file as a Hugging Face Dataset
    dataset = Dataset.from_parquet(parquet_file)
    
    # Preprocess the dataset
    processed_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    
    # Set the format of the dataset for PyTorch
    processed_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "start_locs", "labels"])
    
    # Use a data collator to handle batching
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Initialize DataLoader
    dataloader = torch.utils.data.DataLoader(
        processed_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=shuffle
    )
    
    return dataloader

def compute_kl(pretrained_model, current_model, batch, device):
    """
    Compute forward Kullback–Leibler divergence as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss

def get_answer_loss(operation, batch, model, device):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """

    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Define position weights: 1 for answer part, 0 for other parts
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part(input)
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss

# Configuration constants
MAX_UNLEARN_STEPS = 5000
BAD_WEIGHT = 0.5
#RETAIN_WEIGHT = 1
NORMAL_WEIGHT = 1
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MAX_BAD_LOSS = 100
SAVE_EVERY = 500
LOG_FILE = "logs/default7B.log"

logging.basicConfig(
    filename=LOG_FILE,
    filemode="w+",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d-%H-%M",
    level=logging.INFO,
)

def train(input_model_path, retain_set, forget_set, retain_val_set, forget_val_set, output_model_path):
    # Initialize Accelerator for distributed training on multiple GPUs
    accelerator = Accelerator()  
    device = accelerator.device

    # Load models and tokenizer
    model = AutoModelForCausalLM.from_pretrained(input_model_path)
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")

    pretrained_model = AutoModelForCausalLM.from_pretrained(input_model_path)

    train_bad_loader = create_dataloader_from_parquet(tokenizer, forget_set, batch_size=BATCH_SIZE)
    train_normal_loader = create_dataloader_from_parquet(tokenizer, retain_set, batch_size=BATCH_SIZE)
    val_bad_loader = create_dataloader_from_parquet(tokenizer, forget_val_set, batch_size=BATCH_SIZE)
    val_normal_loader = create_dataloader_from_parquet(tokenizer, retain_val_set, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = MAX_UNLEARN_STEPS
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Use accelerator to prepare models, optimizer, and dataloaders for distributed training
    model, pretrained_model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler = accelerator.prepare(
        model, pretrained_model, optimizer, train_bad_loader, train_normal_loader, lr_scheduler
    )

    # Training loop
    model.train()
    bad_loss = 0.0
    step = 0
    start_time = time.time()

    while bad_loss < MAX_BAD_LOSS and step < MAX_UNLEARN_STEPS:
        for bad_batch, normal_batch in zip(train_bad_loader, train_normal_loader):
            # Calculate losses
            bad_loss = get_answer_loss("ga", bad_batch, model, device=device)
            # retain_loss = get_retain_ans_loss(normal_batch, tokenizer, model, device=device)
            normal_loss = compute_kl(pretrained_model, model, normal_batch, device=device)

            # Combine losses
            loss = BAD_WEIGHT * bad_loss + NORMAL_WEIGHT * normal_loss

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            logging.info(f"Step: {step}, Bad Loss: {bad_loss:.2f},  KL Loss: {normal_loss:.2f}")
            step += 1

            # Validation and checkpoint saving
            if step % SAVE_EVERY == 0:
                model.eval()
                val_bad_loss, val_normal_loss = 0, 0

                with torch.no_grad():
                    for val_bad_batch in val_bad_loader:
                        val_bad_loss += get_answer_loss("ga", val_bad_batch, model, device=device).item()

                    for val_normal_batch in val_normal_loader:
                        val_normal_loss += compute_kl(pretrained_model, model, val_normal_batch, device=device).item()
                        
                val_bad_loss /= len(val_bad_loader)
                val_normal_loss /= len(val_normal_loader)

                logging.info(f"Validation - Step: {step}, Val Bad Loss: {val_bad_loss:.2f}, Val Normal Loss: {val_normal_loss:.2f}")

                # Save checkpoint using Accelerator
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_model_path, save_function=accelerator.save)
                model.train()

    # Log total time and save final model
    end_time = time.time()
    logging.info(f"Total time: {int(end_time - start_time)} seconds")
    model.save_pretrained(output_model_path, from_pt=True)
    logging.info("Unlearning process complete.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLURM-compatible training script for unlearning model")
    parser.add_argument("--input_model_path", type=str, required=True, help="Path to the input model")
    parser.add_argument("--retain_set", type=str, required=True, help="Parquet file containing retain set")
    parser.add_argument("--forget_set", type=str, required=True, help="Parquet file containing forget set")
    parser.add_argument("--retain_val_set", type=str, required=True, help="Parquet file containing retain validation set")
    parser.add_argument("--forget_val_set", type=str, required=True, help="Parquet file containing forget validation set")
    parser.add_argument("--output_model_path", type=str, required=True, help="Path to save the output model")
    args = parser.parse_args()

    # Configure logging
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/training.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Run training
    train(args.input_model_path, args.retain_set, args.forget_set, args.retain_val_set, args.forget_val_set, args.output_model_path)
