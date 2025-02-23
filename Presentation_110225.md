## Part 1: Task Overview

Hello everyone! Today, I’ll be presenting our work on Machine Unlearning. 

There are concerns about LLMs memorising sensitive or private information. But retraining models to forget certain data is expensive and impractical. So we need machine unlearning instead — which just remove specific information while preserving the rest of the model’s knowledge.

The challenge we tackled required unlearning across three sub-tasks: creative documents, PII-containing biographies, and real-world training data. 

Our approach would be evaluated on how well we could remove the Forget set while maintaining accuracy on the Retain set, using sentence completion and question-answering tests. These are the two evaluation criteria given by the organisers.

<!-- Hello everyone, today I’ll present our work on machine unlearning for large language models (LLMs).

### Background

LLMs could memorize sensitive data and that will be risky (e.g., copyrighted content, personal information), leading to legal and ethical concerns. Retraining models from scratch is impractical due to high costs. Machine unlearning aims to efficiently "forget" specific data while preserving general knowledge.

### Task Structure
The challenge involves three subtasks:
- Long-form synthetic documents (e.g., fiction).
- Short-form synthetic biographies (with fake PII like names, SSNs).
- Real documents (sampled from the model’s original training data).

Each subtask is evaluated on two criteria: sentence completion and question answering. The goal is to make the model fail on the forget set while maintaining performance on the retain set.

### Key Challenges

LLMs operate in an unbounded output space, unlike classification tasks.

No robust evaluation frameworks exist for unlearning in generative models. -->

## Part 2: Approach & Code Walkthrough

Our core strategy is dual-objective optimization:

- Forget: Maximize loss on the forget set (gradient ascent).
- Retain: Align with the original model on the retain set (KL divergence).

For an easier manipulation, we started with a smaller 1B model, whose dataset is similar to the given 7B benchmark model.

### 1. Data Processing
The function `create_dataloader_from_parquet ` prepares the data. It tokenizes inputs from both datasets, make sure the model recognizes what should be forgotten or retained. Questions and answers are structured differently from general text generation. This guides the learning process in the right way.

### 2. Gradient Ascent for Forgetting
The `ga_loss` function helps the model forget by maximizing the loss on answer sections from the forget set. Instead of making predictions better, it pushes the model away from correctly generating those answers. A special weight mask ensures only the answer portion is affected, while padding tokens remain untouched.

### 3. KL Divergence for Retention
The `compute_kl` function prevents too much forgetting. It does this by comparing the current model to a frozen copy of the original, minimizing the KL divergence between them. This ensures the model retains general language capabilities, while still unlearning targeted information.

### 4. Training Loop
The `unlearn` function iteratively trains the model. It balances two loss functions above. The BAD_WEIGHT setting controls how much the model forgets. The NORMAL_WEIGHT setting helps it stay well-rounded. With the Accelerator library, everything runs smoothly on GPUs. 

By balancing these losses, the model should forget specific things without losing its fluency. This way, we unlearn only what we need—without starting from scratch.

## Part 3. Main challenges you faced:

One of the main challenges we faced was GPU constraints. We used the uni-cluster. Since our algorithm needed to load both the teacher and student models, their activation consumed a significant amount of GPU memory. As a result, we couldn’t adjust many parameters, such as using a large batch size or adding a random answer loss to maintain normal utility. 

We also spent a lot of time studying how to distribute training effectively, but in the end, only the 1B model was successfully trained.

Another challenge was the cluster queueing system. Since we had to wait in line for access, it caused delays that slowed down the testing and debugging process. Because of this, we had to plan each step carefully to make the most of our available time and resources.

## Part 4. Your results, your rank:

Our team ranked 15 place. Here’s a quick overview of our model’s unlearning evaluation results:

- ⁠MMLU Score: 0.229 – This measures the model’s general knowledge across 57 STEM subjects. Since it falls below the 0.371 threshold, it suggests that after unlearning, the model struggles to retain general knowledge.

- MIA Score: 0.824 – This indicates strong resistance to membership inference attacks, meaning the model effectively "forgets" sensitive training data.

- ⁠Task Aggregate Score: 0.0 – This score combines two key factors: the regurgitation _/rɪˌɡəːdʒɪˈteɪʃn/_ score and the exact match knowledge score. This indicates poor performance in one or more tasks.

- Our Final Score is 0.351. This average of the above scores reflects a balance between privacy protection and performance. It suggests that while our model excels in preventing data leakage, it has low general knowledge retention and task performance.

<!-- ### 1. Data Processing

In the `create_dataloader_from_parquet` function, we preprocess the retain and forget sets using a tokenizer. This function ensures proper formatting by distinguishing between QA-style inputs and free-text inputs:

```PYTHON
if "?" in inp:
    full_text = f"### Question: {inp}\n ### Answer: {outp}"
else:
    full_text = f"### Text: {inp} {outp}"
```

- QA pairs (input contains `?`): Structured as `### Question: ... ### Answer: ....`
- Text generation: Structured as` ### Text: ....`

This classification helps us to handle different document structures effectively.

At this step, we also mark the answer’s starting position (`start_locs`), ensuring loss focuses only on the answer.

### 2. Unlearning Mechanism:

### Loss function

- **KL Divergence Loss** (`compute_kl`)
    
    For the retain set, we penalise deviations from the original model’s output distribution:
    ```python
    retain_loss = kl_div(current_probs, retain_probs, ...)  # Align distributions  
    ```

- **Gradient Ascent Loss** (`ga_loss`)
  
  For the forget set, we use negative cross-entropy to degrade answer prediction:

    ```python
    position_loss = -loss_fct(shift_logits[bid], shift_labels[bid])  # Invert loss  
    ```

    Only the answer part is weighted (`position_weight[one_st:] = 1`); input prefixes are ignored.



### 3. Training Workflow (`unlearn` function)

- **Dual Data Streams**: Load retain and forget sets in parallel.
    
    Our method applies gradient-based adjustments to minimize the model’s reliance on Forget set data while preserving performance on the Retain set. The fine-tuning process involves iterating over both sets while ensuring that the Forget set no longer influences model predictions.

- **Weighted Loss Function**:

    ```python
    loss = BAD_WEIGHT * bad_loss + NORMAL_WEIGHT * normal_loss  # 0.2 vs 1.0 
    ``` 
    Balances forgetting strength vs. retention stability.
    
- **Efficiency**: Uses Hugging Face `Accelerator` for multi-GPU support.

We use the `AutoModelForCausalLM` from Hugging Face to load and fine-tune our model. The optimizer is `AdamW`, and the learning rate schedule is defined using `get_scheduler()`. This ensures that our model can adapt effectively during unlearning.

```PYTHON
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = MAX_UNLEARN_STEPS
lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
```

### Why This Approach?
1. **Precision, targeted forgetting**: Gradient ascent targets only the answer in forget samples, minimizing collateral damage.
2. **Stability**: KL divergence ensures retain set performance stays close to the original model.
3. **Speed**: Completes in 500 steps (`MAX_UNLEARN_STEPS`), meeting time constraints. 

### Validation

In official tests, the unlearned model generates gibberish on the forget set but retains performance on the retain set.

-->
