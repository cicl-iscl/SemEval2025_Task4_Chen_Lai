# Interim Report 0212

## Task Overview
This project addresses the challenge of **Machine Unlearning** in Large Language Models (LLMs). The focus is to ensure selective forgetting of specific data (Forget sets) while retaining critical information (Retain sets). This involves handling synthetic and real datasets for tasks like sentence completion and question-answering, and assessing unlearning strategies using comprehensive evaluation metrics.

#### Deadlines
- Evaluation period: 10 to 30th January 2025
- Paper submission: 28 February 2025

#### SubTasks
- **Subtask 1**: Long-form synthetic creative documents.
- **Subtask 2**: Short form synthetic biographies containing personally identifiable information (PII).
- **Subtask 3**: Real documents from the target model’s training dataset.

#### Model Use
- A fine-tuned 7B model (base model: OLMo-7B-0724-Instruct-hf).  
- And a smaller fine-tuned 1B LLM.

#### Evaluation framework
- Regurgitation rates using ROUGE-L.
- Exact match rate for question-answering tasks.
- Membership inference attacks (MIA) detection rates.
- Performance benchmarking on MMLU datasets.

## Progress So Far
Our implementation focuses on training a fine-tuned unlearning pipeline for LLMs based on the task description. Below is a summary of the work completed.

### Algorithms
We implement an **iterative unlearning training loop** that balances:
- Minimise loss on Forget sets (unlearning data).
- Preserving high accuracy on Retain sets (critical information).
- Maintaing general knowledge and utility in the model, minise deviation from the original model.


### Process Flow Summary
The core implementation focuses on building a robust pipeline to selectively unlearn specific information from a pretrained language model while retaining crucial knowledge.
1. Data Preparation: Load and tokenize datasets into consistent input-output formats for Forget and Retain sets.
2. Loss Computation: Use KL divergence, answer loss, and retain loss to guide selective forgetting and retention.
3. Training: Iteratively compute losses and update the model parameters to achieve unlearning objectives.
4. Validation: Periodically evaluate the model's forgetting and retention performance.
5. Saving: Save intermediate and final model checkpoints for evaluation and potential reuse.

#### 1 Data Preparation
The first step involves reading datasets from Parquet files and processing them into PyTorch DataLoaders. The `create_dataloader_from_parquet` function tokenizes the data into a consistent input-output format (`### Input: ... ### Output: ...`) while ensuring truncation and padding to meet maximum sequence length constraints.

#### 2 Loss Computation
To guide the unlearning process:
- The **KL Divergence** between the pretrained model and the fine-tuned model helps retain general knowledge.
- The **Answer Loss** focuses on specific regions of tokenized inputs, allowing for gradient ascent (forgetting) or gradient descent (retaining) as necessary.
- The **Retain Loss** applies the answer loss specifically to the retain dataset, ensuring that the model maintains the correct associations for retained data.

#### 3 Model Finetune and Training Loop
The primary training function (`train`) orchestrates the unlearning process by iteratively calculating the combined loss. This process includes:
- Loading a pretrained model and datasets.
- Computing Forget Loss, Retain Loss, and Normal Loss for each training step.
- Backpropagation and model optimization using weighted combinations of these loss metrics.

#### 4 Validation and Checkpoints
At regular intervals, the model is evaluated on validation datasets for forgetting and retention performance. Checkpoints are saved periodically, ensuring the model’s progress is well-documented and recoverable.

#### 5 Execution and Argument Parsing
The program's entry point (`__main__`) handles user-defined arguments for paths and configurations. It triggers the training process by passing these inputs to the `train` function.

### Next Steps
The current codes have been sent to the cluster and waiting for the results. 

#### 1 Evaluation and Robust Testing
The model should undergone testing using the evaluation scripts given to quaitfy its unlearing performance. 

This includes assessing its effectiveness at forgetting targeted data while retaining essential knowledge, as measured by metrics like ROUGE-L and exact match rates. Benchmark tests on MMLU can verify the preservation of general language understanding abilities, while privacy-focused tests, such as membership inference attacks, will ensure the model does not inadvertently reveal sensitive data.

#### 2 Optimisation of Training Parameters
Experiment with hyperparamenter such as loss weights, learning rates (lr), batch sizes etc to improve performance. Adversarial testing should be conducted to determine whether the model can truly forget the targeted information under challenging or unexpected conditions. 

#### 3 Documentation and Submission
Wrap up a clear descriptions of methods, results, and code structure.
