# Interim Report

### Deadlines
- Evaluation period: 10 to 30th January 2025
- Paper submission: 28 February 2025

### Task Overview
The task focuses on Machine Unlearning in LLMs, specifically tackling the challenge of selectively forgetting specific data (Forget sets) without impairing the retention of critical data (Retain sets).

#### SubTasks
- **Subtask 1**: **Long form synthetic** creative documents spanning different genres.
- **Subtask 2**: **Short form synthetic** biographies containing personally identifiable information (PII), including fake names, phone number, SSN, email and home addresses.
- **Subtask 3**: **Real documents sampled** from the target model’s training dataset.

### Model Use
- A fine-tuned 7B model (base model: OLMo-7B-0724-Instruct-hf).  
- And a smaller fine-tuned 1B LLM.
``` 
<hf_token> access id: 
llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning.
```

### Evaluation framework

To evaluate each submission, we compute 
- task specific regurgitation rates (measured using **rouge-L scores**) 
on the sentence completion prompts and exact match rate for the question answers 
on both retain and forget sets; 
we invert forget set metrics to 1 - their value.  

In addition, we compute 
- Membership Inference Attack (MIA) rates using loss based attack on a sample of member+nonmember datasets, and 
- compute model performance on the MMLU benchmark. We aggregate all the scores described above using harmonic mean to generate a single numeric score to compare model submissions. 
We’re releasing our evaluation script with the data repository. 

You can download the evaluation script along with the MIA dataset from the repository 
`llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public` using commands listed above.

### Unlearning Benchmark
TBA

### Our solutions
