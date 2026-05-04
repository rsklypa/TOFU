# TOFU


## Installation
Run the following to create a conda environment with the necessary dependencies.
```bash
conda create -n cal python=3.11
```
Next, after the activation of ```cal``` environment, install required libraries.
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchao==0.12.0 triton==3.4.0 --index-url https://download.pytorch.org/whl/cu128
pip install datasets==3.6.0
pip install einops
pip install transformers
pip install peft
pip install trl
pip install bitsandbytes
```

Next, after the activation of ```cal``` environment, we recommend installing our code as a package. To do this, run the following.
```
pip install -e .
```

### HF repo

[Project](https://huggingface.co/PowCal)

### Model Families 

| **Category** | **Family** | **Models** |
|:-------------:|:-----------|:------------|
|  **Base** | **Pythia Family** | Pythia-12B |
|  | **Mistral Family** | Mistral-Nemo-Base-2407 |
|  | **OLMo Family** | OLMo-2-1124-13B |
|  **Aligned** | **LLaMA Family** | Llama-3.1-8B<br>Meta-Llama-3-8B |
|  | **Qwen Family** | Qwen-3-8B |
|  | **Phi Family** | Phi-4 |
|  **Judge** | **Judge Family** | Meta-Llama-3-70B-Instruct |
---
