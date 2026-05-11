# TOFU
This repository contains the official implementation for the paper [Diversity in Large Language Models under Supervised Fine-Tuning](https://arxiv.org/abs/2605.00195v1). It provides everything needed to reproduce our results, including scripts for model quantization, training, inference, and evaluation. You can find the quantized model weights and datasets at [TOFU-SFT](https://huggingface.co/TOFU-SFT).


## Overview
### Formulation
While Supervised Fine-Tuning (SFT) is essential for aligning Large Language Models with user intent, it often inadvertently suppresses generative diversity. Our research attributes this decline to two primary drivers: the neglect of low-frequency patterns within datasets and the forgetting of preexisting knowledge. To address these challenges, we introduce **Tempered Focal (TOFU) loss**, a principled objective designed to mitigate diversity collapse via gradient reweighting and temperature adjustment:

<p align="center">
  <img src="assets/eq_tofu.svg" width="300">
</p>

In this formulation, term <img align="top" src="assets/eq_term.svg"> is detached from the gradient computation.

### Performance
Across the creative writing and instruction-following benchmarks, TOFU achieves superior diversity while maintaining highly competitive response quality. Furthermore, we find that in mathematical Chain-of-Thought reasoning, TOFU encourages a higher exploration mode, thereby increasing the probability of capturing correct solutions. Crucially, this expanded diversity does not come at the cost of factual integrity or safety alignment. Altogether, these results position TOFU as a robust framework for improving model expressivity and functional utility across a wide range of downstream applications.

### Utilization
Our loss function is a seamless, drop-in replacement for the standard SFT loss. To use it with TRL simply copy the function from ```train/tofu_loss.py``` into your training code and pass it to ```SFTTrainer```:
```python
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(..., compute_loss_func=tofu_loss(gamma=..., beta=...)),
    train_dataset=dataset,
)
```
While gamma and beta are tunable, the default values (```gamma=3.0, beta=0.8```) were fixed upfront and used as-is across all models and benchmarks in our experiments, achieving state-of-the-art performance throughout. For precision-sensitive applications such as math reasoning, we recommend lowering the inference temperature, e.g. ```T=0.3``` as used in our math reasoning experiments.


## Experiments Reproduction

### Installation
Run the following to create a conda environment with the necessary dependencies.
```bash
conda create -n tofu python=3.11
```
Next, after the activation of ```tofu``` environment, install required libraries.
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchao==0.12.0 triton==3.4.0 --index-url https://download.pytorch.org/whl/cu128
pip install datasets==3.6.0
pip install einops
pip install transformers
pip install peft
pip install trl
pip install bitsandbytes
pip install sacrebleu
```

Next, after the activation of ```tofu``` environment, we recommend installing our code as a package. To do this, run the following.
```bash
pip install -e .
```

### Quantization
You can quantize a model using the following command:
```bash
python utils/quantize.py --model /path/to/model --output_dir /path/to/output/directory
```

### Training & Inference
After quantization, you can further train the model using:
```bash
python experiments/train_q_sft.py [options]
```
This supports training on datasets such as Alpaca and UltraFeedback, and allows selecting different methods, including CE, λ-PR, GEM, Focal Loss, and TOFU.
Inference and evaluation scripts are available in the ```inference``` and ```evaluation``` directories.


## Citation Information
If you find this repository useful for your research, please consider citing our preprint:
```
@misc{klypa2026diversitylargelanguagemodels,
      title={Diversity in Large Language Models under Supervised Fine-Tuning}, 
      author={Roman Klypa and Oleksandr Cherednichenko},
      year={2026},
      eprint={2605.00195},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.00195}, 
}
```