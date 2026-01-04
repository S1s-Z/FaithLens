<h1 align="center"> 
    <img src="./images/logo.png" alt="FaithLens Logo" width="40" style="vertical-align: middle;">
    FaithLens
</h1>

<p align="center">
<a href="https://arxiv.org/abs/2512.20182">
<img alt="Static Badge" src="https://img.shields.io/badge/arXiv-FaithLens-red"></a>
<a href="https://huggingface.co/ssz1111/FaithLens">
    <img alt="Static Badge" src="https://img.shields.io/badge/Model-HuggingFace-yellow"></a>
<a href="https://huggingface.co/datasets/ssz1111/FaithLens">
    <img alt="Static Badge" src="https://img.shields.io/badge/Dataset-HuggingFace-blue"></a>
</p>

This repo contains **FaithLens**, a cost-efficient and explainable model for **faithfulness hallucination detection**.
<p align="center">
    <img src="./images/intro.png" width="360">
</p>



## Abstract
Recognizing whether outputs from large language models (LLMs) contain faithfulness hallucination is crucial for real-world applications, e.g., retrieval-augmented generation and summarization. In this paper, we introduce **FaithLens**, a cost-efficient and effective faithfulness hallucination detection model that can jointly provide binary predictions and corresponding explanations to improve the trustworthiness. To achieve this, we first synthesize training data with explanations via advanced LLMs and apply a well-defined data filtering strategy to ensure label correctness, explanation quality, and data diversity. Subsequently, we fine-tune the model on these well-curated training data as a cold start and further optimize it with rule-based reinforcement learning, using rewards for both prediction correctness and explanation quality. Results on 12 diverse tasks show that the 8B-parameter FaithLens outperforms advanced models such as GPT-4.1 and o3. Also, FaithLens can produce high-quality explanations, delivering a distinctive balance of trustworthiness, efficiency, and effectiveness.

## Introduction

**FaithLens** is a specialized hallucination detection model that jointly:
1. Predicts whether a claim is **faithful** or **hallucinated** with respect to a given document.
2. Generates **human-readable explanations** to justify its prediction.

FaithLens is trained in two stages:
- **Cold-start SFT** on high-quality synthetic data with explanations.
- **Rule-based RL** to further optimize both prediction accuracy and explanation quality.

Experiments across 12 diverse benchmarks show that FaithLens outperforms advanced LLMs (e.g., GPT-4.1, o3) while maintaining significantly lower inference cost.


## Training

<p align="center">
    <img src="./images/training.png" width="720">
</p>


### Cold-Start Supervised Fine-Tuning (SFT)

In the cold-start SFT stage, we initialize FaithLens with high-quality synthetic data that includes both hallucination labels and corresponding explanations. This stage equips the model with the ability to jointly perform faithfulness hallucination detection and explanation generation.

To run SFT training, execute:

```bash
bash training/sft/train_llama8b_instruct.sh
```

### Rule-Based Reinforcement Learning (RL)

After SFT, FaithLens is further optimized using a rule-based reinforcement learning (RL) stage.
This stage adopts a GRPO-style policy optimization framework and leverages a composite reward consisting of prediction correctness, explanation quality, and output format.

To start RL training, run:

```bash
bash training/verl/rl_training.sh
```

## Detecting and Explaining Faithfulness Hallucination

### Installation

#### Method 1
You can follow these steps to use our package:
```sh
git clone https://github.com/S1s-Z/FaithLens.git
cd Faithlens
pip install .
```

After the installation, you can run ```python quickstart.py``` to try our model.

#### Method 2
The following command will install the MiniCheck package and all necessary dependencies.

```sh
pip install "faithlens @ git+https://github.com/S1s-Z/FaithLens.git@master"
```

### Usage
Our FaithLens is available on HuggingFace 🤗. More model details can be found from this [link](https://huggingface.co/ssz1111/FaithLens). 

Below is a simple use case of FaithLens. FaithLens will be automatically downloaded from Huggingface for the first time and cached in the specified directory.

```python
from faithlens.inference import FaithLensInfer
import json

# Predicted Label:
# 0 indicates the claim is hallucinated (the claim is not supported by the document)
# 1 indicates the claim is faithful (supported by the document)

detection_model = FaithLensInfer(model_name="ssz1111/FaithLens", device="cuda:0")
single_result = detection_model.infer(
    docs=["Relegation-threatened Romanian club Ceahlaul Piatra Neamt have sacked Brazilian coach Ze Maria for the second time in a week. Former Brazil defender Ze Maria was fired on Wednesday after a poor run, only to be reinstated the next day after flamboyant owner Angelo Massone decided to 'give the coaching staff another chance.' But the 41-year-old former Inter Milan and Parma right back, capped 25 times by Brazil, angered Massone again after Ceahlaul were beaten 2-0 by mid-table FC Botosani on Saturday. Ze Maria represented Brazil on 25 occasions during an international career spanning five years . The result left Ceahlaul 16th in the standings, six points adrift of safety. Ze Maria replaced Florin Marin in January to become Ceahlaul's third coach this season. He will be replaced by Serbian Vanya Radinovic."],
    claims=["Former brazil defender ze maria was fired on wednesday after a poor run. The 41-year-old was reinstated the next day after flamboyant owner angelo massone decided to'give the coaching staff another chance' but the 41-year-old angered massone again after ceahlaul were beaten 2-0 by mid-table fc botosani on saturday."],
)
print("Single Result:")
print(json.dumps(single_result, ensure_ascii=False, indent=2))
```



## Evaluation

We provide scripts to reproduce the benchmark results reported in the paper, including evaluations on LLM-AggreFact and HoVer.

To run inference and benchmark evaluation, execute:
```bash
bash benchmarks/infer.sh
```

## Environments
We provide separate environments for different stages during the model training, as the dependencies required for supervised fine-tuning, reinforcement learning, and benchmark evaluation are not identical. Each stage has its own requirements.txt, located in the corresponding directory. Meanwhile, if you only wish to use FaithLens for detecting hallucinations, you only need to use the requirements.txt file within the first-level folder.

## Citation
```bibtex
@misc{si2025faithlensdetectingexplainingfaithfulness,
      title={FaithLens: Detecting and Explaining Faithfulness Hallucination}, 
      author={Shuzheng Si and Qingyi Wang and Haozhe Zhao and Yuzhuo Bai and Guanqiao Chen and Kangyang Luo and Gang Chen and Fanchao Qi and Minjia Zhang and Baobao Chang and Maosong Sun},
      year={2025},
      eprint={2512.20182},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.20182}, 
}
```
