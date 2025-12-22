<h1 align="center">FaithLens</h1>

<p align="center">

<a href="https://arxiv.org/abs/XXXX.XXXXX">
<img alt="Static Badge" src="https://img.shields.io/badge/arXiv-FaithLens-red"></a>

<a href="https://huggingface.co/">
    <img alt="Static Badge" src="https://img.shields.io/badge/Model-HuggingFace-yellow">
</a>

<a href="https://huggingface.co/datasets/">
    <img alt="Static Badge" src="https://img.shields.io/badge/Dataset-HuggingFace-blue">
</a>

</p>

This repository contains the official implementation for **FaithLens**, a cost-efficient and explainable model for **faithfulness hallucination detection**.

---

## Introduction

**FaithLens** is a specialized hallucination detection model that jointly:
1. Predicts whether a claim is **faithful** or **hallucinated** with respect to a given document.
2. Generates **human-readable explanations** to justify its prediction.

FaithLens is trained in two stages:
- **Cold-start Supervised Fine-Tuning (SFT)** on high-quality synthetic data with explanations.
- **Rule-based Reinforcement Learning (RL)** to further optimize both prediction accuracy and explanation quality.

Experiments across 12 diverse benchmarks show that FaithLens outperforms advanced LLMs (e.g., GPT-4.1, o3) while maintaining significantly lower inference cost.

---

