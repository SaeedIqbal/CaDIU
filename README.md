# **Causal Disentangled Industrial Unlearning (CaDIU)**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

**Structured Forgetting for GDPR-Compliant, Edge-Deployable Industrial Anomaly Detection**

This repository contains the official implementation of **CaDIU**, a theoretically grounded framework for **Continual Learning and Private Unlearning (CLPU)** in industrial visual inspection. CaDIU redefines unlearning not as brute-force deletion, but as **structured information control**—enabling exact, efficient, and private forgetting of client-specific defects while preserving transferable anomaly knowledge.

> **Paper**: [Causal Disentangled Industrial Unlearning in Continual Lifelong Vision](https://arxiv.org/abs/xxxx.xxxxx)  
> **Benchmarks**: VisA, MVTec-AD, Real-IAD, BTAD  
> **Key Metrics**: APF ≥ 0.96, IRE ≤ 0.12, CLS ≤ 0.03, MER = 0.08

---

## 📌 **Research Gaps Addressed**

Existing CLPU methods (e.g., CLPU-DER++) fail to reconcile three critical limitations in industrial settings:

1. **Disentanglement Gap**: Full model isolation blocks knowledge transfer of universal anomaly primitives (e.g., texture disruptions), degrading detection accuracy on new products.
2. **Causal Leakage**: Unlearning downstream tasks in multi-stage pipelines (e.g., coarse → fine inspection) leaves indirect traces in upstream stages, violating system-level privacy.
3. **Memory Inefficiency**: Storing full ViT-Base models (≈340 MB) per temporary task is infeasible for edge deployment in high-resolution vision (e.g., VisA, Real-IAD).

---

## 🚀 **Our Contributions**

We introduce **Causal Disentangled Industrial Unlearning (CaDIU)**, the first framework to enable **efficient, private, and certifiable** lifelong industrial vision:

✅ **Privacy-Preserving Disentanglement**: Separates universal anomaly primitives from IP-sensitive semantics via causally disentangled ViT branches.  
✅ **Causal Unlearning**: Eliminates transitive privacy leakage in multi-stage pipelines using Defect Propagation Graphs (DPG) with ancestral parameter restoration.  
✅ **Information-Theoretic Memory Efficiency**: Achieves exact unlearning with **8% memory** of CLPU-DER++ via Sufficient Statistic Memory (SSM) storing only minimal sufficient statistics.  
✅ **GDPR Compliance**: Guarantees exact unlearning per GDPR Article 17 ("right to erasure") with verifiable metrics.

---

## 📊 **Evaluation Metrics**

| **Metric** | **Symbol** | **Ideal Value** | **Description** |
|------------|------------|-----------------|-----------------|
| Anomaly Primitive Fidelity | APF ↑ | ≥ 0.95 | Transfer of universal defect features |
| IP Reconstruction Error | IRE ↓ | ≤ 0.12 | Recoverability of proprietary semantics |
| Causal Leakage Score | CLS ↓ | ≤ 0.03 | Transitive privacy leakage in pipelines |
| Memory Efficiency Ratio | MER ↓ | 0.08 | Memory vs. CLPU-DER++ (1.0 = baseline) |

---

## 🏆 **State-of-the-Art Results**

### **Single-Stage Benchmarks (VisA, MVTec-AD)**

| **Method** | **VisA APF ↑** | **VisA IRE ↓** | **MVTec APF ↑** | **MVTec IRE ↓** | **Avg. APF ↑** | **MER ↓** |
|------------|----------------|----------------|------------------|------------------|----------------|-----------|
| Ind | 0.61 | 0.05 | 0.59 | 0.04 | 0.60 | 1.00 |
| Seq | 0.72 | 2.10 | 0.70 | 2.05 | 0.71 | 0.01 |
| EWC | 0.85 | 1.62 | 0.83 | 1.58 | 0.84 | 0.01 |
| LwF | 0.88 | 1.53 | 0.86 | 1.49 | 0.87 | 0.01 |
| LSF | 0.90 | 0.91 | 0.88 | 0.87 | 0.89 | 0.01 |
| DER++ | 0.94 | 0.89 | 0.93 | 0.85 | 0.94 | 0.01 |
| CLPU-DER++ | 0.94 | 0.89 | 0.93 | 0.85 | 0.94 | 1.00 |
| **CaDIU (Ours)** | **0.97** | **0.11** | **0.96** | **0.12** | **0.97** | **0.08** |

### **Multi-Stage Benchmarks (Real-IAD, BTAD)**

| **Method** | **Real-IAD APF ↑** | **Real-IAD CLS ↓** | **BTAD APF ↑** | **BTAD CLS ↓** | **Avg. APF ↑** | **MER ↓** |
|------------|--------------------|--------------------|----------------|----------------|----------------|-----------|
| Ind | 0.63 | 0.02 | 0.60 | 0.03 | 0.62 | 1.00 |
| Seq | 0.70 | 0.52 | 0.68 | 0.49 | 0.69 | 0.01 |
| EWC | 0.82 | 0.47 | 0.80 | 0.45 | 0.81 | 0.01 |
| LwF | 0.85 | 0.46 | 0.83 | 0.44 | 0.84 | 0.01 |
| LSF | 0.87 | 0.41 | 0.85 | 0.40 | 0.86 | 0.01 |
| DER++ | 0.91 | 0.38 | 0.89 | 0.36 | 0.90 | 0.01 |
| CLPU-DER++ | 0.92 | 0.33 | 0.90 | 0.32 | 0.91 | 1.00 |
| **CaDIU (Ours)** | **0.96** | **0.02** | **0.95** | **0.03** | **0.96** | **0.08** |

> **Key Insight**: CaDIU is the **only method** that simultaneously achieves **high APF (≥0.95)**, **low privacy leakage (IRE/CLS ≤ 0.12)**, and **low memory (MER = 0.08)**.

---

## 🗂️ **Repository Structure**

```bash
CaDIU/
├── configs/                  # YAML configs for datasets/models
├── data/                     # Industrial AD dataset loaders
├── models/                   # Core CaDIU modules (DAE, DPG, SSM)
├── methods/                  # Full CaDIU pipeline and baselines
├── utils/                    # Metrics, logging, visualization
├── experiments/              # Reproducible experiment scripts
├── results/                  # Precomputed tables and figures
├── scripts/                  # Helper scripts (data download, workflow)
├── environment.yml           # Conda environment
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🚀 **Quick Start**

```bash
# Clone repository
git clone https://github.com/your-username/CaDIU.git
cd CaDIU

# Create conda environment
conda env create -f environment.yml
conda activate cadiu

# Download datasets
bash scripts/download_data.sh

# Train CaDIU on VisA
python experiments/train.py --config configs/visa.yaml

# Unlearn a task
python experiments/unlearn.py --config configs/unlearn_visa.yaml

# Evaluate
python experiments/evaluate.py --config configs/eval_visa.yaml
```

---

## 📚 **Citation**

If you use CaDIU in your research, please cite our paper:

```bibtex
@article{iqbal2025cadiu,
  title={Structured Forgetting in Industrial Anomaly Detection: Disentanglement, Causality, and Information-Theoretic Unlearning},
  author={Saeed Iqbal},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

---

## 📜 **License**

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

**Structured Forgetting is not a constraint—it is the foundation of trustworthy lifelong industrial AI.**
