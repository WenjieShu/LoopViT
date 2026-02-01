# LoopViT: Scaling Visual ARC with Looped Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red.svg)](https://github.com/WenjieShu/LoopViT)

This is the official implementation of **LoopViT**, a recursive vision transformer architecture designed to solve abstract reasoning tasks in the [Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC).

### [Paper] | [Project Page]

**Wen-Jie Shu$^{1,*}$ \quad Xuerui Qiu$^2$ \quad Rui-Jie Zhu$^3$ \quad Harold Haodong Chen$^1$ \quad Yexin Liu$^1$ \quad Harry Yang$^1$**
$^1$HKUST \quad $^2$CASIA \quad $^3$UC Santa Cruz
$^*$Email: wenjieshu2003@gmail.com

---

## 🚀 Overview

Recent advances in visual reasoning have demonstrated that Vision Transformers can solve abstract grid-to-grid tasks (ARC) without symbolic engines. However, the prevailing feed-forward paradigm—where reasoning depth is bound to parameter count—is structurally mismatched with the induction of algorithms.

**LoopViT** addresses this by introducing a recursive architecture that decouples computational depth from model capacity:

- **Recursive Reasoning**: Iteratively processes a weight-tied *Hybrid Block* (Convolutions + Global Attention) to form a **latent chain of thought**.
- **Dynamic Exit**: A parameter-free mechanism based on **predictive entropy**. The model halts inference automatically when its internal state "crystallizes" into a low-uncertainty attractor.
- **Efficiency**: With only **11.2M parameters**, LoopViT achieves **62.2% accuracy** on ARC-AGI-1, outperforming massive 73M-parameter ensembles.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/WenjieShu/LoopViT.git
   cd LoopViT
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### Data Preparation
The model expects the ARC-AGI dataset. By default, place it in `raw_data/ARC-AGI`.

### Training
To train the base LoopViT model:
```bash
python offline_train_loop_ARC.py --architecture loop_vit --batch-size 32 --max-loop-steps 8
```

### Test-Time Training (TTT)
To enable test-time training for enhanced reasoning on specific tasks:
```bash
python test_time_train_ARC.py --resume-checkpoint <path_to_checkpoint>
```

---

## 🏗️ Project Structure

```text
LoopViT/
├── src/                        # Core model definitions
│   ├── ARC_LoopViT.py          # Recursive LoopViT architecture
│   ├── ARC_ViT.py              # Baseline ViT modules
│   └── ARC_loader.py           # ARC dataset loader & augmentations
├── utils/                      # Utilities
│   ├── eval_utils.py           # Evaluation logic
│   └── preprocess.py           # Grid preprocessing
├── offline_train_loop_ARC.py   # Training script
└── test_time_train_ARC.py      # Test-time training interface
```

---

## ✒️ Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{shu2026loopvit,
  title={LoopViT: Scaling Visual ARC with Looped Transformers},
  author={Shu, Wen-Jie and Qiu, Xuerui and Zhu, Rui-Jie and Chen, Harold Haodong and Liu, Yexin and Yang, Harry},
  journal={arXiv preprint},
  year={2026}
}
```
