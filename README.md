# LoopViT: Scaling Visual ARC with Looped Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red.svg)](https://arxiv.org/abs/2602.02156)

This is the official implementation of **LoopViT**, a recursive vision transformer architecture designed to solve abstract reasoning tasks in the [Abstraction and Reasoning Corpus (ARC)](https://github.com/fchollet/ARC).

### [Paper] | [Project Page]

**Wen-Jie Shu<sup>1,*</sup>, Xuerui Qiu<sup>2</sup>, Rui-Jie Zhu<sup>3</sup>, Harold Haodong Chen<sup>1</sup>, Yexin Liu<sup>1</sup>, Harry Yang<sup>1</sup>**

<sup>1</sup>HKUST &nbsp;&nbsp; <sup>2</sup>CASIA &nbsp;&nbsp; <sup>3</sup>UC Santa Cruz  
<sup>*</sup>Email: wenjieshu2003@gmail.com

---

## 🚀 Overview: Rethinking ARC as a Looped Process

Conventional Vision Transformers (ViTs) follow a **feed-forward paradigm**, where reasoning depth is strictly bound to the parameter count. However, abstract reasoning (ARC) is rarely a single-pass perceptual decision; it resembles an **iterative latent deliberation** where an internal state is repeatedly refined.

**Loop-ViT** establishes a new paradigm for visual reasoning by decoupling computational depth from model capacity:

- **Looped Vision Transformer**: We propose the first looped ViT architecture, establishing iterative recurrence as a powerful paradigm for abstract visual reasoning—demonstrating that **pure visual representations** are sufficient for ARC without needing linguistic or symbolic priors.
- **Scaling Time over Space**: Instead of solely relying on raw capacity ("Space"), Loop-ViT allows models to adapt computational effort ("Time") via a weight-tied **Hybrid Block** (Convolutions + Global Attention). This design aligns with the local, cellular-update nature of ARC transformations.
- **Predictive Crystallization (Dynamic Exit)**: We introduce a parameter-free mechanism where predictions "crystallize" (predictive entropy decays) over iterations. Loop-ViT halts early on easier tasks, significantly improving the accuracy-FLOPs Pareto frontier.
- **Empirical Superiority**: 
    - **Loop-ViT (Small, 3.8M)** achieves **60.1%** on ARC-AGI-1, surpassing the 18M VARC baseline (**54.5%**) with 1/5 the parameters.
    - **Loop-ViT (Large, 18M)** reaches **65.8%**, outperforming massive ensembles of feed-forward experts.

---

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/WenjieShu/LoopViT.git
   cd LoopViT
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📖 Usage

### Data Preparation
The model expects the ARC-AGI dataset. Please refer to the `raw_data` section in the [VARC repository](https://github.com/kyegomez/VARC) for detailed data processing instructions.
By default, place the data in `raw_data/ARC-AGI`.
The directory structure should be:
```
raw_data/
  ARC-AGI/
    data/
      training/
      evaluation/
      test/
```

### Training (Offline)
We provide a shell script to replicate our main experimental setup:
```bash
# Trains a 6-layer loop-core model (recurring 6 times)
bash script/offline_train_loop_VARC_ViT.sh
```
This script acts as a wrapper around `offline_train_loop_ARC.py` with the recommended hyperparameters.

### Test-Time Training (TTT)
To reproduce the TTT results on ARC-1:
```bash
# Runs TTT on ARC-1 evaluation tasks
bash script/test_time_training_VARC_LoopViT_ARC1.sh
```
This will iterate over tasks defined in `script/arc1_task_list.sh`.

#### Early Exit TTT (Dynamic Compute)
To run TTT with dynamic early exit and visualize the loop steps:
```bash
bash script/test_time_training_VARC_LoopViT_ARC1_early_exit.sh
```
This script enables `--exit-on-entropy-stable` and saves visualizations of attention maps and reasoning steps.

See `script/` for more examples of training and TTT scripts.

---

## 🏗️ Project Structure

```text
LoopViT/
├── src/                        # Core model definitions
│   ├── ARC_LoopViT_v1.py       # LoopViT model architecture (v1)
│   ├── ARC_loader.py           # ARC dataset loader & augmentations
│   ├── ARC_ViT.py              # Base ViT components
│   └── attn_hook.py            # Attention hooking for visualization
├── utils/                      # Utilities
│   ├── eval_utils.py           # Evaluation logic
│   ├── eval_utils_ttt.py       # TTT evaluation logic
│   └── vis_renderer.py         # Visualization renderer
├── script/                     # Shell scripts for training/eval
├── offline_train_loop_ARC.py   # Main offline training script
└── test_time_train_ARC.py      # Test-time training interface
```

## Acknowledgements
This codebase builds upon the [VARC](https://github.com/kyegomez/VARC) repository. We thank the authors for their open-source contribution which facilitated our research.

---


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
