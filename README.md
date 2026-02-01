# LoopViT: Loop Transformer for ARC

This repository contains the official implementation of **LoopViT** for the Abstraction and Reasoning Corpus (ARC). LoopViT introduces a recurrent mechanism into the Vision Transformer architecture to better solve iterative reasoning tasks.

## Structure

- `src/`: Core model definitions (`ARC_LoopViT`, `ARC_ViT`, etc.).
- `utils/`: Utility scripts for data loading, processing, and evaluation.
- `offline_train_loop_ARC.py`: Main script for offline training.
- `test_time_train_ARC.py`: Script for test-time training (TTT).

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
Ensure the ARC-AGI dataset is located in `raw_data/ARC-AGI` (or specify your path via arguments).

### Training 
To train the LoopViT model:

```bash
python offline_train_loop_ARC.py --architecture loop_vit --batch-size 32
```

### Test-Time Training (TTT)
To run test-time training on evaluation tasks:

```bash
python test_time_train_ARC.py --resume-checkpoint <path_to_checkpoint>
```

## Models
We provide several model variants:
- **LoopViT**: The primary recurrent architecture.
- **LoopViT Variable**: Variant with variable loop depth.
- **ARC_ViT / ViT1 / ViT2**: Baseline Vision Transformer implementations.
