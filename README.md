# SPIDER Pfirrmann MRI Core

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Foundation Model](https://img.shields.io/badge/Foundation%20Model-MRI--CORE-blue.svg)](https://github.com/mazurowski-lab/mri_foundation)

A research-oriented deep learning framework for automated, fine-grained **Pfirrmann grading (Grades 1 through 5)** of lumbar intervertebral disc degeneration from sagittal T2-weighted spinal MRI scans (SPIDER benchmark), leveraging the **MRI-CORE** foundation model.

---

## 1. Project Overview & Model Hierarchy

Accurate radiological grading of lumbar disc degeneration requires capturing subtle tissue changes—from juvenile hyperintense hydration (Grade 1) to total disc space collapse and signal void (Grade 5). 

This repository implements two production-grade architectures built upon the pretrained **MRI-CORE** Vision Transformer (ViT-B):

| Dimension | **Model 4 (Single-View 2D)** | **Model 5 (Multi-View 3D Volumetric)** |
| :--- | :--- | :--- |
| **Input Format** | Single 2D sagittal slice crop ($1024 \times 1024$) | 5 Contiguous sagittal slices ($[-2, -1, 0, +1, +2]$) |
| **Training Pipeline** | `training/train_classifier_v2.py` | `training/train_classifier_v7.py` |
| **Neural Classifier** | `training/models/pfirrmann_classifier_v2.py` | `training/models/pfirrmann_classifier_v5.py` |
| **Classifier Class** | `PfirrmannClassifierV2` | `PfirrmannClassifierV5` |
| **Inference Pipeline**| `evaluation/model_4/inference_model_4.py` | `evaluation/model_5/inference_model_5.py` |
| **Config File** | `configs/training_config_model_4.py` | `configs/training_config_model_5.py` |
| **Dataset Manifest** | `dataset_OG.csv` (7,003 slice crops, 204 patients) | `dataset_multi_view_v2.csv` (1,412 volumes, 204 patients) |
| **SAM Neck Policy** | **Unfrozen & Trainable** ($\eta = 1\times 10^{-5}$) | **Unfrozen & Trainable** ($\eta = 1\times 10^{-5}$) |
| **Aggregation / Pooling**| `SpatialAttentionPooling` (2D token attention) | `SpatialAttentionPooling` + `ViewAttentionAggregator` |
| **Positional Embed** | $64\times 64$ 2D Interpolated pos_embed | 2D pos_embed + Learned 1D View Positional Vectors |
| **Output Supervision**| Single classification head (Balanced CE) | **Dual Heads**: Primary Multi-View + Auxiliary Center |
| **Loss Objective** | Inverse-Frequency Weighted Cross-Entropy | `SoftOrdinalFocalLoss` ($\sigma=0.70, \gamma=2.0$) + Aux ($\lambda=0.30$) |
| **Exact Accuracy** | **63.34%** (463 / 731) | **54.36%** (81 / 149) |
| **Clinically Acceptable**<br>*($\le 1$ Grade Error)* | **96.03%** (702 / 731) | **97.32%** (145 / 149) |
| **Non-Adjacent Errors** | 3.97% (29 / 731) | **2.68%** (4 / 149, **0 errors of distance $\ge 3$**) |

> [!NOTE]
> **Lineage Context**:
> Earlier experimental iterations (**Models 1, 2, and 3**) were excluded from this repository because those architectures kept the SAM neck **frozen**, imposing a severe bottleneck that prevented adaptation to local spine radiomics. **Models 4 and 5** resolve this by unfreezing the neck with differential optimization and domain-specific attention mechanisms.

---

## 2. Model Architectures

### 2.1. Model 4: Single-View 2D Pfirrmann Classifier
- **Script**: `training/train_classifier_v2.py`
- **Architecture**: `training/models/pfirrmann_classifier_v2.py` (`PfirrmannClassifierV2`)
- **Key Mechanics**:
  - **Frozen Backbone**: Pretrained MRI-CORE ViT-B (blocks 0–11 frozen under `torch.no_grad()`).
  - **Trainable SAM Neck**: $1\times 1$ Conv ($768 \to 256$) + $3\times 3$ Conv ($256 \to 256$) with LayerNorm.
  - **Spatial Attention Pooling**: Dynamically learns a 2D Softmax attention mask over 4,096 tokens, isolating the nucleus pulposus from extraneous vertebral tissue.
  - **Classifier Head**: 2-layer MLP ($256 \to 128 \to 5$) with BatchNorm1d, GELU, and Dropout (0.2).
  - **Optimization**: Two-group AdamW (Neck: $1\times 10^{-5}$, Head: $1\times 10^{-4}$) with CosineAnnealingLR.

### 2.2. Model 5: Multi-View 3D Volumetric Pfirrmann Classifier
- **Script**: `training/train_classifier_v7.py`
- **Architecture**: `training/models/pfirrmann_classifier_v5.py` (`PfirrmannClassifierV5`)
- **Key Mechanics**:
  - **Volumetric Input**: 5 contiguous sagittal MRI crops (offsets $[-2, -1, 0, +1, +2]$) processed in mini-batches to keep peak VRAM $< 3.0$ GB.
  - **View Positional Embeddings**: Learned 1D embeddings assigned to each slice offset, preserving lateral-to-medial spatial geometry.
  - **View Attention Aggregator**: Self-attention mechanism learning cross-view interdependencies across all 5 slices.
  - **Dual Output Heads**: Primary multi-view head supervised via `SoftOrdinalFocalLoss` ($\sigma = 0.70, \gamma = 2.0$) and an auxiliary head supervising the mid-sagittal center slice directly ($\lambda = 0.30$).
  - **Clinical Safety**: Restricts non-adjacent errors to **2.68%** with zero multi-grade catastrophic errors.

---

## 3. Repository Structure

```
spider-pfirrmann-mri-core/
├── configs/
│   ├── paths.py                                  # Centralized filesystem path configurations
│   ├── training_config_model_4.py                # Hyperparameter specification for Model 4
│   └── training_config_model_5.py                # Hyperparameter specification for Model 5
│
├── training/
│   ├── models/
│   │   ├── pfirrmann_classifier_v2.py            # Model 4: Single-View 2D Classifier architecture
│   │   └── pfirrmann_classifier_v5.py            # Model 5: Multi-View 3D Classifier architecture
│   ├── dataset.py                                # PyTorch Dataset implementation for 2D single slices
│   ├── test_dataset.py                           # Dataset unit tests and sanity checks
│   ├── losses.py                                 # SoftOrdinalFocalLoss implementation
│   ├── train_classifier_v2.py                    # Model 4 end-to-end training pipeline
│   └── train_classifier_v7.py                    # Model 5 end-to-end multi-view training pipeline
│
├── evaluation/
│   ├── model_4/
│   │   ├── inference_model_4.py                  # Model 4 unified inference & reporting pipeline
│   │   └── results/                              # Output target for Model 4 evaluations
│   └── model_5/
│       ├── inference_model_5.py                  # Model 5 unified inference & reporting pipeline
│       └── results/                              # Output target for Model 5 evaluations
│
├── mri_core/                                     # MRI-CORE foundation model source package
│   ├── models/                                   # ViT-B & SAM transformer definitions
│   └── cfg.py                                    # MRI-CORE argument parser and config utilities
│
├── docs/                                         # Technical documentation and evaluation reports
│
├── README_INFERENCE_MODEL_4.md                   # Dedicated mentor execution guide for Model 4 inference
├── README_INFERENCE_MODEL_5.md                   # Dedicated mentor execution guide for Model 5 inference
├── README_TRAINING_MODEL_4.md                    # Dedicated mentor guide for Model 4 continued training
├── README_TRAINING_MODEL_5.md                    # Dedicated mentor guide for Model 5 continued training
├── requirements.txt                              # Pinned Python package dependencies
└── README.md                                     # Main project README
```

---

## 4. Quickstart: Installation & Inference Execution

### 4.1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/rajatarindam/spider-pfirrmann-mri-core.git
cd spider-pfirrmann-mri-core

# Create and activate Python virtual environment
python -m venv .venv
. .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

### 4.2. Running Model 4 Inference (Single-View 2D)

Executes test inference across all 731 slices on the best checkpoint, generating metrics CSVs, all 731 visual montages, 31 core evaluation figures (`r/`), and 10 confidence distribution bar charts (`conf/`):

```bash
python evaluation/model_4/inference_model_4.py
```
*Outputs are saved to `evaluation/model_4/results/`.*  
*(For advanced CLI flags like `--checkpoint latest_checkpoint.pt` or `--skip_visuals`, refer to [`README_INFERENCE_MODEL_4.md`](README_INFERENCE_MODEL_4.md)).*

### 4.3. Running Model 5 Inference (Multi-View 3D)

Executes multi-view test inference across all 149 3D disc volumes on the best checkpoint, generating metrics CSVs, 149 panoramic 5-slice composite montages, 31 core evaluation figures (`r/`), and 10 confidence distribution bar charts (`conf/`):

```bash
python evaluation/model_5/inference_model_5.py
```
*Outputs are saved to `evaluation/model_5/results/`.*  
*(For advanced CLI flags and batch options, refer to [`README_INFERENCE_MODEL_5.md`](README_INFERENCE_MODEL_5.md)).*

---

## 5. Continued Training & Checkpoint Resumption

Both training pipelines feature native crash recovery and auto-resume capabilities:

### Resuming Model 4:
```bash
# Auto-resumes from outputs/model_4/checkpoints/latest_checkpoint.pt
python training/train_classifier_v2.py

# Or extend total training epochs:
python training/train_classifier_v2.py --epochs 70
```
*(See [`README_TRAINING_MODEL_4.md`](README_TRAINING_MODEL_4.md) for full configuration modification options).*

### Resuming Model 5:
```bash
# Auto-resumes from outputs/model_5/phase1_2/checkpoints/latest_checkpoint.pt
python training/train_classifier_v7.py

# Or extend total training epochs:
python training/train_classifier_v7.py --epochs 70
```
*(See [`README_TRAINING_MODEL_5.md`](README_TRAINING_MODEL_5.md) for full configuration modification options).*

---

## 6. Citations & Acknowledgments

This research is built upon the **MRI-CORE** foundation model developed by the Mazurowski Lab at Duke University. If you utilize this codebase, please cite the original foundation model paper:

```bibtex
@article{mri_core_2025,
  title   = {MRI-CORE: A Foundation Model for Magnetic Resonance Imaging},
  author  = {Mazurowski, Maciej A. and collaborators},
  journal = {arXiv preprint arXiv:2506.12186},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.12186}
}
```

- **MRI-CORE Paper**: [arXiv:2506.12186](https://arxiv.org/abs/2506.12186)
- **MRI-CORE Repository**: [mazurowski-lab/mri_foundation](https://github.com/mazurowski-lab/mri_foundation)
- **Dataset Benchmark**: SPIDER Lumbar Spine MRI Challenge (Lumbar Intervertebral Disc Pfirrmann Grading).
