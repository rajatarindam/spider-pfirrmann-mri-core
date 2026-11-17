# Model 5 — Multi-View Training and Checkpoint Resume Guide

> [!NOTE]
> **Model Overview**:
> **Model 5** is a multi-view 3D disc volume classifier processing **5 contiguous sagittal slices** (offsets: `[-2, -1, 0, +1, +2]`) per lumbar disc.
> - **Backbone**: Pretrained MRI-CORE ViT-B (blocks 0–11 frozen under `no_grad`, processing slices in mini-batches).
> - **SAM Neck**: Trainable conv layers (768 → 256, initial learning rate `1e-5`).
> - **Spatial Pooling**: `SpatialAttentionPooling` replacing uniform average pooling.
> - **View Embeddings**: Learned 1D view positional embeddings assigned to each slice offset.
> - **View Attention Aggregator**: Self-attention mechanism learning cross-slice interaction weights.
> - **Dual Heads**:
>   - Primary Classifier Head: Operates on fused 5-view volume representations.
>   - Auxiliary Center Head: Supervised directly on the anatomical center slice (`Offset 0`, weight $\lambda = 0.30$).
> - **Loss Function**: `SoftOrdinalFocalLoss` combining Gaussian-smoothed ordinal targets ($\sigma = 0.70$) with focal hard-example modulating factors ($\gamma = 2.0$).

---

## 1. Key File Locations in Server

All file paths in the remote server environment:

| Component | Server Path |
| :--- | :--- |
| **Main Project Root** | `/home/arindam/spider-pfirrmann-mri-core` |
| **Model 5 Training Script** | `/home/arindam/spider-pfirrmann-mri-core/training/train_classifier_v7.py` |
| **Training Configuration File** | `/home/arindam/spider-pfirrmann-mri-core/configs/training_config_model_5.py` |
| **Classifier Model Definition** | `/home/arindam/spider-pfirrmann-mri-core/training/models/pfirrmann_classifier_v5.py` |
| **Dataset CSV** | `/home/arindam/spider-pfirrmann-mri-core/data/new_final/dataset_multi_view_v2.csv` *(or `data/new_final/final_data/dataset_multi_view_v2.csv`)* |
| **MRI Images Directory** | `/home/arindam/spider-pfirrmann-mri-core/data/new_final/final_data/images` |
| **Checkpoints Directory** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/checkpoints/` |
| **Best Model Checkpoint** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/checkpoints/best_model.pt` |
| **Latest Epoch Checkpoint** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/checkpoints/latest_checkpoint.pt` |
| **Periodic Checkpoints** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/checkpoints/checkpoint_epoch_*.pt` |
| **Training Log CSV** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/logs/training_log.csv` |

---

## 2. How and Where to Change Configuration

The primary configuration file for Model 5 is:
```
configs/training_config_model_5.py
```

### What Can Be Changed Before Resuming Training:

You can edit these parameters in `configs/training_config_model_5.py` before running the training script:

1. **`NUM_EPOCHS`** (Default: `50`):
   - To continue training past 50 epochs, increase this number (e.g., change `NUM_EPOCHS = 50` to `NUM_EPOCHS = 70`).
   - *Alternatively, pass `--epochs 70` on the command line without editing the file.*

2. **`BATCH_SIZE`** (Default: `4`):
   - Number of disc volumes per batch. Each volume contains 5 slices (so `BATCH_SIZE = 4` processes 20 slices per iteration).
   - If running on smaller GPUs, reduce to `BATCH_SIZE = 2` or `1`.

3. **Differential Learning Rates & Weight Decay**:
   - `LEARNING_RATE_NECK = 1e-5`: Learning rate for SAM neck conv layers.
   - `LEARNING_RATE_HEAD = 1e-4`: Learning rate for spatial attention pooling, view attention aggregator, and dual classification heads.
   - `WEIGHT_DECAY = 1e-3`: Regularization penalty (set slightly higher for the neck to prevent overfitting on 980 training disc volumes).

4. **Warm Restarts Scheduler (`CosineAnnealingWarmRestarts`)**:
   - `SCHED_T_0 = 15`: Initial cycle length in epochs.
   - `SCHED_T_MULT = 2`: Multiplier for subsequent cycle lengths (Cycle 1: 15 epochs, Cycle 2: 30 epochs).
   - `SCHED_ETA_MIN_HEAD = 1e-5` / `SCHED_ETA_MIN_NECK = 1e-6`: Floor learning rates.

5. **Soft Ordinal Focal Loss & Auxiliary Head**:
   - `ORDINAL_SIGMA = 0.70`: Gaussian spread parameter for smoothing discrete labels into ordinal probability distributions.
   - `FOCAL_GAMMA = 2.0`: Modulating factor focusing gradient updates on difficult, borderline Pfirrmann grades.
   - `USE_AUX_CENTER_LOSS = True`: Enables auxiliary mid-sagittal supervision.
   - `AUX_CENTER_LOSS_WEIGHT = 0.30`: Weight factor ($\lambda = 0.30$) for auxiliary center-slice cross-entropy loss.

6. **Gradient Regularization & Checkpoint Period**:
   - `GRAD_CLIP_MAX_NORM = 0.5`: Tight gradient clipping threshold to stabilize multi-slice attention gradients.
   - `CHECKPOINT_PERIOD = 10`: Saves a permanent snapshot every 10 epochs (`checkpoint_epoch_010.pt`, `checkpoint_epoch_020.pt`, etc.) to prevent storage bloat while maintaining recovery points.

---

## 3. Guide: How to Resume Training

Activate your virtual environment and execute from the repository root:

```bash
cd /home/arindam/spider-pfirrmann-mri-core
source .venv/bin/activate
```

### Method 1: Automatic Auto-Resume (Easiest & Recommended)
If `outputs/model_5/phase1_2/checkpoints/latest_checkpoint.pt` exists, running the script with **no arguments** will automatically detect it:
```bash
python training/train_classifier_v7.py
```
**What happens automatically**:
- Detects `latest_checkpoint.pt` in `outputs/model_5/phase1_2/checkpoints/`.
- Restores all model weights, optimizer state, scheduler state, current epoch number, and historical best validation macro F1.
- Resumes training seamlessly from `start_epoch + 1` up to `NUM_EPOCHS`.
- Appends new epoch evaluations to `outputs/model_5/phase1_2/logs/training_log.csv`.

---

### Method 2: Resume with Increased Total Epochs via CLI
If the model was previously trained up to Epoch 50 and you want to continue training for an additional 20 epochs (to Epoch 70) without modifying `configs/training_config_model_5.py`:
```bash
python training/train_classifier_v7.py --epochs 70
```

---

### Method 3: Resume from a Specific Checkpoint File
If you want to resume training specifically from the best model or a specific periodic snapshot:
```bash
python training/train_classifier_v7.py --resume_checkpoint outputs/model_5/phase1_2/checkpoints/best_model.pt --epochs 70
```

---

### Method 4: Force a Fresh Training Run (Epoch 1)
If you want to disregard all existing checkpoints and restart Phase 1+2 training from scratch:
```bash
python training/train_classifier_v7.py --fresh
```
*(Caution: This will start a new training run from Epoch 1).*

---

## 4. Checkpoint & Logging Mechanics

During continued training, three types of checkpoints and one log file are managed under `outputs/model_5/phase1_2/`:

1. **`best_model.pt`**:
   - Saved into `outputs/model_5/phase1_2/checkpoints/best_model.pt`.
   - Only updated when validation macro F1 (`val_macro_f1`) strictly surpasses the previous best score.

2. **`latest_checkpoint.pt`**:
   - Saved into `outputs/model_5/phase1_2/checkpoints/latest_checkpoint.pt`.
   - Overwritten at the end of **every single epoch**. Contains full model state, optimizer state, scheduler state, current epoch number, and metric history for instant crash recovery.

3. **`checkpoint_epoch_*.pt`**:
   - Periodic snapshot saved every `CHECKPOINT_PERIOD` epochs (e.g. `checkpoint_epoch_010.pt`, `checkpoint_epoch_020.pt`).

4. **`training_log.csv`**:
   - Saved into `outputs/model_5/phase1_2/logs/training_log.csv`.
   - Automatically preserved upon resume: new epoch rows are appended without overwriting previous epochs.
   - Tracks: `Epoch`, `Train_Loss`, `Val_Loss`, `Val_Accuracy`, `Val_Macro_F1`, `Val_Weighted_F1`, `LR_Neck`, `LR_Head`, and per-grade F1 scores (`G1_F1` to `G5_F1`).
