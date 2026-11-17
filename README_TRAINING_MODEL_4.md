# Model 4 — Training and Checkpoint Resume Guide

> [!NOTE]
> **Model Overview**:
> **Model 4** is a single-view 2D sagittal Pfirrmann classifier for lumbar MRI discs.
> - **Backbone**: Pretrained MRI-CORE ViT-B (blocks 0–11 frozen under `no_grad` to preserve generic MRI representations).
> - **SAM Neck**: Trainable 1x1 + 3x3 Conv layers (768 → 256, initial learning rate `1e-5`).
> - **Spatial Pooling**: `SpatialAttentionPooling` learning 2D spatial attention maps over 64×64 token grids.
> - **Classification Head**: 2-layer MLP with BatchNorm1d and GELU (learning rate `1e-4`).
> - **Loss Function**: Inverse-frequency balanced CrossEntropyLoss.

---

## 1. Key File Locations in Server

All file paths in the remote server environment:

| Component | Server Path |
| :--- | :--- |
| **Main Project Root** | `/home/arindam/spider-pfirrmann-mri-core` |
| **Training Script** | `/home/arindam/spider-pfirrmann-mri-core/training/train_classifier_v2.py` |
| **Training Configuration File** | `/home/arindam/spider-pfirrmann-mri-core/configs/training_config_model_4.py` |
| **Classifier Model Definition** | `/home/arindam/spider-pfirrmann-mri-core/training/models/pfirrmann_classifier_v2.py` |
| **Dataset CSV** | `/home/arindam/spider-pfirrmann-mri-core/dataset_OG.csv` *(or `data/new_final/final_data/dataset_OG.csv`)* |
| **MRI Images Directory** | `/home/arindam/spider-pfirrmann-mri-core/data/new_final/final_data/images` |
| **Checkpoints Directory** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_4/checkpoints/` |
| **Best Model Checkpoint** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_4/checkpoints/best_model.pt` |
| **Latest Epoch Checkpoint** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_4/checkpoints/latest_checkpoint.pt` |
| **Training Log CSV** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_4/logs/training_log.csv` |

---

## 2. How and Where to Change Configuration

The primary configuration file for Model 4 is:
```
configs/training_config_model_4.py
```

### What Can Be Changed Before Resuming Training:

You can edit these parameters in `configs/training_config_model_4.py` before running the training script:

1. **`NUM_EPOCHS`** (Default: `50`):
   - To extend training beyond the previous run, increase this number (e.g., change `NUM_EPOCHS = 50` to `NUM_EPOCHS = 70`).
   - *Alternatively, pass `--epochs 70` on the command line without editing the file.*

2. **`BATCH_SIZE`** (Default: `6`):
   - Controls the batch size per forward pass.
   - 6 fits comfortably on a 16 GB GPU (NVIDIA T4). If running on a GPU with 24 GB+ VRAM, you can test `BATCH_SIZE = 8` or `12`.
   - If running on smaller GPUs (e.g. 6 GB / 8 GB), reduce to `BATCH_SIZE = 2` or `4`.

3. **Differential Learning Rates**:
   - `NECK_LEARNING_RATE = 1e-5`: Learning rate for the unfrozen SAM neck conv layers.
   - `HEAD_LEARNING_RATE = 1e-4`: Learning rate for spatial attention pooling and classifier MLP.
   - `WEIGHT_DECAY = 1e-4`: AdamW L2 weight regularization.
   - `SCHEDULER_ETA_MIN = 1e-6`: Floor learning rate for CosineAnnealingLR.

4. **Class Weighting & Loss**:
   - `USE_CLASS_WEIGHTS = True`: Enables inverse-frequency class weights computed automatically across the training set to combat Pfirrmann grade imbalance.
   - `CLASS_WEIGHT_METHOD = "inverse_frequency"`: Can be kept as `inverse_frequency` or set to `smoothed` (square-root inverse frequency).

5. **Early Stopping & Checkpoint Metric**:
   - `PRIMARY_METRIC = "val_macro_f1"`: Metric used to determine if a new checkpoint beats the current best model.
   - `EARLY_STOPPING_PATIENCE = 15`: Number of consecutive validation evaluations without F1 improvement before stopping training early.

6. **Hardware & Workers**:
   - `NUM_WORKERS = 4`: Number of asynchronous DataLoader subprocesses.
   - `PIN_MEMORY = True`: Keeps memory page-locked for fast CUDA transfers.

---

## 3. Guide: How to Resume Training

Activate your virtual environment and execute from the repository root:

```bash
cd /home/arindam/spider-pfirrmann-mri-core
source .venv/bin/activate
```

### Method 1: Automatic Auto-Resume (Easiest & Recommended)
If `outputs/model_4/checkpoints/latest_checkpoint.pt` exists, running the script with **no arguments** will automatically detect it:
```bash
python training/train_classifier_v2.py
```
**What happens automatically**:
- Detects `latest_checkpoint.pt`.
- Restores all model weights, optimizer states, scheduler state, current epoch number, best validation macro F1, and early stopping counter.
- Resumes training seamlessly from `start_epoch + 1` up to `NUM_EPOCHS`.
- Appends new epoch evaluations to `outputs/model_4/logs/training_log.csv`.

---

### Method 2: Resume with Increased Total Epochs via CLI
If the model was previously trained up to Epoch 50 and you want to continue training for an additional 20 epochs (to Epoch 70) without modifying `configs/training_config_model_4.py`:
```bash
python training/train_classifier_v2.py --epochs 70
```

---

### Method 3: Resume from a Specific Checkpoint File
If you want to resume training specifically from the best model or an earlier checkpoint:
```bash
python training/train_classifier_v2.py --resume_checkpoint outputs/model_4/checkpoints/best_model.pt --epochs 70
```

---

### Method 4: Force a Fresh Training Run (Epoch 1)
If you want to disregard all existing checkpoints and restart training from scratch:
```bash
python training/train_classifier_v2.py --fresh
```
*(Caution: This will start a new training run from Epoch 1).*

---

## 4. Checkpoint & Logging Mechanics

During continued training, two checkpoints and one log file are updated:

1. **`best_model.pt`**:
   - Saved into `outputs/model_4/checkpoints/best_model.pt`.
   - Only overwritten when the validation macro F1 score (`val_macro_f1`) strictly exceeds the previous best score.

2. **`latest_checkpoint.pt`**:
   - Saved into `outputs/model_4/checkpoints/latest_checkpoint.pt`.
   - Overwritten at the end of **every single epoch**. Contains full state dicts (model, optimizer, scheduler, epoch, metrics, early stopping counter) for fail-safe resumption if the server is interrupted.

3. **`training_log.csv`**:
   - Saved into `outputs/model_4/logs/training_log.csv`.
   - Automatically preserved upon resume: new epoch rows are appended without overwriting historical epochs.
   - Tracks: `Epoch`, `Train_Loss`, `Val_Loss`, `Val_Accuracy`, `Val_Macro_F1`, `Val_Weighted_F1`, `LR_Neck`, `LR_Head`, and per-grade F1 scores (`G1_F1` to `G5_F1`).
