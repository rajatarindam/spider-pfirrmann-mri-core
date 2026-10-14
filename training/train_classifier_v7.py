"""
train_classifier_v7.py
──────────────────────
Model 5 Phase 1 + Phase 2 Combined Training Script — Multi-View Pfirrmann Grading.

Key Enhancements vs v6:
1. Architecture:
   - Uses PfirrmannClassifierV5 with restored SAM Neck and SpatialAttentionPooling.
   - Dual output heads: primary multi-view fused logits + auxiliary center slice logits.
2. Loss Function:
   - SoftOrdinalFocalLoss: Gaussian soft ordinal target encoding (sigma=0.70)
     + Focal loss (gamma=2.0) + Class weights + Auxiliary center slice supervision (0.3x).
3. Optimization:
   - 2-group AdamW: SAM Neck at lr=1e-5, Heads/Attention at lr=1e-4.
   - CosineAnnealingWarmRestarts scheduler (T_0=10, T_mult=2).
   - Tighter gradient clipping at max_norm=0.5.
4. Data Augmentation:
   - Coherent multi-slice geometric transforms (HFlip, rotation +-10 deg, scaling 0.92-1.08)
     applied consistently across all 5 slices of a disc group.
   - Independent radiometric jitter (brightness +-20%, contrast, noise, cutout).

Usage:
    python model_5/training/train_classifier_v7.py
    python model_5/training/train_classifier_v7.py --resume
    python model_5/training/train_classifier_v7.py --epochs 50
"""

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# =============================================================================
# PROJECT ROOT & IMPORT RESOLUTION
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent

# Check potential project roots
candidates = [
    _SCRIPT_DIR.parent.parent,         # .../context/
    _SCRIPT_DIR.parent.parent.parent,  # .../spider-pfirrmann-mri-core/
    _SCRIPT_DIR.parent,
    _SCRIPT_DIR,
]

PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for c in candidates:
    if (c / "data").exists() or (c / "mri_core").exists():
        PROJECT_ROOT = c
        break

for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "context"), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Model, Loss, and Config imports
try:
    from model_5.training.models.pfirrmann_classifier_v5 import PfirrmannClassifierV5
    from model_5.training.losses import SoftOrdinalFocalLoss
except ImportError:
    try:
        from training.models.pfirrmann_classifier_v5 import PfirrmannClassifierV5
        from training.losses import SoftOrdinalFocalLoss
    except ImportError:
        from models.pfirrmann_classifier_v5 import PfirrmannClassifierV5
        from losses import SoftOrdinalFocalLoss

try:
    from configs.training_config_model_5 import (
        NUM_CLASSES, CLASS_NAMES, OFFSETS, NUM_VIEWS,
        GRADE_TO_LABEL, LABEL_TO_GRADE,
        BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, PIN_MEMORY, RANDOM_SEED, IMAGE_SIZE,
        LEARNING_RATE_NECK, LEARNING_RATE_HEAD, WEIGHT_DECAY,
        USE_LR_SCHEDULER, SCHED_T_0, SCHED_T_MULT, SCHED_ETA_MIN_HEAD, SCHED_ETA_MIN_NECK,
        USE_CLASS_WEIGHTS, LOSS_TYPE, ORDINAL_SIGMA, FOCAL_GAMMA,
        USE_AUX_CENTER_LOSS, AUX_CENTER_LOSS_WEIGHT,
        GRAD_CLIP_MAX_NORM,
        USE_AUGMENTATION, AUG_HFLIP_PROB, AUG_ROTATION_DEG, AUG_ROTATION_PROB,
        AUG_SCALE_MIN, AUG_SCALE_MAX, AUG_SCALE_PROB,
        AUG_BRIGHTNESS_DELTA, AUG_CONTRAST_RANGE, AUG_CONTRAST_PROB,
        AUG_NOISE_STD, AUG_ERASING_PROB, AUG_ERASING_SCALE,
        PRIMARY_METRIC, SAVE_BEST_CHECKPOINT, SAVE_LATEST_CHECKPOINT,
    )
except ImportError:
    from training_config_model_5 import *

from configs.paths import MODEL_5_DATASET_CSV, MODEL_5_CHECKPOINT_DIR, MODEL_5_LOGS_DIR

# Dedicated checkpoint and log directory for Phase 1+2 run
PHASE_OUTPUT_DIR = MODEL_5_CHECKPOINT_DIR.parent / "phase1_2"
PHASE_CKPT_DIR = PHASE_OUTPUT_DIR / "checkpoints"
PHASE_LOGS_DIR = PHASE_OUTPUT_DIR / "logs"

BEST_MODEL_PATH   = PHASE_CKPT_DIR / "best_model.pt"
LATEST_CKPT_PATH  = PHASE_CKPT_DIR / "latest_checkpoint.pt"
TRAINING_LOG_PATH = PHASE_LOGS_DIR / "training_log.csv"


# =============================================================================
# REPRODUCIBILITY & DEVICE
# =============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA unavailable. Training will run on CPU.")
    return torch.device("cpu")


# =============================================================================
# DATASET WITH COHERENT MULTI-SLICE AUGMENTATION
# =============================================================================

class PfirrmannMultiViewDataset(Dataset):
    """
    Multi-view Pfirrmann dataset with Coherent 3D Multi-Slice Augmentation.

    Coherent geometric transforms (HFlip, rotation, scaling) are applied
    consistently across all 5 slices of a disc group. Radiometric variations
    (brightness, contrast, noise, cutout) are applied per-slice.
    """

    OFFSET_COLS = [f"Image_{off}" for off in OFFSETS]

    def __init__(
        self,
        csv_path: Path = MODEL_5_DATASET_CSV,
        split: str | None = None,
        training: bool = False,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.training = training

        if not self.csv_path.exists():
            # Check fallback in context or repo root
            alt = PROJECT_ROOT / "data" / "new_final" / "final_data" / "dataset_multi_view_v2.csv"
            if alt.exists():
                self.csv_path = alt
            else:
                alt2 = Path("D:/BAAA project/spider-pfirrmann-mri-core/data/new_final/final_data/dataset_multi_view_v2.csv")
                if alt2.exists():
                    self.csv_path = alt2
                else:
                    raise FileNotFoundError(f"Dataset CSV not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path).fillna("")

        # Validate columns
        for col in ["Patient_ID", "IVD_Label", "Pfirrmann_Grade", "Split"] + self.OFFSET_COLS:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' missing from {self.csv_path}.")

        # Split filter
        if split is not None:
            self.df = self.df[
                self.df["Split"].astype(str).str.lower() == split.lower()
            ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}'.")

        self.df["Pfirrmann_Grade"] = self.df["Pfirrmann_Grade"].astype(int)

        print(
            f"PfirrmannMultiViewDataset | split={split or 'all':5s} | "
            f"discs={len(self.df)} | training={training}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, rel_path: str) -> Path | None:
        p = str(rel_path).strip()
        if not p or p.lower() in ("nan", "none", ""):
            return None
        # Check standard relative path
        img_path = PROJECT_ROOT / p
        if img_path.exists():
            return img_path
        # Check context prefix
        img_path = PROJECT_ROOT / "context" / p
        if img_path.exists():
            return img_path
        # Check repo root
        img_path = Path("D:/BAAA project/spider-pfirrmann-mri-core") / p
        if img_path.exists():
            return img_path
        return None

    def _apply_coherent_augmentations(self, slice_imgs: list[Image.Image]) -> list[np.ndarray]:
        """
        Applies coherent geometric transforms across all 5 slices of a disc group,
        followed by independent per-slice radiometric jitter.
        """
        # 1. Coherent Geometric Decisions
        do_hflip = random.random() < AUG_HFLIP_PROB
        do_rot   = random.random() < AUG_ROTATION_PROB
        rot_deg  = random.uniform(-AUG_ROTATION_DEG, AUG_ROTATION_DEG) if do_rot else 0.0

        do_scale = random.random() < AUG_SCALE_PROB
        scale    = random.uniform(AUG_SCALE_MIN, AUG_SCALE_MAX) if do_scale else 1.0

        transformed_slices = []

        for img in slice_imgs:
            # A. Geometric Transform (Coherent)
            if do_hflip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if do_rot or do_scale:
                w, h = img.size
                if do_rot:
                    img = img.rotate(rot_deg, resample=Image.BILINEAR)
                if do_scale:
                    new_w, new_h = max(10, int(w * scale)), max(10, int(h * scale))
                    img = img.resize((new_w, new_h), resample=Image.BILINEAR)
                    img = ImageOps.fit(img, (w, h), method=Image.BILINEAR)

            arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W] in [0, 1]

            # B. Radiometric Jitter (Independent per slice)
            # Brightness jitter
            delta_b = random.uniform(-AUG_BRIGHTNESS_DELTA, AUG_BRIGHTNESS_DELTA)
            arr = np.clip(arr + delta_b, 0.0, 1.0)

            # Contrast jitter
            if random.random() < AUG_CONTRAST_PROB:
                c_factor = random.uniform(*AUG_CONTRAST_RANGE)
                mean_val = arr.mean()
                arr = np.clip((arr - mean_val) * c_factor + mean_val, 0.0, 1.0)

            # Gaussian noise
            noise = np.random.normal(0.0, AUG_NOISE_STD, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)

            # Random Erasing / Cutout
            if random.random() < AUG_ERASING_PROB:
                h_pix, w_pix = arr.shape
                area = h_pix * w_pix
                erase_area = random.uniform(*AUG_ERASING_SCALE) * area
                aspect = random.uniform(0.5, 2.0)
                e_h = max(2, min(h_pix - 1, int(math.sqrt(erase_area * aspect))))
                e_w = max(2, min(w_pix - 1, int(math.sqrt(erase_area / aspect))))
                if e_h < h_pix and e_w < w_pix:
                    top = random.randint(0, h_pix - e_h)
                    left = random.randint(0, w_pix - e_w)
                    arr[top : top + e_h, left : left + e_w] = 0.0

            transformed_slices.append(arr)

        return transformed_slices

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        # Load raw PIL images
        raw_pil_slices = []
        mask = []

        for col in self.OFFSET_COLS:
            img_path = self._resolve_image_path(row[col])
            if img_path and img_path.exists():
                img = Image.open(img_path).convert("L")
                if img.size != IMAGE_SIZE:
                    img = img.resize(IMAGE_SIZE, resample=Image.BILINEAR)
                raw_pil_slices.append(img)
                mask.append(True)
            else:
                raw_pil_slices.append(Image.new("L", IMAGE_SIZE, 0))
                mask.append(False)

        # Apply augmentation if in training mode
        if self.training and USE_AUGMENTATION:
            processed_slices = self._apply_coherent_augmentations(raw_pil_slices)
        else:
            processed_slices = [
                np.asarray(img, dtype=np.float32) / 255.0 for img in raw_pil_slices
            ]

        # Convert to 3-channel [3, 1024, 1024] tensors for ViT backbone
        tensor_slices = [
            torch.from_numpy(np.stack([arr, arr, arr], axis=0))
            for arr in processed_slices
        ]

        images = torch.stack(tensor_slices, dim=0)          # [5, 3, 1024, 1024]
        mask   = torch.tensor(mask, dtype=torch.bool)        # [5]

        grade  = int(row["Pfirrmann_Grade"])
        label  = torch.tensor(GRADE_TO_LABEL[grade], dtype=torch.long)

        return images, mask, label


# =============================================================================
# DATA LOADERS & CLASS WEIGHTS
# =============================================================================

def create_dataloaders() -> tuple:
    train_ds = PfirrmannMultiViewDataset(split="train", training=True)
    val_ds   = PfirrmannMultiViewDataset(split="val",   training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return train_ds, val_ds, train_loader, val_loader


def calculate_class_weights(
    train_dataset: PfirrmannMultiViewDataset,
    device: torch.device,
) -> torch.Tensor:
    labels = train_dataset.df["Pfirrmann_Grade"].astype(int).to_numpy()
    indices = np.array([GRADE_TO_LABEL[g] for g in labels], dtype=np.int64)

    counts = np.bincount(indices, minlength=NUM_CLASSES)
    total = len(indices)
    weights = total / (NUM_CLASSES * counts.astype(np.float64))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("\n" + "=" * 65)
    print("CLASS WEIGHTS (Inverse-Frequency Balanced)")
    print("=" * 65)
    for grade in range(1, NUM_CLASSES + 1):
        idx = GRADE_TO_LABEL[grade]
        print(f"  Grade {grade}: {counts[idx]:4d} discs  |  weight: {weights[idx]:.4f}")
    print("=" * 65 + "\n")

    return class_weights


# =============================================================================
# OPTIMIZER & SCHEDULER SETUP
# =============================================================================

def create_optimizer_and_scheduler(
    model: PfirrmannClassifierV5,
) -> tuple[AdamW, CosineAnnealingWarmRestarts]:
    """
    Creates a 2-group AdamW optimizer:
      Group 0: Pretrained SAM Neck (lr=1e-5)
      Group 1: Spatial attention, view embeddings, classifier heads (lr=1e-4)
    """
    neck_params = [p for p in model.image_encoder.neck.parameters() if p.requires_grad]

    head_modules = [
        model.spatial_attention_pool,
        model.view_pos_embed,
        model.view_aggregator,
        model.classifier,
        model.aux_classifier,
    ]
    head_params = []
    for m in head_modules:
        head_params.extend([p for p in m.parameters() if p.requires_grad])

    optimizer = AdamW(
        [
            {"params": neck_params, "lr": LEARNING_RATE_NECK, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": LEARNING_RATE_HEAD, "weight_decay": WEIGHT_DECAY},
        ]
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=SCHED_T_0,
        T_mult=SCHED_T_MULT,
        eta_min=SCHED_ETA_MIN_NECK,
    )

    return optimizer, scheduler


# =============================================================================
# TRAIN & EVALUATION LOOPS
# =============================================================================

def train_one_epoch(
    model: PfirrmannClassifierV5,
    loader: DataLoader,
    criterion: SoftOrdinalFocalLoss,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float, float]:
    model.train()
    running_loss  = 0.0
    running_aux   = 0.0
    total         = 0
    all_preds     = []
    all_labels    = []

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", dynamic_ncols=True)

    for images, mask, labels in progress:
        images = images.to(device, non_blocking=True)  # [B, 5, 3, 1024, 1024]
        mask   = mask.to(device,   non_blocking=True)  # [B, 5]
        labels = labels.to(device, non_blocking=True)  # [B]

        optimizer.zero_grad(set_to_none=True)

        primary_logits, center_logits = model(images, mask=mask)
        loss, loss_dict = criterion(primary_logits, labels, center_logits)

        loss.backward()

        # Gradient clipping to 0.5 for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)

        optimizer.step()

        batch_sz      = labels.size(0)
        running_loss += loss.item() * batch_sz
        running_aux  += loss_dict["loss_center"] * batch_sz
        total        += batch_sz

        preds = torch.argmax(primary_logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            aux=f"{running_aux / total:.4f}",
        )

    train_loss = running_loss / total
    train_aux  = running_aux / total
    train_acc  = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    train_f1   = f1_score(
        all_labels, all_preds,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    return train_loss, train_aux, train_acc, train_f1


def validate_one_epoch(
    model: PfirrmannClassifierV5,
    loader: DataLoader,
    criterion: SoftOrdinalFocalLoss,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float, float, list[float]]:
    model.eval()
    running_loss = 0.0
    total        = 0
    all_preds    = []
    all_labels   = []

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", dynamic_ncols=True)

    with torch.no_grad():
        for images, mask, labels in progress:
            images = images.to(device, non_blocking=True)
            mask   = mask.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            primary_logits, center_logits = model(images, mask=mask)
            loss, _ = criterion(primary_logits, labels, center_logits)

            batch_sz      = labels.size(0)
            running_loss += loss.item() * batch_sz
            total        += batch_sz

            preds = torch.argmax(primary_logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            progress.set_postfix(loss=f"{running_loss / total:.4f}")

    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)

    val_loss = running_loss / total
    val_acc  = float(np.mean(preds_arr == labels_arr))
    val_f1   = f1_score(
        all_labels, all_preds,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    # Ordinal Mean Absolute Error (MAE in Pfirrmann grades)
    val_mae  = float(np.mean(np.abs(preds_arr - labels_arr)))

    # Per-class F1 breakdown
    per_class_f1 = f1_score(
        all_labels, all_preds,
        average=None,
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    ).tolist()

    return val_loss, val_acc, val_f1, val_mae, per_class_f1


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(state: dict, is_best: bool, epoch: int) -> None:
    PHASE_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-epoch snapshot
    epoch_path = PHASE_CKPT_DIR / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, epoch_path)

    # Latest checkpoint (for resume)
    torch.save(state, LATEST_CKPT_PATH)

    # Best model (by val_macro_f1)
    if is_best:
        torch.save(state, BEST_MODEL_PATH)
        print(
            f"  --> [BEST MODEL SAVED] "
            f"val_macro_f1={state['best_metrics']['val_macro_f1']:.4f} "
            f"val_accuracy={state['best_metrics']['val_accuracy']:.4f} "
            f"val_mae={state['best_metrics']['val_mae']:.4f}"
        )


def load_checkpoint(
    checkpoint_path: Path,
    model: PfirrmannClassifierV5,
    optimizer: AdamW,
    scheduler: CosineAnnealingWarmRestarts,
) -> tuple[int, dict]:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"] + 1
    best_metrics = ckpt.get("best_metrics", {"val_accuracy": 0.0, "val_macro_f1": 0.0, "val_mae": 99.0})
    print(
        f"  Resumed at epoch {start_epoch} | "
        f"Best val_macro_f1 so far: {best_metrics['val_macro_f1']:.4f}"
    )
    return start_epoch, best_metrics


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Model 5: Phase 1 + Phase 2 Pfirrmann Classifier"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest_checkpoint.pt")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Path to checkpoint")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Total epochs (default {NUM_EPOCHS})")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = get_device()

    print("\n" + "=" * 75)
    print("MODEL 5 (PHASE 1 + 2) — MULTI-VIEW PFIRRMANN CLASSIFIER")
    print(f"  Project root     : {PROJECT_ROOT}")
    print(f"  Dataset CSV      : {MODEL_5_DATASET_CSV}")
    print(f"  Checkpoints      : {PHASE_CKPT_DIR}")
    print(f"  Logs             : {PHASE_LOGS_DIR}")
    print(f"  Device           : {device}")
    print(f"  Neck LR          : {LEARNING_RATE_NECK}")
    print(f"  Head LR          : {LEARNING_RATE_HEAD}")
    print(f"  Scheduler        : CosineAnnealingWarmRestarts (T_0={SCHED_T_0}, T_mult={SCHED_T_MULT})")
    print(f"  Loss             : SoftOrdinalFocalLoss (sigma={ORDINAL_SIGMA}, gamma={FOCAL_GAMMA}, aux=0.30)")
    print(f"  Grad clip        : {GRAD_CLIP_MAX_NORM}")
    print("=" * 75 + "\n")

    # 1. Dataloaders
    train_ds, val_ds, train_loader, val_loader = create_dataloaders()

    # 2. Model
    model = PfirrmannClassifierV5(num_views=NUM_VIEWS, num_classes=NUM_CLASSES).to(device)
    model.print_summary()

    # 3. Loss function with class weights
    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(train_ds, device)
    else:
        class_weights = None

    criterion = SoftOrdinalFocalLoss(
        num_classes=NUM_CLASSES,
        sigma=ORDINAL_SIGMA,
        gamma=FOCAL_GAMMA,
        class_weights=class_weights,
        aux_weight=AUX_CENTER_LOSS_WEIGHT,
    ).to(device)

    # 4. Optimizer & Scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model)

    start_epoch = 1
    best_metrics = {"val_accuracy": 0.0, "val_macro_f1": 0.0, "val_mae": 99.0}

    # 5. Checkpoint Resume
    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        start_epoch, best_metrics = load_checkpoint(ckpt_path, model, optimizer, scheduler)
    elif LATEST_CKPT_PATH.exists():
        print(f"\n[AUTO-RESUME] Found existing checkpoint: {LATEST_CKPT_PATH}")
        start_epoch, best_metrics = load_checkpoint(LATEST_CKPT_PATH, model, optimizer, scheduler)
    else:
        print(f"\n[FRESH RUN] No previous checkpoint found. Starting fresh from epoch 1.")

    # 6. CSV Logging Setup
    PHASE_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAINING_LOG_PATH.exists() or start_epoch == 1:
        with open(TRAINING_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_aux_loss", "train_accuracy", "train_macro_f1",
                "val_loss",   "val_accuracy",   "val_macro_f1",   "val_mae",
                "lr_neck",    "lr_head",
                "time_taken",
                "val_f1_g1", "val_f1_g2", "val_f1_g3", "val_f1_g4", "val_f1_g5",
            ])

    # 7. Training Loop
    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_aux, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1, val_mae, per_class_f1 = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start
        lr_neck = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        # Check if best model
        is_best = val_f1 > best_metrics["val_macro_f1"]
        if is_best:
            best_metrics["val_accuracy"] = val_acc
            best_metrics["val_macro_f1"] = val_f1
            best_metrics["val_mae"]      = val_mae

        # Append to CSV log
        with open(TRAINING_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.8f}", f"{train_aux:.8f}", f"{train_acc:.8f}", f"{train_f1:.8f}",
                f"{val_loss:.8f}",   f"{val_acc:.8f}",   f"{val_f1:.8f}",   f"{val_mae:.8f}",
                f"{lr_neck:.10f}",   f"{lr_head:.10f}",
                f"{epoch_time:.2f}",
                *[f"{f:.4f}" for f in per_class_f1],
            ])

        # Checkpoint save
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metrics": best_metrics,
        }
        save_checkpoint(state, is_best, epoch)

        # Summary line
        f1_str = " | ".join([f"G{i+1}:{per_class_f1[i]:.2f}" for i in range(5)])
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}  MAE: {val_mae:.2f} | "
            f"Best F1: {best_metrics['val_macro_f1']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        print(f"           Per-Class Val F1: [{f1_str}] | LR Neck: {lr_neck:.1e} | LR Head: {lr_head:.1e}")

    print("\nTraining Complete.")
    print(f"Best Val Macro F1 : {best_metrics['val_macro_f1']:.4f}")
    print(f"Best Val Accuracy : {best_metrics['val_accuracy']:.4f}")
    print(f"Best Val MAE      : {best_metrics['val_mae']:.4f}")
    print(f"Best Model Saved  : {BEST_MODEL_PATH}")
    print(f"Log File Saved    : {TRAINING_LOG_PATH}")


if __name__ == "__main__":
    main()
