"""
train_classifier_v6.py
──────────────────────
Model 5 Training Script — Multi-View Pfirrmann Grading.

Key features:
    - Multi-view dataset (5 contiguous MRI slices per disc)
    - Training-only augmentation: random HFlip, brightness jitter, Gaussian noise
    - Class-weighted CrossEntropyLoss with label_smoothing=0.1
    - Single AdamW optimizer group at constant lr=1e-4 (no scheduler)
    - Gradient clipping (max_norm=1.0) before every optimizer step
    - Best model checkpoint saved by val_macro_f1 (primary metric)
    - Latest checkpoint saved every epoch (full resume support)
    - Per-epoch checkpoint for rollback flexibility
    - CSV training log appended after every epoch

Usage:
    python training/train_classifier_v6.py
    python training/train_classifier_v6.py --resume
    python training/train_classifier_v6.py --resume_checkpoint path/to/ckpt.pt
    python training/train_classifier_v6.py --epochs 60
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================
# PROJECT ROOT & IMPORT RESOLUTION
# =============================================================================
# Works whether this script is run from the project root or from training/.

_SCRIPT_DIR = Path(__file__).resolve().parent  # …/training/

if (_SCRIPT_DIR.parent / "configs").exists():
    PROJECT_ROOT = _SCRIPT_DIR.parent          # standard: root/training/
elif (_SCRIPT_DIR / "configs").exists():
    PROJECT_ROOT = _SCRIPT_DIR
else:
    PROJECT_ROOT = _SCRIPT_DIR.parent

for _p in (str(PROJECT_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import classifier
try:
    from training.models.pfirrmann_classifier_v4 import PfirrmannClassifierV4
except ImportError:
    try:
        from models.pfirrmann_classifier_v4 import PfirrmannClassifierV4
    except ImportError:
        raise ImportError(
            "Cannot import PfirrmannClassifierV4. "
            "Ensure pfirrmann_classifier_v4.py is in training/models/."
        )

# Import training config
try:
    from configs.training_config_model_5 import (
        NUM_CLASSES, CLASS_NAMES, OFFSETS, NUM_VIEWS,
        GRADE_TO_LABEL, LABEL_TO_GRADE,
        BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, PIN_MEMORY, RANDOM_SEED, IMAGE_SIZE,
        LEARNING_RATE, WEIGHT_DECAY,
        USE_CLASS_WEIGHTS, LABEL_SMOOTHING,
        GRAD_CLIP_MAX_NORM,
        USE_AUGMENTATION, AUG_HFLIP_PROB, AUG_BRIGHTNESS_DELTA, AUG_NOISE_STD,
        PRIMARY_METRIC, SAVE_BEST_CHECKPOINT, SAVE_LATEST_CHECKPOINT,
    )
except ImportError:
    try:
        from training.training_config_model_5 import (
            NUM_CLASSES, CLASS_NAMES, OFFSETS, NUM_VIEWS,
            GRADE_TO_LABEL, LABEL_TO_GRADE,
            BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, PIN_MEMORY, RANDOM_SEED, IMAGE_SIZE,
            LEARNING_RATE, WEIGHT_DECAY,
            USE_CLASS_WEIGHTS, LABEL_SMOOTHING,
            GRAD_CLIP_MAX_NORM,
            USE_AUGMENTATION, AUG_HFLIP_PROB, AUG_BRIGHTNESS_DELTA, AUG_NOISE_STD,
            PRIMARY_METRIC, SAVE_BEST_CHECKPOINT, SAVE_LATEST_CHECKPOINT,
        )
    except ImportError:
        try:
            from training_config_model_5 import (
                NUM_CLASSES, CLASS_NAMES, OFFSETS, NUM_VIEWS,
                GRADE_TO_LABEL, LABEL_TO_GRADE,
                BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, PIN_MEMORY, RANDOM_SEED, IMAGE_SIZE,
                LEARNING_RATE, WEIGHT_DECAY,
                USE_CLASS_WEIGHTS, LABEL_SMOOTHING,
                GRAD_CLIP_MAX_NORM,
                USE_AUGMENTATION, AUG_HFLIP_PROB, AUG_BRIGHTNESS_DELTA, AUG_NOISE_STD,
                PRIMARY_METRIC, SAVE_BEST_CHECKPOINT, SAVE_LATEST_CHECKPOINT,
            )
        except ImportError:
            raise ImportError(
                "Cannot import training_config_model_5. "
                "Ensure training_config_model_5.py is in configs/ or training/."
            )

# Import paths
from configs.paths import MODEL_5_DATASET_CSV, MODEL_5_CHECKPOINT_DIR, MODEL_5_LOGS_DIR


# =============================================================================
# OUTPUT PATHS
# =============================================================================

BEST_MODEL_PATH      = MODEL_5_CHECKPOINT_DIR / "best_model.pt"
LATEST_CKPT_PATH     = MODEL_5_CHECKPOINT_DIR / "latest_checkpoint.pt"
TRAINING_LOG_PATH    = MODEL_5_LOGS_DIR       / "training_log.csv"


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
    print("WARNING: CUDA unavailable. Training will run on CPU (very slow).")
    return torch.device("cpu")


# =============================================================================
# DATASET
# =============================================================================

class PfirrmannMultiViewDataset(Dataset):
    """
    Multi-view Pfirrmann grading dataset for Model 5.

    Each sample is one disc group: 5 MRI slices at offsets [-2,-1,0,+1,+2].

    With dataset_multi_view_v2.csv all 5 offset columns are guaranteed to
    contain valid paths (missing slots filled by nearest-neighbour duplication
    at CSV generation time). The mask is kept for backward-compatibility.

    Augmentation is applied per-slice in training mode only:
        - Random horizontal flip (p=AUG_HFLIP_PROB)
        - Brightness jitter ± AUG_BRIGHTNESS_DELTA
        - Gaussian noise (σ = AUG_NOISE_STD)
    """

    OFFSET_COLS = [f"Image_{off}" for off in OFFSETS]

    def __init__(
        self,
        csv_path: Path = MODEL_5_DATASET_CSV,
        split: str | None = None,
        training: bool = False,
    ):
        super().__init__()
        self.csv_path  = Path(csv_path)
        self.training  = training

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {self.csv_path}\n"
                "Upload dataset_multi_view_v2.csv to the expected location."
            )

        self.df = pd.read_csv(self.csv_path).fillna("")

        # Validate required columns
        for col in ["Patient_ID", "IVD_Label", "Pfirrmann_Grade", "Split"] + self.OFFSET_COLS:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' missing from {self.csv_path}.")

        # Filter by split
        if split is not None:
            self.df = self.df[
                self.df["Split"].astype(str).str.lower() == split.lower()
            ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split='{split}'.")

        # Validate grades
        grades = pd.to_numeric(self.df["Pfirrmann_Grade"], errors="coerce")
        if grades.isna().any():
            raise ValueError("Dataset contains invalid or missing Pfirrmann grades.")
        self.df["Pfirrmann_Grade"] = grades.astype(int)

        print(
            f"PfirrmannMultiViewDataset | split={split or 'all':5s} | "
            f"discs={len(self.df)} | training={training}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, relative_path: str) -> tuple[torch.Tensor, bool]:
        """
        Load a single MRI slice image.

        Returns:
            (tensor [3, H, W], is_valid: bool)
        """
        path_str = str(relative_path).strip()
        if not path_str or path_str.lower() in ("nan", "none", ""):
            # Should not happen with v2 CSV, but handled defensively
            return torch.zeros((3, *IMAGE_SIZE), dtype=torch.float32), False

        image_path = PROJECT_ROOT / path_str
        if not image_path.exists():
            print(f"WARNING: Image not found on disk, padding: {image_path}")
            return torch.zeros((3, *IMAGE_SIZE), dtype=torch.float32), False

        img = Image.open(image_path).convert("L")
        if img.size != IMAGE_SIZE:
            img = img.resize(IMAGE_SIZE, resample=Image.BILINEAR)

        arr = np.asarray(img, dtype=np.uint8).astype(np.float32) / 255.0  # [H, W] ∈ [0,1]

        # ── Training augmentation (applied to grayscale arr before stacking) ──
        if self.training and USE_AUGMENTATION:
            # Random horizontal flip
            if random.random() < AUG_HFLIP_PROB:
                arr = np.fliplr(arr).copy()

            # Brightness jitter
            delta = random.uniform(-AUG_BRIGHTNESS_DELTA, AUG_BRIGHTNESS_DELTA)
            arr   = np.clip(arr + delta, 0.0, 1.0)

            # Gaussian noise
            noise = np.random.normal(0.0, AUG_NOISE_STD, arr.shape).astype(np.float32)
            arr   = np.clip(arr + noise, 0.0, 1.0)

        # Stack to 3-channel [3, H, W] for ViT backbone
        tensor = torch.from_numpy(np.stack([arr, arr, arr], axis=0))
        return tensor, True

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        images, mask = [], []
        for col in self.OFFSET_COLS:
            img_tensor, is_valid = self._load_image(row[col])
            images.append(img_tensor)
            mask.append(is_valid)

        images = torch.stack(images, dim=0)          # [5, 3, H, W]
        mask   = torch.tensor(mask, dtype=torch.bool) # [5]

        grade = int(row["Pfirrmann_Grade"])
        label = torch.tensor(GRADE_TO_LABEL[grade], dtype=torch.long)

        return images, mask, label


# =============================================================================
# DATA LOADERS
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


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def calculate_class_weights(
    train_dataset: PfirrmannMultiViewDataset,
    device: torch.device,
) -> torch.Tensor:
    labels  = train_dataset.df["Pfirrmann_Grade"].astype(int).to_numpy()
    indices = np.array([GRADE_TO_LABEL[g] for g in labels], dtype=np.int64)

    counts = np.bincount(indices, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        missing = [i + 1 for i in np.where(counts == 0)[0]]
        raise ValueError(f"Training split is missing Pfirrmann grade(s): {missing}")

    total   = len(indices)
    weights = total / (NUM_CLASSES * counts.astype(np.float64))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print()
    print("=" * 60)
    print("CLASS WEIGHTS (computed from training split)")
    print("=" * 60)
    for grade in range(1, NUM_CLASSES + 1):
        idx = GRADE_TO_LABEL[grade]
        print(f"  Grade {grade}: {counts[idx]:4d} discs  |  weight: {weights[idx]:.4f}")
    print("=" * 60)
    print()

    return class_weights


# =============================================================================
# OPTIMIZER
# =============================================================================

def create_optimizer(model: PfirrmannClassifierV4) -> AdamW:
    """
    Single parameter group at constant lr=1e-4.

    Only freshly initialised parameters (projection, view_pos_embed,
    view_aggregator, classifier) are included.  The frozen image_encoder
    is excluded implicitly via requires_grad=False.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found in the model.")

    return AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# =============================================================================
# TRAIN & VALIDATION LOOPS
# =============================================================================

def train_one_epoch(
    model: PfirrmannClassifierV4,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    total        = 0
    all_preds    = []
    all_labels   = []

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", dynamic_ncols=True)

    for images, mask, labels in progress:
        images = images.to(device, non_blocking=True)   # [B, 5, 3, H, W]
        mask   = mask.to(device,   non_blocking=True)   # [B, 5]
        labels = labels.to(device, non_blocking=True)   # [B]

        optimizer.zero_grad(set_to_none=True)

        logits = model(images, mask=mask)               # [B, 5]
        loss   = criterion(logits, labels)

        loss.backward()

        # Gradient clipping — essential for B=1 stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)

        optimizer.step()

        batch_sz      = labels.size(0)
        running_loss += loss.item() * batch_sz
        total        += batch_sz

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        progress.set_postfix(loss=f"{running_loss / total:.4f}")

    train_loss = running_loss / total
    train_acc  = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    train_f1   = f1_score(
        all_labels, all_preds,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    return train_loss, train_acc, train_f1


def validate_one_epoch(
    model: PfirrmannClassifierV4,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float]:
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

            logits = model(images, mask=mask)
            loss   = criterion(logits, labels)

            batch_sz      = labels.size(0)
            running_loss += loss.item() * batch_sz
            total        += batch_sz

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            progress.set_postfix(loss=f"{running_loss / total:.4f}")

    val_loss = running_loss / total
    val_acc  = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    val_f1   = f1_score(
        all_labels, all_preds,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )

    return val_loss, val_acc, val_f1


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def save_checkpoint(state: dict, is_best: bool, epoch: int) -> None:
    MODEL_5_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-epoch snapshot
    epoch_path = MODEL_5_CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, epoch_path)

    # Latest checkpoint (for --resume)
    torch.save(state, LATEST_CKPT_PATH)

    # Best model (by val_macro_f1)
    if is_best:
        torch.save(state, BEST_MODEL_PATH)
        print(
            f"  --> New best model saved  "
            f"(val_macro_f1={state['best_metrics']['val_macro_f1']:.4f})"
        )


def load_checkpoint(
    checkpoint_path: Path,
    model: PfirrmannClassifierV4,
    optimizer: AdamW,
) -> tuple[int, dict]:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_epoch  = ckpt["epoch"] + 1
    best_metrics = ckpt.get("best_metrics", {"val_accuracy": 0.0, "val_macro_f1": 0.0})
    print(f"  Resumed at epoch {start_epoch}  "
          f"| best val_macro_f1 so far: {best_metrics['val_macro_f1']:.4f}")
    return start_epoch, best_metrics


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Model 5: Multi-View Pfirrmann Classifier"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest_checkpoint.pt",
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, default="",
        help="Path to a specific checkpoint file to resume from",
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help=f"Total training epochs (default: {NUM_EPOCHS})",
    )
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    device = get_device()

    print("\n" + "=" * 70)
    print("MODEL 5 — MULTI-VIEW PFIRRMANN CLASSIFIER TRAINING")
    print(f"  Project root  : {PROJECT_ROOT}")
    print(f"  Dataset CSV   : {MODEL_5_DATASET_CSV}")
    print(f"  Checkpoints   : {MODEL_5_CHECKPOINT_DIR}")
    print(f"  Logs          : {MODEL_5_LOGS_DIR}")
    print(f"  Device        : {device}")
    print(f"  LR            : {LEARNING_RATE}  (constant, no scheduler)")
    print(f"  Label smooth  : {LABEL_SMOOTHING}")
    print(f"  Grad clip     : {GRAD_CLIP_MAX_NORM}")
    print(f"  Augmentation  : {USE_AUGMENTATION}")
    print("=" * 70 + "\n")

    # ── Dataloaders ────────────────────────────────────────────────────────────
    train_ds, val_ds, train_loader, val_loader = create_dataloaders()

    # ── Model ──────────────────────────────────────────────────────────────────
    model = PfirrmannClassifierV4(num_views=NUM_VIEWS, num_classes=NUM_CLASSES).to(device)
    model.print_summary()

    # ── Loss ───────────────────────────────────────────────────────────────────
    if USE_CLASS_WEIGHTS:
        class_weights = calculate_class_weights(train_ds, device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING,
    )

    # ── Optimizer ──────────────────────────────────────────────────────────────
    optimizer = create_optimizer(model)

    start_epoch  = 1
    best_metrics = {"val_accuracy": 0.0, "val_macro_f1": 0.0}

    # ── Resume ─────────────────────────────────────────────────────────────────
    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        start_epoch, best_metrics = load_checkpoint(ckpt_path, model, optimizer)
    elif args.resume:
        if LATEST_CKPT_PATH.exists():
            start_epoch, best_metrics = load_checkpoint(LATEST_CKPT_PATH, model, optimizer)
        else:
            print(f"No checkpoint at {LATEST_CKPT_PATH}. Starting from epoch 1.")

    # ── CSV log initialisation ─────────────────────────────────────────────────
    MODEL_5_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_exists = TRAINING_LOG_PATH.exists()

    if not log_exists or (start_epoch == 1 and not args.resume and not args.resume_checkpoint):
        with open(TRAINING_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_accuracy", "train_macro_f1",
                "val_loss",   "val_accuracy",   "val_macro_f1",
                "lr",
                "time_taken",
            ])

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}...\n")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        epoch_time = time.time() - epoch_start
        lr_current = optimizer.param_groups[0]["lr"]

        # Determine if this is the best checkpoint
        is_best = val_f1 > best_metrics["val_macro_f1"]
        if is_best:
            best_metrics["val_accuracy"]  = val_acc
            best_metrics["val_macro_f1"]  = val_f1

        # Append to CSV log
        with open(TRAINING_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.8f}", f"{train_acc:.8f}", f"{train_f1:.8f}",
                f"{val_loss:.8f}",   f"{val_acc:.8f}",   f"{val_f1:.8f}",
                f"{lr_current:.10f}",
                f"{epoch_time:.4f}",
            ])

        # Save checkpoints
        checkpoint_state = {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metrics":         best_metrics,
        }
        save_checkpoint(checkpoint_state, is_best, epoch)

        # Epoch summary line
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f} | "
            f"Best Val F1: {best_metrics['val_macro_f1']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

    print("\nTraining complete.")
    print(f"Best val_macro_f1 : {best_metrics['val_macro_f1']:.4f}")
    print(f"Best val_accuracy : {best_metrics['val_accuracy']:.4f}")
    print(f"Best model saved  : {BEST_MODEL_PATH}")
    print(f"Training log      : {TRAINING_LOG_PATH}")


if __name__ == "__main__":
    main()
