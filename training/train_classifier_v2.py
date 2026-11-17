"""
train_classifier_v2.py
──────────────────────
Model 4 Modular Training Script: Single-View 2D Pfirrmann Classifier.

Architecture:
    - Backbone: Pretrained MRI-CORE ViT-B (blocks 0–11 frozen under no_grad)
    - Positional Embedding: Corrected 64x64 interpolated pos_embed
    - SAM Neck: Unfrozen & trainable conv layers (768 -> 256, lr = 1e-5)
    - Pooling: SpatialAttentionPooling(256)
    - Classifier Head: 2-layer MLP with BatchNorm1d and GELU (lr = 1e-4)
    - Optimizer: 2-group AdamW (neck vs. head) with CosineAnnealingLR
    - Loss: Class-weighted CrossEntropyLoss (inverse-frequency balanced)

Usage:
    # 1. Standard auto-resume (automatically detects latest checkpoint and resumes):
    python training/train_classifier_v2.py

    # 2. Change total epochs:
    #    Option A: Change NUM_EPOCHS directly in configs/training_config_model_4.py
    #    Option B: Pass CLI flag:
    python training/train_classifier_v2.py --epochs 60

    # 3. Start completely fresh from Epoch 1 (ignoring existing checkpoints):
    python training/train_classifier_v2.py --fresh

    # 4. Resume from a specific checkpoint file:
    python training/train_classifier_v2.py --resume_checkpoint path/to/checkpoint.pt
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

_SCRIPT_DIR = Path(__file__).resolve().parent

# Check potential project roots
candidates = [
    _SCRIPT_DIR.parent,                # If run from <root>/training/ -> <root>
    _SCRIPT_DIR.parent.parent,         # If run from <root>/context/training/ -> <root>
    _SCRIPT_DIR,
]

PROJECT_ROOT = _SCRIPT_DIR.parent
for c in candidates:
    if (c / "data").exists() or (c / "mri_core").exists():
        PROJECT_ROOT = c
        break

for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "context"), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Model import
try:
    from training.models.pfirrmann_classifier_v2 import PfirrmannClassifierV2
except ImportError:
    try:
        from model_4.training.models.pfirrmann_classifier_v2 import PfirrmannClassifierV2
    except ImportError:
        from models.pfirrmann_classifier_v2 import PfirrmannClassifierV2

# Configuration imports
try:
    from configs.training_config_model_4 import (
        NUM_CLASSES, CLASS_NAMES, GRADE_TO_LABEL, LABEL_TO_GRADE,
        EXPECTED_SPLIT_COUNTS, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS,
        PIN_MEMORY, RANDOM_SEED, IMAGE_SIZE,
        NECK_LEARNING_RATE, HEAD_LEARNING_RATE, WEIGHT_DECAY,
        USE_LR_SCHEDULER, SCHEDULER_ETA_MIN,
        USE_CLASS_WEIGHTS, CLASS_WEIGHT_METHOD,
        PRIMARY_METRIC, SAVE_BEST_CHECKPOINT, SAVE_LATEST_CHECKPOINT,
    )
except ImportError:
    from training_config_model_4 import *

# Path imports
from configs.paths import (
    MODEL_4_DATASET_CSV,
    MODEL_4_IMAGES_DIR,
    MODEL_4_OUTPUTS_DIR,
    MODEL_4_CHECKPOINT_DIR,
    MODEL_4_LOGS_DIR,
    MODEL_4_BEST_MODEL_PATH,
    MODEL_4_LATEST_CHECKPOINT_PATH,
    MODEL_4_TRAINING_LOG_PATH,
)

DATASET_CSV = MODEL_4_DATASET_CSV
IMAGES_DIR = MODEL_4_IMAGES_DIR
OUTPUT_DIR = MODEL_4_OUTPUTS_DIR
CHECKPOINT_DIR = MODEL_4_CHECKPOINT_DIR
LOG_DIR = MODEL_4_LOGS_DIR

BEST_MODEL_PATH = MODEL_4_BEST_MODEL_PATH
LATEST_CHECKPOINT_PATH = MODEL_4_LATEST_CHECKPOINT_PATH
TRAINING_LOG_PATH = MODEL_4_TRAINING_LOG_PATH


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
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA unavailable. Training will run on CPU.")
    return torch.device("cpu")


def get_latest_checkpoint_path() -> Path | None:
    """Find the latest checkpoint across standard checkpoint directories."""
    candidates = [
        LATEST_CHECKPOINT_PATH,
        PROJECT_ROOT / "outputs" / "model_4" / "checkpoints" / "latest_checkpoint.pt",
        PROJECT_ROOT / "outputs" / "model_4_28th_epoch_model" / "latest_checkpoint.pt",
        _SCRIPT_DIR.parent / "outputs" / "model_4" / "checkpoints" / "latest_checkpoint.pt",
        _SCRIPT_DIR.parent / "outputs" / "model_4_28th_epoch_model" / "latest_checkpoint.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


# =============================================================================
# DATASET
# =============================================================================

class PfirrmannDatasetModel4(Dataset):
    """
    Single-disc 2D Pfirrmann grading dataset for Model 4.

    CSV:
        data/new_final/final_data/dataset_OG.csv

    Image:
        1024 x 1024 grayscale PNG -> [3, 1024, 1024] float32 tensor in [0, 1]
    Label:
        Integer class index 0-4
    """

    def __init__(self, csv_path: Path = DATASET_CSV, split: str | None = None):
        super().__init__()
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            alt = PROJECT_ROOT / "data" / "new_final" / "final_data" / "dataset_OG.csv"
            if alt.exists():
                self.csv_path = alt
            else:
                raise FileNotFoundError(f"Dataset CSV not found:\n{self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_columns = ["MRI_Output_Path", "Pfirrmann_Grade", "Split"]
        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(f"Required column '{column}' not found in dataset CSV.")

        if split is not None:
            split = split.lower()
            valid_splits = {"train", "val", "test"}
            if split not in valid_splits:
                raise ValueError(f"Invalid split '{split}'. Expected one of {valid_splits}.")

            self.df = self.df[
                self.df["Split"].astype(str).str.lower() == split
            ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split: {split}")

        grades = pd.to_numeric(self.df["Pfirrmann_Grade"], errors="coerce")
        if grades.isna().any():
            raise ValueError("Dataset contains invalid Pfirrmann grades.")

        invalid_grades = sorted(set(grades.astype(int)) - set(GRADE_TO_LABEL.keys()))
        if invalid_grades:
            raise ValueError(f"Invalid Pfirrmann grades found: {invalid_grades}")

        self.df["Pfirrmann_Grade"] = grades.astype(int)

        print(
            f"PfirrmannDatasetModel4 | "
            f"Split: {split or 'all':5s} | "
            f"Samples: {len(self.df)}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, relative_path: str) -> Path:
        rel = str(relative_path).strip()
        candidates = [
            PROJECT_ROOT / rel,
            PROJECT_ROOT / "context" / rel,
            Path("D:/BAAA project/spider-pfirrmann-mri-core") / rel,
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Image not found at any candidate path for:\n{rel}")

    def _load_image(self, relative_path: str) -> torch.Tensor:
        image_path = self._resolve_image_path(relative_path)
        image = Image.open(image_path).convert("L")

        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, resample=Image.BILINEAR)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = np.stack([arr, arr, arr], axis=0)
        return torch.from_numpy(tensor)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        image = self._load_image(row["MRI_Output_Path"])
        grade = int(row["Pfirrmann_Grade"])
        label = torch.tensor(GRADE_TO_LABEL[grade], dtype=torch.long)
        return image, label


# =============================================================================
# DATA LOADERS & DATASET VALIDATION
# =============================================================================

def create_dataloaders(batch_size: int = BATCH_SIZE) -> tuple:
    train_dataset = PfirrmannDatasetModel4(csv_path=DATASET_CSV, split="train")
    val_dataset   = PfirrmannDatasetModel4(csv_path=DATASET_CSV, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def validate_dataset_splits() -> None:
    if not DATASET_CSV.exists():
        alt = PROJECT_ROOT / "data" / "new_final" / "final_data" / "dataset_OG.csv"
        if not alt.exists():
            raise FileNotFoundError(f"Dataset CSV not found:\n{DATASET_CSV}")
        csv_file = alt
    else:
        csv_file = DATASET_CSV

    df = pd.read_csv(csv_file)
    if "Split" not in df.columns:
        raise ValueError("Dataset CSV does not contain the required 'Split' column.")

    split_counts = df["Split"].astype(str).str.lower().value_counts().to_dict()

    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        actual = int(split_counts.get(split, 0))
        expected = EXPECTED_SPLIT_COUNTS[split]
        status = "OK" if actual == expected else "MISMATCH"
        print(f"  {split.capitalize():<10} Expected: {expected:<8} Found: {actual:<8} {status}")

        if actual != expected:
            raise ValueError(f"Unexpected {split} split size. Expected {expected}, found {actual}.")

    print("=" * 70 + "\n")


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def calculate_class_weights(train_dataset: PfirrmannDatasetModel4, device: torch.device) -> torch.Tensor:
    labels = train_dataset.df["Pfirrmann_Grade"].astype(int).to_numpy()
    class_indices = np.array([GRADE_TO_LABEL[int(grade)] for grade in labels], dtype=np.int64)

    counts = np.bincount(class_indices, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        missing = (np.where(counts == 0)[0] + 1).tolist()
        raise ValueError(f"Training split contains no samples for grade(s): {missing}")

    total_samples = len(class_indices)
    weights = total_samples / (NUM_CLASSES * counts.astype(np.float64))

    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("\n" + "=" * 70)
    print("CLASS WEIGHTS (Calculated from TRAIN split only)")
    print("=" * 70)
    print(f"  {'Grade':<10}{'Train Samples':<16}{'Weight':<12}")
    print("  " + "-" * 38)
    for grade in range(1, NUM_CLASSES + 1):
        idx = GRADE_TO_LABEL[grade]
        print(f"  {grade:<10}{counts[idx]:<16}{weights[idx]:.6f}")
    print("=" * 70 + "\n")

    return class_weights


# =============================================================================
# MODEL, OPTIMIZER & SCHEDULER
# =============================================================================

def create_model(device: torch.device) -> PfirrmannClassifierV2:
    orig_argv = list(sys.argv)
    sys.argv = [orig_argv[0]]
    try:
        model = PfirrmannClassifierV2().to(device)
    finally:
        sys.argv = orig_argv
    return model


def create_optimizer(model: PfirrmannClassifierV2) -> AdamW:
    neck_params = [p for p in model.image_encoder.neck.parameters() if p.requires_grad]

    if hasattr(model, "spatial_attention"):
        spatial_module = model.spatial_attention
    elif hasattr(model, "spatial_attention_pool"):
        spatial_module = model.spatial_attention_pool
    else:
        raise AttributeError("PfirrmannClassifierV2 has no 'spatial_attention' or 'spatial_attention_pool'.")

    head_params = [p for p in spatial_module.parameters() if p.requires_grad]
    head_params.extend([p for p in model.classifier.parameters() if p.requires_grad])

    if not neck_params:
        raise RuntimeError("No trainable neck parameters were found.")
    if not head_params:
        raise RuntimeError("No trainable head parameters were found.")

    optimizer = AdamW(
        [
            {"params": neck_params, "lr": NECK_LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
            {"params": head_params, "lr": HEAD_LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        ]
    )
    return optimizer


def create_scheduler(optimizer: AdamW, epochs: int = NUM_EPOCHS) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=SCHEDULER_ETA_MIN,
    )


# =============================================================================
# TRAIN & VALIDATE LOOPS
# =============================================================================

def train_one_epoch(
    model: PfirrmannClassifierV2,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    total = 0
    all_predictions = []
    all_labels = []

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", dynamic_ncols=True)

    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_sz = labels.size(0)
        running_loss += loss.item() * batch_sz
        total += batch_sz

        preds = torch.argmax(logits, dim=1)
        all_predictions.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

        progress.set_postfix(loss=f"{running_loss / total:.4f}")

    train_loss = running_loss / total
    train_acc  = float(np.mean(np.array(all_predictions) == np.array(all_labels)))
    train_f1   = f1_score(
        all_labels, all_predictions,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    return train_loss, train_acc, train_f1


def validate_one_epoch(
    model: PfirrmannClassifierV2,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_predictions = []
    all_labels = []

    progress = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", dynamic_ncols=True)

    with torch.no_grad():
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_sz = labels.size(0)
            running_loss += loss.item() * batch_sz
            total += batch_sz

            preds = torch.argmax(logits, dim=1)
            all_predictions.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

            progress.set_postfix(loss=f"{running_loss / total:.4f}")

    val_loss = running_loss / total
    val_acc  = float(np.mean(np.array(all_predictions) == np.array(all_labels)))
    val_f1   = f1_score(
        all_labels, all_predictions,
        average="macro",
        labels=list(range(NUM_CLASSES)),
        zero_division=0,
    )
    return val_loss, val_acc, val_f1


# =============================================================================
# CHECKPOINT & LOGGING HELPERS
# =============================================================================

def save_checkpoint(
    path: Path,
    model: PfirrmannClassifierV2,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    train_macro_f1: float,
    val_loss: float,
    val_accuracy: float,
    val_macro_f1: float,
    best_val_macro_f1: float,
    early_stopping_counter: int = 0,
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "train_macro_f1": train_macro_f1,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "best_val_macro_f1": best_val_macro_f1,
        "early_stopping_counter": early_stopping_counter,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    checkpoint_path: Path,
    model: PfirrmannClassifierV2,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
    device: torch.device,
) -> tuple[int, float, int]:
    print("\n" + "=" * 70)
    print("RESUMING MODEL 4 TRAINING")
    print("=" * 70)
    print(f"  Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_val_macro_f1",
    }
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise KeyError(f"Checkpoint is missing required keys: {sorted(missing_keys)}")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = int(checkpoint["epoch"])
    best_val_macro_f1 = float(checkpoint["best_val_macro_f1"])
    early_stopping_counter = int(checkpoint.get("early_stopping_counter", 0))

    print(f"  Resumed after epoch  : {start_epoch}")
    print(f"  Best validation F1    : {best_val_macro_f1 * 100:.2f}%")
    print(f"  Next training epoch   : {start_epoch + 1}")
    print("=" * 70 + "\n")

    return start_epoch, best_val_macro_f1, early_stopping_counter


def initialize_training_log() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if not TRAINING_LOG_PATH.exists():
        with open(TRAINING_LOG_PATH, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_macro_f1",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "lr_neck",
                "lr_head",
                "time_taken",
            ])


def append_training_log(
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    train_macro_f1: float,
    val_loss: float,
    val_accuracy: float,
    val_macro_f1: float,
    lr_neck: float,
    lr_head: float,
    time_taken: float,
) -> None:
    with open(TRAINING_LOG_PATH, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            epoch,
            f"{train_loss:.8f}",
            f"{train_accuracy:.8f}",
            f"{train_macro_f1:.8f}",
            f"{val_loss:.8f}",
            f"{val_accuracy:.8f}",
            f"{val_macro_f1:.8f}",
            f"{lr_neck:.10f}",
            f"{lr_head:.10f}",
            f"{time_taken:.4f}",
        ])


def print_parameter_summary(model: PfirrmannClassifierV2) -> None:
    trainable = 0
    frozen = 0
    print("\n" + "=" * 70)
    print("TRAINABLE PARAMETER SUMMARY")
    print("=" * 70)
    for name, p in model.named_parameters():
        count = p.numel()
        if p.requires_grad:
            trainable += count
            status = "TRAINABLE"
        else:
            frozen += count
            status = "FROZEN"
        print(f"  {status:<10} {count:>12,} {name}")
    print("-" * 70)
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  Frozen parameters    : {frozen:,}")
    print(f"  Total parameters     : {trainable + frozen:,}")
    print("=" * 70 + "\n")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model 4 for SPIDER Pfirrmann grading.")
    parser.add_argument("--resume", action="store_true", help=f"Resume from {LATEST_CHECKPOINT_PATH}")
    parser.add_argument("--resume_checkpoint", type=str, default="", help="Path to a specific checkpoint to resume from.")
    parser.add_argument("--fresh", action="store_true", help="Start fresh from epoch 1 (ignore existing checkpoints).")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Total epochs to train through (default from config: {NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Batch size (default from config: {BATCH_SIZE})")
    args = parser.parse_args()

    if args.fresh and (args.resume or args.resume_checkpoint):
        parser.error("Cannot combine --fresh with --resume or --resume_checkpoint.")

    set_seed(RANDOM_SEED)
    device = get_device()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    validate_dataset_splits()

    print("=" * 70)
    print("SPIDER PFIRRMANN TRAINING - MODEL 4 (MODULAR)")
    print("=" * 70)
    print(f"  Project root     : {PROJECT_ROOT}")
    print(f"  Dataset CSV      : {DATASET_CSV}")
    print(f"  Images directory : {IMAGES_DIR}")
    print(f"  Checkpoints      : {CHECKPOINT_DIR}")
    print(f"  Logs             : {LOG_DIR}")
    print(f"  Device           : {device}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Target epochs    : {args.epochs}")
    print(f"  Neck LR          : {NECK_LEARNING_RATE}")
    print(f"  Head LR          : {HEAD_LEARNING_RATE}")
    print(f"  Weight decay     : {WEIGHT_DECAY}")
    print(f"  Scheduler        : CosineAnnealingLR (eta_min={SCHEDULER_ETA_MIN})")
    print("=" * 70 + "\n")

    # 1. Dataloaders
    train_dataset, val_dataset, train_loader, val_loader = create_dataloaders(batch_size=args.batch_size)

    # 2. Class weights & Criterion
    class_weights = calculate_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model
    model = create_model(device)
    print_parameter_summary(model)

    # 4. Optimizer & Scheduler
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, epochs=args.epochs)

    # 5. Checkpoint Resume Logic (Auto-Resume by default if checkpoint exists)
    start_epoch = 0
    best_val_macro_f1 = float("-inf")
    early_stopping_counter = 0

    latest_ckpt = get_latest_checkpoint_path()

    if args.fresh:
        print("[FRESH RUN] Starting cleanly from Epoch 1 (ignoring existing checkpoints).\n")
    elif args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found:\n{ckpt_path}")
        start_epoch, best_val_macro_f1, early_stopping_counter = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, device
        )
    elif args.resume or (latest_ckpt is not None):
        if latest_ckpt is not None:
            print(f"[AUTO-RESUME] Found existing checkpoint: {latest_ckpt}")
            start_epoch, best_val_macro_f1, early_stopping_counter = load_checkpoint(
                latest_ckpt, model, optimizer, scheduler, device
            )
        elif args.resume:
            raise FileNotFoundError(f"Resume requested, but latest checkpoint not found:\n{LATEST_CHECKPOINT_PATH}")
    else:
        print("[FRESH RUN] No previous checkpoint found. Starting from Epoch 1.\n")

    # 6. Initialize log file
    initialize_training_log()

    # 7. Check if already trained through target epochs
    if start_epoch >= args.epochs:
        print(f"Checkpoint is already at epoch {start_epoch}, which is >= target epochs ({args.epochs}).")
        print(f"Nothing to train. If you want to train further, either:\n"
              f"  1. Change NUM_EPOCHS in configs/training_config_model_4.py to a higher number (e.g. {start_epoch + 10})\n"
              f"  2. Or run: python training/train_classifier_v2.py --epochs {start_epoch + 10}\n")
        return

    # 8. Training loop
    print(f"Starting training from epoch {start_epoch + 1} to {args.epochs}...\n")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device, epoch
        )

        lr_neck = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        time_taken = time.time() - epoch_start_time

        # Check if best model
        is_best = val_f1 > best_val_macro_f1
        if is_best:
            best_val_macro_f1 = val_f1
            save_checkpoint(
                BEST_MODEL_PATH, model, optimizer, scheduler, epoch,
                train_loss, train_acc, train_f1,
                val_loss, val_acc, val_f1,
                best_val_macro_f1, early_stopping_counter,
            )
            print(f"  --> [BEST MODEL SAVED] val_macro_f1={best_val_macro_f1:.4f} saved to {BEST_MODEL_PATH.name}")

        # Step scheduler
        scheduler.step()

        # Save latest checkpoint (always updated)
        save_checkpoint(
            LATEST_CHECKPOINT_PATH, model, optimizer, scheduler, epoch,
            train_loss, train_acc, train_f1,
            val_loss, val_acc, val_f1,
            best_val_macro_f1, early_stopping_counter,
        )

        # Append to CSV log
        append_training_log(
            epoch, train_loss, train_acc, train_f1,
            val_loss, val_acc, val_f1,
            lr_neck, lr_head, time_taken,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc * 100:.2f}%  F1: {train_f1 * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc * 100:.2f}%  F1: {val_f1 * 100:.2f}% | "
            f"Best F1: {best_val_macro_f1 * 100:.2f}% | Time: {time_taken:.1f}s"
        )

    print("\nTraining Complete.")
    print(f"Best Validation Macro F1 : {best_val_macro_f1 * 100:.2f}%")
    print(f"Best Model Saved To      : {BEST_MODEL_PATH}")
    print(f"Latest Checkpoint Saved  : {LATEST_CHECKPOINT_PATH}")
    print(f"Training Log Saved To    : {TRAINING_LOG_PATH}\n")


if __name__ == "__main__":
    main()
