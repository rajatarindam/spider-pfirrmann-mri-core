"""
Model 4 training script for SPIDER Pfirrmann grading.

Single-file implementation:
    - Dataset logic
    - DataLoaders
    - Model construction
    - Differential learning rates
    - CosineAnnealingLR
    - Class-weighted CrossEntropyLoss
    - Training/validation
    - CSV logging
    - Best/latest checkpointing
    - Resume support

Run from the project root:

    python training/train_classifier_v2.py

Resume the default latest checkpoint:

    python training/train_classifier_v2.py --resume

Resume from a specific checkpoint:

    python training/train_classifier_v2.py \
        --resume_checkpoint /path/to/checkpoint.pt
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
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PROJECT IMPORT
# =============================================================================
# The verified Model 4 architecture remains in its own model file.
# The training setup itself is kept in this single training script.

from training.models.pfirrmann_classifier_v2 import PfirrmannClassifierV2


# =============================================================================
# PATHS
# =============================================================================

DATASET_CSV = (
    PROJECT_ROOT
    / "data"
    / "new_final"
    / "final_data"
    / "dataset_OG.csv"
)

IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "new_final"
    / "final_data"
    / "images"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_4"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "latest_checkpoint.pt"
TRAINING_LOG_PATH = LOG_DIR / "training_log.csv"


# =============================================================================
# CONFIGURATION
# =============================================================================

BATCH_SIZE = 6
NUM_WORKERS = 4
PIN_MEMORY = True

NUM_EPOCHS = 50

NECK_LEARNING_RATE = 1e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 5

SCHEDULER_ETA_MIN = 1e-6

RANDOM_SEED = 42

IMAGE_SIZE = (1024, 1024)

GRADE_TO_LABEL = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}

LABEL_TO_GRADE = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
}

EXPECTED_SPLIT_COUNTS = {
    "train": 4864,
    "val": 1408,
    "test": 731,
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DEVICE
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    print("WARNING: CUDA unavailable. Training will use CPU.")
    return torch.device("cpu")


# =============================================================================
# DATASET
# =============================================================================

class PfirrmannDatasetModel4(Dataset):
    """
    Single-disc Pfirrmann grading dataset for Model 4.

    CSV:
        data/new_final/final_data/dataset_OG.csv

    Image:
        1024 x 1024 grayscale PNG

    Output:
        image: [3, 1024, 1024] float32 tensor in [0, 1]
        label: integer class index 0-4
    """

    def __init__(self, csv_path=DATASET_CSV, split=None):
        super().__init__()

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found:\n{self.csv_path}"
            )

        self.df = pd.read_csv(self.csv_path)

        required_columns = [
            "MRI_Output_Path",
            "Pfirrmann_Grade",
            "Split",
        ]

        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(
                    f"Required column '{column}' not found in dataset CSV."
                )

        if split is not None:
            split = split.lower()

            valid_splits = {"train", "val", "test"}

            if split not in valid_splits:
                raise ValueError(
                    f"Invalid split '{split}'. "
                    f"Expected one of {valid_splits}."
                )

            self.df = self.df[
                self.df["Split"].astype(str).str.lower() == split
            ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split: {split}"
            )

        grades = pd.to_numeric(
            self.df["Pfirrmann_Grade"],
            errors="coerce",
        )

        if grades.isna().any():
            raise ValueError(
                "Dataset contains invalid Pfirrmann grades."
            )

        invalid_grades = sorted(
            set(grades.astype(int))
            - set(GRADE_TO_LABEL.keys())
        )

        if invalid_grades:
            raise ValueError(
                f"Invalid Pfirrmann grades found: {invalid_grades}"
            )

        self.df["Pfirrmann_Grade"] = grades.astype(int)

        print(
            f"PfirrmannDatasetModel4 | "
            f"Split: {split or 'all':5s} | "
            f"Samples: {len(self.df)}"
        )

    def __len__(self):
        return len(self.df)

    def _load_image(self, relative_path):
        image_path = PROJECT_ROOT / str(relative_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(image_path).convert("L")

        if image.size != IMAGE_SIZE:
            raise ValueError(
                f"Expected image size {IMAGE_SIZE}, "
                f"got {image.size} for:\n{image_path}"
            )

        image = np.asarray(image, dtype=np.uint8)

        image = image.astype(np.float32) / 255.0

        image = np.stack(
            [image, image, image],
            axis=0,
        )

        return torch.from_numpy(image)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = self._load_image(
            row["MRI_Output_Path"]
        )

        grade = int(row["Pfirrmann_Grade"])

        label = GRADE_TO_LABEL[grade]

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image, label


# =============================================================================
# DATA LOADERS
# =============================================================================

def create_dataloaders():
    train_dataset = PfirrmannDatasetModel4(
        csv_path=DATASET_CSV,
        split="train",
    )

    val_dataset = PfirrmannDatasetModel4(
        csv_path=DATASET_CSV,
        split="val",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    return (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    )


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def validate_dataset_splits():
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found:\n{DATASET_CSV}"
        )

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Images directory not found:\n{IMAGES_DIR}"
        )

    df = pd.read_csv(DATASET_CSV)

    if "Split" not in df.columns:
        raise ValueError(
            "Dataset CSV does not contain the required 'Split' column."
        )

    split_counts = (
        df["Split"]
        .astype(str)
        .str.lower()
        .value_counts()
        .to_dict()
    )

    print()
    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    for split in ["train", "val", "test"]:
        actual = int(split_counts.get(split, 0))
        expected = EXPECTED_SPLIT_COUNTS[split]

        print(
            f"{split.capitalize():<10}"
            f"Expected: {expected:<8}"
            f"Found: {actual:<8}"
            f"{'OK' if actual == expected else 'MISMATCH'}"
        )

        if actual != expected:
            raise ValueError(
                f"Unexpected {split} split size. "
                f"Expected {expected}, found {actual}."
            )

    print("=" * 70)


# =============================================================================
# CLASS WEIGHTS
# =============================================================================

def calculate_class_weights(train_dataset, device):
    labels = (
        train_dataset.df["Pfirrmann_Grade"]
        .astype(int)
        .to_numpy()
    )

    class_indices = np.array(
        [GRADE_TO_LABEL[int(grade)] for grade in labels],
        dtype=np.int64,
    )

    counts = np.bincount(
        class_indices,
        minlength=NUM_CLASSES,
    )

    if np.any(counts == 0):
        missing = (
            np.where(counts == 0)[0] + 1
        )

        raise ValueError(
            "Training split contains no samples "
            f"for grade(s): {missing.tolist()}"
        )

    total_samples = len(class_indices)

    weights = (
        total_samples
        / (
            NUM_CLASSES
            * counts.astype(np.float64)
        )
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    print()
    print("=" * 70)
    print("CLASS WEIGHTS")
    print("=" * 70)
    print("Calculated from TRAIN split only.")
    print()

    print(
        f"{'Grade':<10}"
        f"{'Train Samples':<16}"
        f"{'Weight':<12}"
    )

    print("-" * 38)

    for grade in range(1, NUM_CLASSES + 1):
        print(
            f"{grade:<10}"
            f"{counts[grade - 1]:<16}"
            f"{weights[grade - 1]:.6f}"
        )

    print("=" * 70)

    return class_weights


# =============================================================================
# MODEL
# =============================================================================

def create_model(device):
    model = PfirrmannClassifierV2()

    model = model.to(device)

    return model


# =============================================================================
# OPTIMIZER
# =============================================================================

def create_optimizer(model):
    # Model 4:
    #   Neck: 1e-5
    #   Spatial Attention + Classifier: 1e-4
    #
    # Frozen ViT patch embedding, positional embedding, transformer blocks,
    # and any frozen final norm are deliberately excluded.

    neck_params = [
        parameter
        for parameter in model.image_encoder.neck.parameters()
        if parameter.requires_grad
    ]

    if hasattr(model, "spatial_attention"):
        spatial_attention_module = model.spatial_attention
    elif hasattr(model, "spatial_attention_pool"):
        spatial_attention_module = model.spatial_attention_pool
    else:
        raise AttributeError(
            "PfirrmannClassifierV2 does not contain "
            "'spatial_attention' or 'spatial_attention_pool'."
        )

    head_params = [
        parameter
        for parameter in spatial_attention_module.parameters()
        if parameter.requires_grad
    ]

    head_params.extend(
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    )

    if not neck_params:
        raise RuntimeError(
            "No trainable neck parameters were found."
        )

    if not head_params:
        raise RuntimeError(
            "No trainable head parameters were found."
        )

    neck_ids = {id(parameter) for parameter in neck_params}
    head_ids = {id(parameter) for parameter in head_params}

    if neck_ids & head_ids:
        raise RuntimeError(
            "A parameter appears in both optimizer groups."
        )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer_parameter_ids = neck_ids | head_ids
    trainable_parameter_ids = {
        id(parameter)
        for parameter in trainable_parameters
    }

    if optimizer_parameter_ids != trainable_parameter_ids:
        raise RuntimeError(
            "Optimizer parameter groups do not exactly match "
            "the model's trainable parameters."
        )

    optimizer = AdamW(
        [
            {
                "params": neck_params,
                "lr": NECK_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": head_params,
                "lr": HEAD_LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
        ]
    )

    return optimizer


# =============================================================================
# SCHEDULER
# =============================================================================

def create_scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=SCHEDULER_ETA_MIN,
    )


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
):
    model.train()

    running_loss = 0.0
    total = 0

    all_predictions = []
    all_labels = []

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [Train]",
        leave=True,
        dynamic_ncols=True,
    )

    for images, labels in progress:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total += batch_size

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        running_loss_value = (
            running_loss / total
        )

        progress.set_postfix(
            loss=f"{running_loss_value:.4f}"
        )

    train_loss = running_loss / total

    train_accuracy = np.mean(
        np.array(all_predictions)
        == np.array(all_labels)
    )

    train_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )

    return (
        train_loss,
        train_accuracy,
        train_macro_f1,
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    epoch,
):
    model.eval()

    running_loss = 0.0
    total = 0

    all_predictions = []
    all_labels = []

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [Val]",
        leave=True,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for images, labels in progress:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            batch_size = labels.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            total += batch_size

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            all_predictions.extend(
                predictions.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            all_labels.extend(
                labels.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            running_loss_value = (
                running_loss / total
            )

            progress.set_postfix(
                loss=f"{running_loss_value:.4f}"
            )

    val_loss = running_loss / total

    val_accuracy = np.mean(
        np.array(all_predictions)
        == np.array(all_labels)
    )

    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )

    return (
        val_loss,
        val_accuracy,
        val_macro_f1,
    )


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    train_accuracy,
    train_macro_f1,
    val_loss,
    val_accuracy,
    val_macro_f1,
    best_val_macro_f1,
    early_stopping_counter,
):
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

    torch.save(
        checkpoint,
        path,
    )


# =============================================================================
# RESUME
# =============================================================================

def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    device,
):
    print()
    print("=" * 70)
    print("RESUMING MODEL 4 TRAINING")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    required_keys = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "best_val_macro_f1",
        "early_stopping_counter",
    }

    missing_keys = (
        required_keys
        - set(checkpoint.keys())
    )

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state_dict"]
    )

    start_epoch = int(
        checkpoint["epoch"]
    )

    best_val_macro_f1 = float(
        checkpoint["best_val_macro_f1"]
    )

    early_stopping_counter = int(
        checkpoint["early_stopping_counter"]
    )

    print(
        f"Resuming after epoch : {start_epoch}"
    )

    print(
        f"Best validation F1   : "
        f"{best_val_macro_f1:.6f}"
    )

    print(
        "Early stopping       : disabled"
    )

    print("=" * 70)

    return (
        start_epoch,
        best_val_macro_f1,
        early_stopping_counter,
    )


# =============================================================================
# CSV LOGGING
# =============================================================================

def initialize_training_log():
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TRAINING_LOG_PATH.exists():
        with open(
            TRAINING_LOG_PATH,
            "w",
            newline="",
        ) as file:
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
    epoch,
    train_loss,
    train_accuracy,
    train_macro_f1,
    val_loss,
    val_accuracy,
    val_macro_f1,
    lr_neck,
    lr_head,
    time_taken,
):
    with open(
        TRAINING_LOG_PATH,
        "a",
        newline="",
    ) as file:
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


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train Model 4 for SPIDER Pfirrmann grading."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from outputs/model_4/checkpoints/"
            "latest_checkpoint.pt."
        ),
    )

    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint to resume from.",
    )

    args = parser.parse_args()

    if args.resume and args.resume_checkpoint:
        parser.error(
            "Use either --resume or --resume_checkpoint, not both."
        )

    return args


# =============================================================================
# PARAMETER SUMMARY
# =============================================================================

def print_parameter_summary(model):
    trainable_parameters = 0
    frozen_parameters = 0

    print()
    print("=" * 70)
    print("TRAINABLE PARAMETER SUMMARY")
    print("=" * 70)

    for name, parameter in model.named_parameters():
        count = parameter.numel()

        if parameter.requires_grad:
            trainable_parameters += count
            status = "TRAINABLE"
        else:
            frozen_parameters += count
            status = "FROZEN"

        print(
            f"{status:<10} "
            f"{count:>12,} "
            f"{name}"
        )

    print("-" * 70)
    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen parameters    : "
        f"{frozen_parameters:,}"
    )

    print(
        f"Total parameters     : "
        f"{trainable_parameters + frozen_parameters:,}"
    )

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_arguments()

    print("=" * 70)
    print("SPIDER PFIRRMANN TRAINING - MODEL 4")
    print("=" * 70)

    set_seed(RANDOM_SEED)

    device = get_device()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_dataset_splits()

    print()
    print(f"Project root        : {PROJECT_ROOT}")
    print(f"Dataset CSV         : {DATASET_CSV}")
    print(f"Images directory    : {IMAGES_DIR}")
    print(f"Output directory    : {OUTPUT_DIR}")

    print()
    print(f"Device              : {device}")

    if device.type == "cuda":
        print(
            f"GPU                 : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory          : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print()
    print(f"Batch size          : {BATCH_SIZE}")
    print(f"Epochs              : {NUM_EPOCHS}")
    print(f"Neck LR             : {NECK_LEARNING_RATE}")
    print(f"Head LR             : {HEAD_LEARNING_RATE}")
    print(f"Weight decay        : {WEIGHT_DECAY}")
    print(f"Scheduler            : CosineAnnealingLR")
    print(f"Scheduler eta_min    : {SCHEDULER_ETA_MIN}")
    print(f"AMP                 : OFF")
    print(f"Early stopping      : OFF")
    print(f"Resume requested    : {args.resume or bool(args.resume_checkpoint)}")

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = create_dataloaders()

    # -------------------------------------------------------------------------
    # CLASS WEIGHTS
    # -------------------------------------------------------------------------

    class_weights = calculate_class_weights(
        train_dataset,
        device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    model = create_model(device)

    print_parameter_summary(model)

    # -------------------------------------------------------------------------
    # OPTIMIZER + SCHEDULER
    # -------------------------------------------------------------------------

    optimizer = create_optimizer(model)

    scheduler = create_scheduler(
        optimizer
    )

    # -------------------------------------------------------------------------
    # RESUME
    # -------------------------------------------------------------------------

    start_epoch = 0
    best_val_macro_f1 = float("-inf")

    # Kept for checkpoint compatibility with Model 3.
    # Early stopping is disabled for Model 4.
    early_stopping_counter = 0

    if args.resume_checkpoint:
        checkpoint_path = Path(
            args.resume_checkpoint
        ).expanduser()

        if not checkpoint_path.is_absolute():
            checkpoint_path = (
                PROJECT_ROOT
                / checkpoint_path
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found:\n"
                f"{checkpoint_path}"
            )

        (
            start_epoch,
            best_val_macro_f1,
            early_stopping_counter,
        ) = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            device,
        )

    elif args.resume:
        if not LATEST_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                "Resume requested, but latest checkpoint "
                f"does not exist:\n{LATEST_CHECKPOINT_PATH}"
            )

        (
            start_epoch,
            best_val_macro_f1,
            early_stopping_counter,
        ) = load_checkpoint(
            LATEST_CHECKPOINT_PATH,
            model,
            optimizer,
            scheduler,
            device,
        )

    # -------------------------------------------------------------------------
    # LOG FILE
    # -------------------------------------------------------------------------

    initialize_training_log()

    # -------------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------------

    if start_epoch >= NUM_EPOCHS:
        print()
        print(
            f"Checkpoint is already at epoch {start_epoch}, "
            f"which is >= configured NUM_EPOCHS={NUM_EPOCHS}."
        )
        print("Nothing to train.")
        return

    for epoch in range(
        start_epoch + 1,
        NUM_EPOCHS + 1,
    ):
        epoch_start_time = time.time()

        (
            train_loss,
            train_accuracy,
            train_macro_f1,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
        )

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
        ) = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device,
            epoch,
        )

        # Record the learning rates used for this epoch.
        lr_neck = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        time_taken = (
            time.time()
            - epoch_start_time
        )

        # ---------------------------------------------------------------------
        # BEST MODEL
        # ---------------------------------------------------------------------

        is_best = (
            val_macro_f1
            > best_val_macro_f1
        )

        if is_best:
            best_val_macro_f1 = val_macro_f1

            save_checkpoint(
                BEST_MODEL_PATH,
                model,
                optimizer,
                scheduler,
                epoch,
                train_loss,
                train_accuracy,
                train_macro_f1,
                val_loss,
                val_accuracy,
                val_macro_f1,
                best_val_macro_f1,
                early_stopping_counter,
            )

            print()
            print(
                f"New best validation Macro F1: "
                f"{best_val_macro_f1:.4f}"
            )

            print(
                f"Best model saved to:\n"
                f"{BEST_MODEL_PATH}"
            )

        # ---------------------------------------------------------------------
        # CSV LOG
        # ---------------------------------------------------------------------

        append_training_log(
            epoch,
            train_loss,
            train_accuracy,
            train_macro_f1,
            val_loss,
            val_accuracy,
            val_macro_f1,
            lr_neck,
            lr_head,
            time_taken,
        )

        # ---------------------------------------------------------------------
        # SCHEDULER
        # ---------------------------------------------------------------------

        scheduler.step()

        # ---------------------------------------------------------------------
        # LATEST CHECKPOINT
        # ---------------------------------------------------------------------

        save_checkpoint(
            LATEST_CHECKPOINT_PATH,
            model,
            optimizer,
            scheduler,
            epoch,
            train_loss,
            train_accuracy,
            train_macro_f1,
            val_loss,
            val_accuracy,
            val_macro_f1,
            best_val_macro_f1,
            early_stopping_counter,
        )

        # ---------------------------------------------------------------------
        # EPOCH SUMMARY
        # ---------------------------------------------------------------------

        print()
        print("=" * 70)
        print(f"EPOCH {epoch:03d}/{NUM_EPOCHS}")
        print("=" * 70)

        print(
            f"Train Loss          : {train_loss:.6f}"
        )

        print(
            f"Train Accuracy      : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Train Macro F1      : "
            f"{train_macro_f1 * 100:.2f}%"
        )

        print(
            f"Val Loss            : {val_loss:.6f}"
        )

        print(
            f"Val Accuracy        : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Val Macro F1        : "
            f"{val_macro_f1 * 100:.2f}%"
        )

        print(
            f"LR Neck             : "
            f"{lr_neck:.10f}"
        )

        print(
            f"LR Head             : "
            f"{lr_head:.10f}"
        )

        print(
            f"Time Taken          : "
            f"{time_taken:.2f} seconds"
        )

        print(
            f"Best Val Macro F1   : "
            f"{best_val_macro_f1 * 100:.2f}%"
        )

        print("=" * 70)

    print()
    print("=" * 70)
    print("MODEL 4 TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best validation Macro F1: "
        f"{best_val_macro_f1 * 100:.2f}%"
    )

    print(
        f"Best checkpoint: "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Latest checkpoint: "
        f"{LATEST_CHECKPOINT_PATH}"
    )

    print(
        f"Training log: "
        f"{TRAINING_LOG_PATH}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
