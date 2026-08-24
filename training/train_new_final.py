import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from configs import paths

from configs.training_config_new_final import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_CLASSES,
    USE_CLASS_WEIGHTS,
    CLASS_WEIGHT_METHOD,
    SCHEDULER_MODE,
    SCHEDULER_FACTOR,
    SCHEDULER_PATIENCE,
    SCHEDULER_MIN_LR,
    USE_LR_SCHEDULER,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    USE_EARLY_STOPPING,
    RANDOM_SEED,
)

from training.dataset_new_final import PfirrmannDatasetNewFinal

# EXACT same classifier used by the original training script.
from training.models.pfirrmann_classifier import PfirrmannClassifier


# =============================================================================
# NEW FINAL OUTPUT PATHS
# =============================================================================

CHECKPOINT_DIR = paths.NEW_FINAL_CHECKPOINT_DIR
LOG_DIR = paths.NEW_FINAL_LOGS_DIR

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_model.pt"
LATEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "latest_checkpoint.pt"
TRAINING_LOG_PATH = LOG_DIR / "training_log.csv"


# =============================================================================
# RESUME
# =============================================================================

# Intentionally TRUE as requested.
RESUME_TRAINING = True


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

    print(
        "WARNING: CUDA unavailable. "
        "Training will use CPU."
    )

    return torch.device("cpu")


# =============================================================================
# DATA LOADERS
# =============================================================================

def create_dataloaders():

    train_dataset = PfirrmannDatasetNewFinal(
        split="train"
    )

    val_dataset = PfirrmannDatasetNewFinal(
        split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
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
# CLASS WEIGHTS
# =============================================================================

def calculate_class_weights(train_dataset, device):

    labels = (
        train_dataset.df["Pfirrmann_Grade"]
        .astype(int)
        .to_numpy()
    )

    # Convert grades 1-5 to class indices 0-4.
    class_indices = labels - 1

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

    if CLASS_WEIGHT_METHOD != "balanced":
        raise ValueError(
            "Unsupported class-weight method: "
            f"{CLASS_WEIGHT_METHOD}"
        )

    total_samples = len(labels)

    weights = (
        total_samples
        / (NUM_CLASSES * counts.astype(np.float64))
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

    for batch_idx, (
        images,
        labels,
    ) in enumerate(loader):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

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

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = np.mean(
        np.array(all_predictions)
        == np.array(all_labels)
    )

    epoch_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        labels=[0, 1, 2, 3, 4],
        zero_division=0,
    )

    return (
        epoch_loss,
        epoch_accuracy,
        epoch_macro_f1,
    )


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0
    total = 0

    all_predictions = []
    all_labels = []

    for images, labels in loader:

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

    val_loss = (
        running_loss / total
    )

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

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "train_macro_f1":
            train_macro_f1,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "val_macro_f1":
            val_macro_f1,

        "best_val_macro_f1":
            best_val_macro_f1,

        "early_stopping_counter":
            early_stopping_counter,
    }

    if scheduler is not None:
        checkpoint[
            "scheduler_state_dict"
        ] = scheduler.state_dict()

    torch.save(
        checkpoint,
        path,
    )


# =============================================================================
# RESUME
# =============================================================================

def load_latest_checkpoint(
    model,
    optimizer,
    scheduler,
    device,
):

    if not LATEST_CHECKPOINT_PATH.exists():

        print()
        print(
            "No new-final checkpoint found."
        )
        print(
            "Starting training from epoch 1."
        )

        return (
            0,
            float("-inf"),
            0,
        )

    print()
    print("=" * 70)
    print("RESUMING NEW-FINAL TRAINING")
    print("=" * 70)

    checkpoint = torch.load(
        LATEST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    # Reset optimizer learning rate to the current configuration.
    # The checkpoint was created with the previous LR (5e-5).
    # This experiment resumes with a constant LR of 1e-4.
    for param_group in optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    if (
        scheduler is not None
        and "scheduler_state_dict" in checkpoint
    ):
        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

    start_epoch = checkpoint[
        "epoch"
    ]

    best_val_macro_f1 = checkpoint.get(
        "best_val_macro_f1",
        float("-inf"),
    )

    early_stopping_counter = 0

    print(
        f"Checkpoint epoch     : "
        f"{start_epoch}"
    )

    print(
        f"Best validation F1   : "
        f"{best_val_macro_f1:.4f}"
    )

    print(
        f"Early stopping count : "
        f"{early_stopping_counter}"
    )

    if scheduler is not None:
        print(
            f"Current learning rate: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

    print("=" * 70)

    return (
        start_epoch,
        best_val_macro_f1,
        early_stopping_counter,
    )


# =============================================================================
# TRAINING LOG
# =============================================================================

def initialize_log():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TRAINING_LOG_PATH.exists():
        return

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
            "learning_rate",
        ])


def log_epoch(
    epoch,
    train_loss,
    train_accuracy,
    train_macro_f1,
    val_loss,
    val_accuracy,
    val_macro_f1,
    learning_rate,
):

    with open(
        TRAINING_LOG_PATH,
        "a",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            epoch,
            train_loss,
            train_accuracy,
            train_macro_f1,
            val_loss,
            val_accuracy,
            val_macro_f1,
            learning_rate,
        ])


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("SPIDER PFIRRMANN TRAINING - NEW FINAL DATASET")
    print("=" * 70)

    set_seed(
        RANDOM_SEED
    )

    device = get_device()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Device              : "
        f"{device}"
    )

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
    print(
        f"Batch size          : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs              : "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Initial LR          : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Weight decay        : "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"Class weights       : "
        f"{USE_CLASS_WEIGHTS}"
    )

    print(
        f"LR scheduler        : "
        f"{'ReduceLROnPlateau' if USE_LR_SCHEDULER else 'OFF'}"
    )

    print(
        f"Scheduler patience  : "
        f"{SCHEDULER_PATIENCE}"
    )

    print(
        f"Scheduler factor    : "
        f"{SCHEDULER_FACTOR}"
    )

    print(
        f"Scheduler min LR    : "
        f"{SCHEDULER_MIN_LR}"
    )

    print(
        f"Early stopping      : "
        f"{USE_EARLY_STOPPING}"
    )

    print(
        f"Early stop patience : "
        f"{EARLY_STOPPING_PATIENCE}"
    )

    print(
        f"Resume training     : "
        f"{RESUME_TRAINING}"
    )

    print()
    print("Creating datasets...")

    (
        train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = create_dataloaders()

    print()
    print(
        f"Train samples       : "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples  : "
        f"{len(val_dataset)}"
    )

    print(
        f"Train batches       : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches  : "
        f"{len(val_loader)}"
    )

    # -------------------------------------------------------------------------
    # Class weights
    # -------------------------------------------------------------------------

    if USE_CLASS_WEIGHTS:

        class_weights = (
            calculate_class_weights(
                train_dataset,
                device,
            )
        )

    else:

        class_weights = None

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    print()
    print("Creating model...")

    # EXACT SAME MODEL CLASS AS ORIGINAL TRAINING.
    model = PfirrmannClassifier()

    model = model.to(
        device
    )

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = AdamW(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters(),
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------------------------------------------------------
    # LR scheduler
    # -------------------------------------------------------------------------

    scheduler = None

    if USE_LR_SCHEDULER:

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=SCHEDULER_MODE,
                factor=SCHEDULER_FACTOR,
                patience=SCHEDULER_PATIENCE,
                min_lr=SCHEDULER_MIN_LR,
            )
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    initialize_log()

    # -------------------------------------------------------------------------
    # Resume
    # -------------------------------------------------------------------------

    start_epoch = 0

    best_val_macro_f1 = float(
        "-inf"
    )

    early_stopping_counter = 0

    if RESUME_TRAINING:

        (
            start_epoch,
            best_val_macro_f1,
            early_stopping_counter,
        ) = load_latest_checkpoint(
            model,
            optimizer,
            scheduler,
            device,
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    for epoch in range(
        start_epoch + 1,
        NUM_EPOCHS + 1,
    ):

        print()
        print("-" * 70)
        print(
            f"EPOCH {epoch}/{NUM_EPOCHS}"
        )
        print("-" * 70)

        # --------------------------------------------------------------
        # Train
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
        ) = validate(
            model,
            val_loader,
            criterion,
            device,
        )

        # --------------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------------
        #
        # IMPORTANT:
        # ReduceLROnPlateau must receive validation Macro F1 AFTER
        # validation has completed.
        #

        if scheduler is not None:

            old_lr = optimizer.param_groups[0][
                "lr"
            ]

            scheduler.step(
                val_macro_f1
            )

            new_lr = optimizer.param_groups[0][
                "lr"
            ]

            if new_lr < old_lr:

                print()
                print(
                    "Learning rate reduced:"
                )

                print(
                    f"  {old_lr:.8f} -> "
                    f"{new_lr:.8f}"
                )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        # --------------------------------------------------------------
        # Print epoch metrics
        # --------------------------------------------------------------

        print()
        print(
            f"Epoch {epoch:03d} Results"
        )

        print(
            f"Train Loss          : "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy      : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Train Macro F1      : "
            f"{train_macro_f1:.4f}"
        )

        print(
            f"Validation Loss     : "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation Accuracy : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Validation Macro F1 : "
            f"{val_macro_f1:.4f}"
        )

        print(
            f"Learning Rate       : "
            f"{current_lr:.8f}"
        )

        # --------------------------------------------------------------
        # Best model / early stopping
        # --------------------------------------------------------------

        improved = (
            val_macro_f1
            >
            best_val_macro_f1
            + EARLY_STOPPING_MIN_DELTA
        )

        if improved:

            best_val_macro_f1 = (
                val_macro_f1
            )

            early_stopping_counter = 0

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

            print(
                "Best model saved."
            )

        else:

            early_stopping_counter += 1

            print(
                f"No validation F1 improvement. "
                f"Patience: "
                f"{early_stopping_counter}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        # --------------------------------------------------------------
        # Latest checkpoint
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Training log
        # --------------------------------------------------------------

        log_epoch(
            epoch,
            train_loss,
            train_accuracy,
            train_macro_f1,
            val_loss,
            val_accuracy,
            val_macro_f1,
            current_lr,
        )

        # --------------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------------

        if (
            USE_EARLY_STOPPING
            and
            early_stopping_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print("=" * 70)
            print(
                "EARLY STOPPING TRIGGERED"
            )

            print(
                f"No validation Macro F1 "
                f"improvement for "
                f"{EARLY_STOPPING_PATIENCE} epochs."
            )

            print(
                f"Best validation Macro F1 : "
                f"{best_val_macro_f1:.4f}"
            )

            print("=" * 70)

            break

    # -------------------------------------------------------------------------
    # Finished
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best validation Macro F1 : "
        f"{best_val_macro_f1:.4f}"
    )

    print(
        f"Best model               : "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Latest checkpoint        : "
        f"{LATEST_CHECKPOINT_PATH}"
    )

    print(
        f"Training log             : "
        f"{TRAINING_LOG_PATH}"
    )


if __name__ == "__main__":
    main()