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

from configs.training_config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    DROP_LAST,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BEST_MODEL_PATH,
    LATEST_CHECKPOINT_PATH,
    TRAINING_LOG_PATH,
    RANDOM_SEED,
    DEVICE,
    VALIDATE_EVERY_EPOCH,
    EARLY_STOPPING_ENABLED,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    RESUME_TRAINING,
    PRINT_EVERY_BATCH,
)

from training.dataset import PfirrmannDataset
from training.models.pfirrmann_classifier import PfirrmannClassifier


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

    if (
        DEVICE == "cuda"
        and torch.cuda.is_available()
    ):

        return torch.device("cuda")

    print(
        "WARNING: CUDA requested but unavailable. "
        "Using CPU."
    )

    return torch.device("cpu")


# =============================================================================
# DATA LOADERS
# =============================================================================

def create_dataloaders():

    train_dataset = PfirrmannDataset(
        split="train"
    )

    val_dataset = PfirrmannDataset(
        split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST,
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
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        # --------------------------------------------------------------
        # Statistics
        # --------------------------------------------------------------

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total += batch_size

        predictions = torch.argmax(
            logits,
            dim=1
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

        # --------------------------------------------------------------
        # Progress
        # --------------------------------------------------------------

        if (
            batch_idx + 1
        ) % PRINT_EVERY_BATCH == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Batch {batch_idx + 1:04d}/"
                f"{len(loader):04d} | "
                f"Loss {loss.item():.4f}"
            )

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        np.mean(
            np.array(all_predictions)
            == np.array(all_labels)
        )
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
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        logits = model(
            images
        )

        loss = criterion(
            logits,
            labels
        )

        batch_size = labels.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total += batch_size

        predictions = torch.argmax(
            logits,
            dim=1
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

    val_accuracy = (
        np.mean(
            np.array(all_predictions)
            == np.array(all_labels)
        )
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

    torch.save(
        checkpoint,
        path
    )


# =============================================================================
# RESUME
# =============================================================================

def load_latest_checkpoint(
    model,
    optimizer,
    device,
):

    if not LATEST_CHECKPOINT_PATH.exists():

        print()
        print(
            "No latest checkpoint found."
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
    print("RESUMING TRAINING")
    print("=" * 70)

    checkpoint = torch.load(
        LATEST_CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
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

    start_epoch = checkpoint[
        "epoch"
    ]

    best_val_macro_f1 = checkpoint.get(
        "best_val_macro_f1",
        float("-inf")
    )

    early_stopping_counter = checkpoint.get(
        "early_stopping_counter",
        0
    )

    print(
        f"Checkpoint epoch      : "
        f"{start_epoch}"
    )

    print(
        f"Best validation F1    : "
        f"{best_val_macro_f1:.4f}"
    )

    print(
        f"Early stopping count  : "
        f"{early_stopping_counter}"
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

    if TRAINING_LOG_PATH.exists():

        return

    with open(
        TRAINING_LOG_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

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
        newline=""
    ) as file:

        writer = csv.writer(
            file
        )

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
    print("SPIDER PFIRRMANN TRAINING")
    print("=" * 70)

    set_seed(
        RANDOM_SEED
    )

    device = get_device()

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

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

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
        f"Batch size          : "
        f"{BATCH_SIZE}"
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
    # Model
    # -------------------------------------------------------------------------

    print()
    print("Creating model...")

    model = PfirrmannClassifier()

    model = model.to(
        device
    )

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------

    optimizer = AdamW(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters()
        ),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
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
        NUM_EPOCHS + 1
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

        if VALIDATE_EVERY_EPOCH:

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

        else:

            val_loss = float("nan")

            val_accuracy = float("nan")

            val_macro_f1 = float("nan")

        # --------------------------------------------------------------
        # Current learning rate
        # --------------------------------------------------------------

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
            EARLY_STOPPING_ENABLED
            and
            early_stopping_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "=" * 70
            )

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

            print(
                "=" * 70
            )

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