"""
Training configuration for SPIDER Pfirrmann grading.
"""

from pathlib import Path


# =============================================================================
# PROJECT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# DATASET
# =============================================================================

# Local testing value.
# This will be benchmarked and changed on the T4 server.
BATCH_SIZE = 2

NUM_WORKERS = 0

PIN_MEMORY = True

DROP_LAST = False


# =============================================================================
# MODEL
# =============================================================================

NUM_CLASSES = 5

FEATURE_DIM = 256


# =============================================================================
# TRAINING
# =============================================================================

# Maximum number of epochs.
NUM_EPOCHS = 50

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4


# =============================================================================
# OPTIMIZER
# =============================================================================

OPTIMIZER = "AdamW"


# =============================================================================
# LOSS
# =============================================================================

LOSS_FUNCTION = "CrossEntropyLoss"

# No class weights for the baseline experiment.
USE_CLASS_WEIGHTS = False


# =============================================================================
# VALIDATION
# =============================================================================

# Validation runs after every epoch.
VALIDATE_EVERY_EPOCH = True


# =============================================================================
# MODEL SELECTION
# =============================================================================

# Primary metric for selecting the best model.
BEST_MODEL_METRIC = "val_macro_f1"

# Higher Macro F1 is better.
BEST_MODEL_MODE = "max"


# =============================================================================
# EARLY STOPPING
# =============================================================================

EARLY_STOPPING_ENABLED = True

EARLY_STOPPING_PATIENCE = 15

# Minimum improvement required to reset patience.
EARLY_STOPPING_MIN_DELTA = 0.0


# =============================================================================
# CHECKPOINTS
# =============================================================================

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "checkpoints"
    / "pfirrmann"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR
    / "best_model.pth"
)

LATEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "latest_checkpoint.pth"
)

TRAINING_LOG_PATH = (
    CHECKPOINT_DIR
    / "training_log.csv"
)


# =============================================================================
# RESUME TRAINING
# =============================================================================

# Explicitly control whether training resumes from latest_checkpoint.pth.
RESUME_TRAINING = False


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

RANDOM_SEED = 42


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = "cuda"


# =============================================================================
# LOGGING
# =============================================================================

# Print detailed batch progress every N batches.
PRINT_EVERY_BATCH = 50