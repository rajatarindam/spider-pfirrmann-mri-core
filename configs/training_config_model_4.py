# =============================================================================
# TRAINING CONFIGURATION - MODEL 4 (SINGLE-VIEW 2D PFIRRMANN CLASSIFIER)
# =============================================================================
#
# Experiment:
#   Dataset  : data/new_final/final_data/dataset_OG.csv
#   Samples  : 7,003 individual 2D sagittal MRI slice crops
#   Splits   : 4,864 train / 1,408 val / 731 test
#   Task     : 5-class Pfirrmann grade classification (slice-level)
#
# Architecture:
#   - Backbone: Pretrained MRI-CORE ViT-B (blocks 0–11 frozen under no_grad)
#   - Positional Embedding: Corrected 64x64 interpolated pos_embed
#   - SAM Neck: Unfrozen & trainable conv layers (768 -> 256, lr = 1e-5)
#   - Pooling: SpatialAttentionPooling(256)
#   - Classifier Head: 2-layer MLP with BatchNorm1d and GELU (lr = 1e-4)
#   - Optimizer: 2-group AdamW (neck vs. head) with CosineAnnealingLR
#   - Loss: Class-weighted CrossEntropyLoss (inverse-frequency balanced)
#
# =============================================================================


# =============================================================================
# DATA & CLASSES
# =============================================================================

NUM_CLASSES = 5

CLASS_NAMES = [
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4",
    "Grade 5",
]

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
# TRAINING HARDWARE & BATCHING
# =============================================================================

BATCH_SIZE = 6

NUM_EPOCHS = 50

NUM_WORKERS = 4

PIN_MEMORY = True

RANDOM_SEED = 42

IMAGE_SIZE = (1024, 1024)


# =============================================================================
# OPTIMIZER (Differential Parameter Groups)
# =============================================================================
#
# Two parameter groups:
#   Group 0 (Pretrained SAM Neck) : lr=1e-5, weight_decay=1e-4
#   Group 1 (Spatial Attn + Head) : lr=1e-4, weight_decay=1e-4

NECK_LEARNING_RATE = 1e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY       = 1e-4


# =============================================================================
# LEARNING RATE SCHEDULER (CosineAnnealingLR)
# =============================================================================

USE_LR_SCHEDULER = True
SCHEDULER_ETA_MIN = 1e-6


# =============================================================================
# LOSS FUNCTION & CLASS WEIGHTS
# =============================================================================

USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_METHOD = "inverse_frequency"


# =============================================================================
# CHECKPOINTING
# =============================================================================

PRIMARY_METRIC = "val_macro_f1"
SAVE_BEST_CHECKPOINT   = True
SAVE_LATEST_CHECKPOINT = True

