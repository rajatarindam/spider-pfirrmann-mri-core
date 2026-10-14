# =============================================================================
# TRAINING CONFIGURATION - MODEL 5 (PHASE 1 + PHASE 2 COMBINED)
# =============================================================================
#
# Experiment:
#   Dataset  : data/new_final/final_data/dataset_multi_view_v2.csv
#   Samples  : 1412 disc groups (each group = 5 contiguous MRI slices)
#   Views    : 5 offsets per disc  (-2, -1, 0, +1, +2)
#   Task     : 5-class Pfirrmann grade classification (disc-level)
#
# Key Architectural & Training Enhancements:
#   Phase 1:
#     - SAM Neck restored & unfrozen (lr=1e-5) for local spatial patch interactions
#     - SpatialAttentionPooling (lr=1e-4) replaces uniform mean pooling
#     - Two-group AdamW differential optimizer (neck vs heads)
#     - CosineAnnealingWarmRestarts scheduler (T_0=10, T_mult=2, eta_min=1e-5 / 1e-6)
#     - Gradient clipping tightened to max_norm=0.5
#     - Coherent 3D multi-slice geometric augmentation + independent radiometric jitter
#   Phase 2:
#     - SoftOrdinalFocalLoss (Gaussian-smoothed targets, sigma=0.70)
#     - Focal modulating factor (gamma=2.0) focusing on hard borderline cases
#     - Auxiliary mid-sagittal center-slice supervision (lambda=0.30)
#
# =============================================================================


# =============================================================================
# DATA
# =============================================================================

NUM_CLASSES = 5

CLASS_NAMES = [
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4",
    "Grade 5",
]

# Slice offsets from disc centre (ordered medial-negative to medial-positive)
OFFSETS = [-2, -1, 0, 1, 2]

NUM_VIEWS = len(OFFSETS)  # 5


# =============================================================================
# GRADE <-> LABEL MAPPINGS
# =============================================================================

GRADE_TO_LABEL = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
}

LABEL_TO_GRADE = {v: k for k, v in GRADE_TO_LABEL.items()}


# =============================================================================
# TRAINING HARDWARE & BATCHING
# =============================================================================

# One disc group (5 views) per iteration — dictated by GPU VRAM constraints
BATCH_SIZE = 4

NUM_EPOCHS = 50

NUM_WORKERS = 2

PIN_MEMORY = True

RANDOM_SEED = 42

IMAGE_SIZE = (1024, 1024)


# =============================================================================
# OPTIMIZER (Differential Parameter Groups)
# =============================================================================
#
# Two parameter groups:
#   Group 0 (Pretrained SAM Neck) : lr=1e-5, weight_decay=1e-4
#   Group 1 (Newly Initialised)   : lr=1e-4, weight_decay=1e-4
#
# =============================================================================

LEARNING_RATE_NECK = 1e-5
LEARNING_RATE_HEAD = 1e-4
WEIGHT_DECAY = 1e-4


# =============================================================================
# LR SCHEDULER (CosineAnnealingWarmRestarts)
# =============================================================================
#
# Periodic warm restarts allow escaping local saddle points.
# Cycle 1: epochs 1–10 (T_0=10)
# Cycle 2: epochs 11–30 (T_mult=2 -> 20 epochs)
# Cycle 3: epochs 31–70
#
# =============================================================================

USE_LR_SCHEDULER = True
SCHED_T_0 = 10
SCHED_T_MULT = 2
SCHED_ETA_MIN_HEAD = 1e-5
SCHED_ETA_MIN_NECK = 1e-6


# =============================================================================
# LOSS (SoftOrdinalFocalLoss + Auxiliary Center Supervision)
# =============================================================================
#
# Replaces nominal CrossEntropy with an ordinal-aware objective:
#   - ORDINAL_SIGMA = 0.70 : Gaussian target distribution centered on ground-truth
#   - FOCAL_GAMMA   = 2.0  : Down-weights easy confident examples
#   - AUX_CENTER_LOSS_WEIGHT = 0.30 : Auxiliary loss on mid-sagittal center slice
#   - USE_CLASS_WEIGHTS = True : Inverse frequency weighting from train split
#
# =============================================================================

LOSS_TYPE = "ordinal_focal"
ORDINAL_SIGMA = 0.70
FOCAL_GAMMA = 2.0
USE_AUX_CENTER_LOSS = True
AUX_CENTER_LOSS_WEIGHT = 0.30
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_METHOD = "balanced"  # N / (C * n_c)


# =============================================================================
# GRADIENT CLIPPING
# =============================================================================

GRAD_CLIP_MAX_NORM = 0.5


# =============================================================================
# AUGMENTATION (Coherent Multi-View + Radiometric Jitter)
# =============================================================================
#
# Coherent across all 5 slices:
#   - Random horizontal flip (p=0.5)
#   - Random rotation (-10 deg to +10 deg, p=0.7)
#   - Random affine scaling (0.92 to 1.08, p=0.5)
#
# Independent per slice:
#   - Brightness jitter (+-20%)
#   - Contrast jitter (0.85 to 1.15, p=0.7)
#   - Gaussian noise (sigma=0.02)
#   - Random erasing / cutout (2% to 8% area, p=0.20)
#
# Strict clinical prohibitions:
#   - NO vertical flip (inverts cranial/caudal anatomy)
#   - NO large rotations (> 15 deg)
#   - NO RGB color jitter
#
# =============================================================================

USE_AUGMENTATION = True

AUG_HFLIP_PROB       = 0.5
AUG_ROTATION_DEG     = 10.0
AUG_ROTATION_PROB    = 0.7
AUG_SCALE_MIN        = 0.92
AUG_SCALE_MAX        = 1.08
AUG_SCALE_PROB       = 0.5

AUG_BRIGHTNESS_DELTA = 0.20
AUG_CONTRAST_RANGE   = (0.85, 1.15)
AUG_CONTRAST_PROB    = 0.7
AUG_NOISE_STD        = 0.02
AUG_ERASING_PROB     = 0.20
AUG_ERASING_SCALE    = (0.02, 0.08)


# =============================================================================
# CHECKPOINTING
# =============================================================================

PRIMARY_METRIC = "val_macro_f1"
SAVE_BEST_CHECKPOINT   = True
SAVE_LATEST_CHECKPOINT = True