# =============================================================================
# TRAINING CONFIGURATION - MODEL 5 MULTI-VIEW EXPERIMENT
# =============================================================================
#
# This file is intentionally isolated from all previous training configurations
# so that Model 5 cannot overwrite or conflict with Models 1-4 outputs.
#
# Experiment:
#   Dataset  : data/new_final/final_data/dataset_multi_view_v2.csv
#   Samples  : 1412 disc groups (each group = 5 contiguous MRI slices)
#   Views    : 5 offsets per disc  (-2, -1, 0, +1, +2)
#   Task     : 5-class Pfirrmann grade classification (disc-level)
#
# Key changes vs Model 4 training config:
#   - Single optimizer group at constant lr=1e-4 (no differential LR)
#   - No LR scheduler (Cosine Annealing caused LR entrapment in Model 4)
#   - Label smoothing = 0.1 (prevents logit saturation & overconfidence)
#   - Gradient clipping max_norm = 1.0 (stabilises B=1 single-disc batches)
#   - Training augmentation enabled (random HFlip + brightness + noise)
#   - Primary checkpoint metric = val_macro_f1 (not val_accuracy)
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
# GRADE ↔ LABEL MAPPINGS
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
# TRAINING
# =============================================================================

# One disc group (5 views) per iteration — dictated by GPU VRAM constraints
BATCH_SIZE = 1

NUM_EPOCHS = 50

NUM_WORKERS = 2

PIN_MEMORY = True

RANDOM_SEED = 42

IMAGE_SIZE = (1024, 1024)


# =============================================================================
# OPTIMIZER
# =============================================================================
#
# Single parameter group at constant lr=1e-4.
#
# In Model 4 two groups were used:
#     neck params  → lr=1e-5  (protecting pretrained SAM conv weights)
#     head params  → lr=1e-4
#
# In Model 5 the SAM neck is frozen and unused in forward(). The only
# trainable parameters are freshly initialised (LinearProjection,
# ViewPositionalEmbedding, ViewAttentionAggregator, Classifier).
# A single learning rate is therefore appropriate.
#
# =============================================================================

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4


# =============================================================================
# LR SCHEDULER
# =============================================================================
#
# No scheduler. Cosine Annealing in Model 4 decayed lr to ~3e-6 by epoch 46,
# causing LR entrapment and deep memorisation of training images.
# A flat constant lr=1e-4 is maintained for all 50 epochs.
#
# =============================================================================

USE_LR_SCHEDULER = False


# =============================================================================
# LOSS
# =============================================================================
#
# CrossEntropyLoss with:
#   - class_weights : computed from training split counts (balanced formula)
#   - label_smoothing = 0.1 : softens hard targets to prevent logit saturation
#       Hard target  : [0, 0, 1, 0, 0]
#       Smooth target: [0.02, 0.02, 0.92, 0.02, 0.02]
#     This directly addresses Model 4's overconfidence pathology where wrong
#     Grade 4 predictions reached 88.8% confidence.
#
# =============================================================================

USE_CLASS_WEIGHTS = True

CLASS_WEIGHT_METHOD = "balanced"  # N / (C * n_c)

LABEL_SMOOTHING = 0.1


# =============================================================================
# GRADIENT CLIPPING
# =============================================================================
#
# With BATCH_SIZE=1, a single unusual disc can produce a large gradient spike.
# Clipping the global gradient norm prevents any one sample from producing a
# destabilising weight update.
#
# =============================================================================

GRAD_CLIP_MAX_NORM = 1.0


# =============================================================================
# AUGMENTATION (training mode only)
# =============================================================================
#
# Applied per-slice, only during training (never during validation or test).
#
# Allowed:
#   - Random horizontal flip (p=0.5)  – spinal anatomy is left-right symmetric
#   - Brightness jitter ±10%         – accounts for scanner intensity variation
#   - Gaussian noise (σ=0.01)        – mild noise robustness
#
# NOT used:
#   - Vertical flip  (superior/inferior anatomy is clinically meaningful)
#   - Rotation       (spinal orientation is fixed in the imaging plane)
#   - Strong colour  (images are grayscale MRI — colour is irrelevant)
#
# =============================================================================

USE_AUGMENTATION = True

AUG_HFLIP_PROB       = 0.5
AUG_BRIGHTNESS_DELTA = 0.10   # uniform sample from [-0.10, +0.10]
AUG_NOISE_STD        = 0.01   # Gaussian noise standard deviation


# =============================================================================
# CHECKPOINTING
# =============================================================================
#
# Primary metric for best-model selection: val_macro_f1
#   (more robust than val_accuracy for the imbalanced 5-class grading task)
#
# Files saved per epoch:
#   1. training_log.csv          – appended after every epoch
#   2. best_model.pt             – updated when val_macro_f1 improves
#   3. latest_checkpoint.pt      – always overwritten (enables --resume)
#   4. checkpoint_epoch_NNN.pt   – per-epoch snapshot (full resume support)
#
# =============================================================================

PRIMARY_METRIC = "val_macro_f1"

SAVE_BEST_CHECKPOINT   = True
SAVE_LATEST_CHECKPOINT = True
