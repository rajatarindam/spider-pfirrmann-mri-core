# =============================================================================
# TRAINING CONFIGURATION - NEW FINAL PATIENT STRATIFIED EXPERIMENT
# =============================================================================
#
# This configuration is intentionally separate from the original
# training_config.py so the first training experiment remains untouched.
#
# New experiment:
#   Dataset  : data/new_final/final_data/dataset_updated_pfirrmann_patient_stratified.csv
#   Samples  : 7003
#   Patients : 204
#
# =============================================================================


# =============================================================================
# DATA
# =============================================================================

NUM_CLASSES = 5

# Pfirrmann grades 1-5
CLASS_NAMES = [
    "Grade 1",
    "Grade 2",
    "Grade 3",
    "Grade 4",
    "Grade 5",
]


# =============================================================================
# TRAINING
# =============================================================================

BATCH_SIZE = 6  

NUM_EPOCHS = 50

NUM_WORKERS = 4

PIN_MEMORY = True

PERSISTENT_WORKERS = True


# =============================================================================
# OPTIMIZER
# =============================================================================

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4


# =============================================================================
# CLASS WEIGHTS
# =============================================================================
#
# Class weights are calculated dynamically from the TRAINING SPLIT ONLY.
#
# Formula:
#
#     weight_c = N / (C * n_c)
#
# where:
#     N   = number of training samples
#     C   = number of classes
#     n_c = number of samples belonging to class c
#
# This prevents validation/test data from influencing training weights.
#
# The resulting weights are passed to:
#
#     nn.CrossEntropyLoss(weight=class_weights)
#
# =============================================================================

USE_CLASS_WEIGHTS = True

CLASS_WEIGHT_METHOD = "balanced"


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================
#
# ReduceLROnPlateau monitors validation Macro F1.
#
# mode="max":
#     Higher Macro F1 is better.
#
# factor=0.5:
#     When a plateau is detected:
#
#         new_lr = old_lr * 0.5
#
# patience=3:
#     Wait for 3 consecutive validation epochs without improvement
#     before reducing the learning rate.
#
# min_lr=1e-6:
#     Never reduce the learning rate below 1e-6.
#
# =============================================================================

USE_LR_SCHEDULER = False

SCHEDULER_MODE = "max"

SCHEDULER_FACTOR = 0.5

SCHEDULER_PATIENCE = 3

SCHEDULER_MIN_LR = 1e-6


# =============================================================================
# EARLY STOPPING
# =============================================================================
#
# Early stopping is separate from the LR scheduler.
#
# The scheduler gets a chance to reduce the learning rate when validation
# Macro F1 plateaus.
#
# If validation Macro F1 still does not improve for 10 epochs, training stops.
#
# =============================================================================

USE_EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 15

EARLY_STOPPING_MIN_DELTA = 0.0


# =============================================================================
# MODEL
# =============================================================================
#
# These settings are kept aligned with the original training experiment.
# =============================================================================

BACKBONE_NAME = "MRI-CORE"

NUM_INPUT_CHANNELS = 3

INPUT_SIZE = 1024

EMBEDDING_DIM = 256


# =============================================================================
# CHECKPOINTING
# =============================================================================

SAVE_BEST_CHECKPOINT = True

SAVE_LAST_CHECKPOINT = True


# =============================================================================
# METRICS
# =============================================================================
#
# Validation Macro F1 is the primary model-selection and scheduler metric.
# =============================================================================

PRIMARY_METRIC = "val_macro_f1"

SCHEDULER_METRIC = "val_macro_f1"


# =============================================================================
# RANDOMNESS
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# RESUME TRAINING
# =============================================================================

RESUME_TRAINING = True