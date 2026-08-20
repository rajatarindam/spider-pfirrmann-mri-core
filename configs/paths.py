"""
Project path configuration.

This module defines all commonly used project directories.
Every module should import paths from here rather than hardcoding
directory locations.
"""

from pathlib import Path

# =============================================================================
# Project Root
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# =============================================================================
# MRI-CORE
# =============================================================================

MRI_CORE_DIR = PROJECT_ROOT / "mri_core"

MRI_CORE_CHECKPOINT_PATH = (
    MRI_CORE_DIR / "pretrained_weights" / "mri_foundation.pth"
)

# =============================================================================
# Data Directories
# =============================================================================

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
RAW_MASKS_DIR = RAW_DATA_DIR / "masks"

UNPACKED_DATA_DIR = DATA_DIR / "unpacked"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"

PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
PROCESSED_MASKS_DIR = PROCESSED_DATA_DIR / "masks"

UNPACKED_IMAGES_DIR = DATA_DIR / "unpacked" / "images"
UNPACKED_MASKS_DIR = DATA_DIR / "unpacked" / "masks"
UNPACKED_METADATA_CSV = DATA_DIR / "unpacked" / "volume_metadata.csv"

# =============================================================================
# Dataset Files
# =============================================================================

OVERVIEW_CSV = RAW_DATA_DIR / "overview.csv"
RADIOLOGICAL_GRADINGS_CSV = RAW_DATA_DIR / "radiological_gradings.csv"

MASTER_GT_CSV = PROCESSED_DATA_DIR / "master_gt.csv"
DATASET_CSV = PROCESSED_DATA_DIR / "dataset.csv"

# =============================================================================
# Output Directories
# =============================================================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"

# =============================================================================
# Project Directories
# =============================================================================

EDA_DIR = PROJECT_ROOT / "eda"
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
TRAINING_DIR = PROJECT_ROOT / "training"
VISUALIZATION_DIR = PROJECT_ROOT / "visualization"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
UTILITIES_DIR = PROJECT_ROOT / "utilities"