"""
Create patient-wise Train / Validation / Test split.

Input:
    data/processed/master_gt.csv

Output:
    data/processed/dataset.csv

Important:
    - Split is performed at Patient_ID level.
    - No patient can appear in more than one split.
    - All existing master GT columns are preserved.
    - Only one new column is added: Split.
    - Fixed random seed makes the split reproducible.
"""

from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MASTER_GT_PATH = PROJECT_ROOT / "data" / "processed" / "master_gt.csv"
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dataset.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================================
# VALIDATE CONFIGURATION
# ============================================================================

def validate_configuration() -> None:
    total = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"Train/Val/Test ratios must sum to 1.0, got {total}"
        )

    if not MASTER_GT_PATH.exists():
        raise FileNotFoundError(
            f"Master GT not found:\n{MASTER_GT_PATH}"
        )


# ============================================================================
# LOAD MASTER GT
# ============================================================================

def load_master_gt() -> pd.DataFrame:
    df = pd.read_csv(MASTER_GT_PATH)

    required_columns = {
        "Patient_ID",
        "Pfirrmann_Grade",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Master GT is missing required columns: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Master GT is empty.")

    if df["Patient_ID"].isna().any():
        raise ValueError(
            "Patient_ID contains missing values."
        )

    return df


# ============================================================================
# CREATE PATIENT-WISE SPLIT
# ============================================================================

def create_patient_split(
    patient_ids: np.ndarray,
) -> dict[int, str]:

    rng = np.random.default_rng(RANDOM_SEED)

    patients = np.array(patient_ids, dtype=int)

    # Shuffle patients, not individual samples.
    rng.shuffle(patients)

    total_patients = len(patients)

    # Use floor for train.
    train_count = int(total_patients * TRAIN_RATIO)

    # Use floor for validation.
    val_count = int(total_patients * VAL_RATIO)

    # Give all remaining patients to test.
    test_count = total_patients - train_count - val_count

    train_patients = patients[:train_count]

    val_patients = patients[
        train_count:train_count + val_count
    ]

    test_patients = patients[
        train_count + val_count:
    ]

    split_map: dict[int, str] = {}

    for patient_id in train_patients:
        split_map[int(patient_id)] = "train"

    for patient_id in val_patients:
        split_map[int(patient_id)] = "val"

    for patient_id in test_patients:
        split_map[int(patient_id)] = "test"

    print()
    print("=" * 70)
    print("PATIENT-WISE SPLIT")
    print("=" * 70)

    print(f"Total patients : {total_patients}")
    print(f"Train patients : {train_count}")
    print(f"Val patients   : {val_count}")
    print(f"Test patients  : {test_count}")
    print(f"Random seed    : {RANDOM_SEED}")

    return split_map


# ============================================================================
# APPLY SPLIT
# ============================================================================

def apply_split(
    df: pd.DataFrame,
    split_map: dict[int, str],
) -> pd.DataFrame:

    df = df.copy()

    df["Split"] = df["Patient_ID"].astype(int).map(split_map)

    if df["Split"].isna().any():
        missing_patients = sorted(
            df.loc[df["Split"].isna(), "Patient_ID"]
            .astype(int)
            .unique()
        )

        raise RuntimeError(
            "Some patients were not assigned a split: "
            f"{missing_patients}"
        )

    return df


# ============================================================================
# VALIDATE PATIENT OVERLAP
# ============================================================================

def validate_patient_overlap(
    df: pd.DataFrame,
) -> None:

    train_patients = set(
        df.loc[df["Split"] == "train", "Patient_ID"]
        .astype(int)
    )

    val_patients = set(
        df.loc[df["Split"] == "val", "Patient_ID"]
        .astype(int)
    )

    test_patients = set(
        df.loc[df["Split"] == "test", "Patient_ID"]
        .astype(int)
    )

    train_val_overlap = train_patients & val_patients
    train_test_overlap = train_patients & test_patients
    val_test_overlap = val_patients & test_patients

    if train_val_overlap:
        raise RuntimeError(
            f"Train/Val patient leakage detected: "
            f"{sorted(train_val_overlap)}"
        )

    if train_test_overlap:
        raise RuntimeError(
            f"Train/Test patient leakage detected: "
            f"{sorted(train_test_overlap)}"
        )

    if val_test_overlap:
        raise RuntimeError(
            f"Val/Test patient leakage detected: "
            f"{sorted(val_test_overlap)}"
        )

    print()
    print("Patient leakage check : PASSED")
    print("Train ∩ Val           : 0")
    print("Train ∩ Test          : 0")
    print("Val ∩ Test            : 0")


# ============================================================================
# VALIDATE SAMPLE COUNTS
# ============================================================================

def print_sample_statistics(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)

    for split_name in ["train", "val", "test"]:

        subset = df[df["Split"] == split_name]

        patient_count = subset["Patient_ID"].nunique()
        sample_count = len(subset)

        print(
            f"{split_name.upper():5s} | "
            f"Patients: {patient_count:3d} | "
            f"Samples: {sample_count:5d}"
        )


# ============================================================================
# PFIRRMANN DISTRIBUTION
# ============================================================================

def print_grade_distribution(
    df: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("PFIRRMANN GRADE DISTRIBUTION")
    print("=" * 70)

    for split_name in ["train", "val", "test"]:

        subset = df[df["Split"] == split_name]

        counts = (
            subset["Pfirrmann_Grade"]
            .value_counts()
            .sort_index()
        )

        total = len(subset)

        print()
        print(f"{split_name.upper()}")

        for grade in range(1, 6):

            count = int(counts.get(grade, 0))

            percentage = (
                100.0 * count / total
                if total > 0
                else 0.0
            )

            print(
                f"  Grade {grade}: "
                f"{count:5d} "
                f"({percentage:6.2f}%)"
            )


# ============================================================================
# SAVE DATASET
# ============================================================================

def save_dataset(
    df: pd.DataFrame,
) -> None:

    DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        DATASET_PATH,
        index=False,
    )

    print()
    print("=" * 70)
    print("DATASET SAVED")
    print("=" * 70)

    print(f"Output : {DATASET_PATH}")
    print(f"Rows   : {len(df)}")
    print(f"Columns: {len(df.columns)}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 70)
    print("PATIENT-WISE DATASET SPLIT")
    print("=" * 70)

    validate_configuration()

    print(f"Master GT : {MASTER_GT_PATH}")
    print(f"Output     : {DATASET_PATH}")

    df = load_master_gt()

    print()
    print(f"Master GT rows     : {len(df)}")
    print(
        f"Unique patients    : "
        f"{df['Patient_ID'].nunique()}"
    )

    patient_ids = (
        df["Patient_ID"]
        .astype(int)
        .unique()
    )

    split_map = create_patient_split(
        patient_ids
    )

    df = apply_split(
        df,
        split_map,
    )

    validate_patient_overlap(df)

    print_sample_statistics(df)

    print_grade_distribution(df)

    save_dataset(df)

    print()
    print("=" * 70)
    print("SPLIT COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()