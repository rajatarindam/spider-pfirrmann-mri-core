"""
Ground Truth Analysis

This module analyzes the SPIDER radiological ground truth for the
Sagittal T2 patient cohort.

Responsibilities:
- Identify Sagittal T2 patients from overview.csv.
- Exclude the patient with insufficient T2 slices for the planned
  slice-offset strategy.
- Analyze radiological grading records for the remaining patients.
- Count Pfirrmann grades.
- Count disc labels.
- Count total valid discs.
- Check missing values and duplicate records.
- Save a ground-truth summary CSV.
- Generate an execution log.

No image loading, preprocessing, visualization, or ground-truth
inflation is performed in this module.
"""

import sys

import pandas as pd
from colorama import Fore, init

from configs.paths import (
    OUTPUTS_DIR,
    OVERVIEW_CSV,
    RADIOLOGICAL_GRADINGS_CSV,
)
from utilities.logger import ExecutionLogger

init(autoreset=True)


# =============================================================================
# Configuration
# =============================================================================

SLICE_STATISTICS_CSV = OUTPUTS_DIR / "eda" / "slice_statistics.csv"


# =============================================================================
# Helper Functions
# =============================================================================


def load_overview() -> pd.DataFrame:
    """
    Load the SPIDER overview CSV.
    """

    return pd.read_csv(OVERVIEW_CSV)


def load_radiological_gradings() -> pd.DataFrame:
    """
    Load the SPIDER radiological grading CSV.
    """

    return pd.read_csv(RADIOLOGICAL_GRADINGS_CSV)


def get_t2_patient_ids(overview: pd.DataFrame) -> set[int]:
    """
    Extract patient IDs belonging to Sagittal T2 studies.
    """

    t2_rows = overview[
        overview["new_file_name"]
        .astype(str)
        .str.lower()
        .str.endswith("_t2")
    ]

    patient_ids = (
        t2_rows["new_file_name"]
        .astype(str)
        .str.extract(r"^(\d+)_t2$")[0]
        .dropna()
        .astype(int)
    )

    return set(patient_ids)


def get_insufficient_slice_patients() -> set[int]:
    """
    Identify patients whose T2 study has insufficient slice count.

    The previously generated slice statistics are used so that this
    module does not need to load MRI volumes again.
    """

    if not SLICE_STATISTICS_CSV.exists():
        raise FileNotFoundError(
            f"Required slice statistics file was not found: "
            f"{SLICE_STATISTICS_CSV}"
        )

    slice_statistics = pd.read_csv(SLICE_STATISTICS_CSV)

    if "MRI Filename" not in slice_statistics.columns:
        raise ValueError(
            "slice_statistics.csv is missing the 'MRI Filename' column."
        )

    if "Slice Count" not in slice_statistics.columns:
        raise ValueError(
            "slice_statistics.csv is missing the 'Slice Count' column."
        )

    minimum_slice_count = slice_statistics["Slice Count"].min()

    insufficient_rows = slice_statistics[
        slice_statistics["Slice Count"] == minimum_slice_count
    ]

    patient_ids = (
        insufficient_rows["MRI Filename"]
        .astype(str)
        .str.extract(r"^(\d+)_t2\.mha$")[0]
        .dropna()
        .astype(int)
    )

    return set(patient_ids)


def analyze_ground_truth(
    overview: pd.DataFrame,
    gradings: pd.DataFrame,
):
    """
    Analyze the radiological ground truth for the valid T2 cohort.
    """

    t2_patient_ids = get_t2_patient_ids(overview)

    excluded_patients = get_insufficient_slice_patients()

    valid_patient_ids = t2_patient_ids - excluded_patients

    filtered_gradings = gradings[
        gradings["Patient"].isin(valid_patient_ids)
    ].copy()

    # IVD label 0 is not a valid disc label.
    valid_disc_records = filtered_gradings[
        filtered_gradings["IVD label"] > 0
    ].copy()

    summary = {
        "Total T2 Patients": len(t2_patient_ids),
        "Excluded Patients": len(excluded_patients),
        "Excluded Patient IDs": ", ".join(
            str(patient_id)
            for patient_id in sorted(excluded_patients)
        ),
        "Remaining T2 Patients": len(valid_patient_ids),
        "Radiological Records": len(filtered_gradings),
        "Valid Disc Records": len(valid_disc_records),
        "Unique Disc Labels": valid_disc_records["IVD label"].nunique(),
        "Minimum Disc Label": valid_disc_records["IVD label"].min(),
        "Maximum Disc Label": valid_disc_records["IVD label"].max(),
        "Missing Values": int(filtered_gradings.isna().sum().sum()),
        "Duplicate Records": int(filtered_gradings.duplicated().sum()),
    }

    grade_distribution = (
        valid_disc_records["Pfirrman grade"]
        .value_counts()
        .sort_index()
    )

    for grade in range(1, 6):
        summary[f"Pfirrmann Grade {grade}"] = int(
            grade_distribution.get(grade, 0)
        )

    disc_label_distribution = (
        valid_disc_records["IVD label"]
        .value_counts()
        .sort_index()
    )

    for label in sorted(disc_label_distribution.index):
        summary[f"Disc Label {label}"] = int(
            disc_label_distribution[label]
        )

    return summary


def save_summary(summary: dict) -> None:
    """
    Save the ground-truth summary CSV.
    """

    output_dir = OUTPUTS_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(
        {
            "Metric": summary.keys(),
            "Value": summary.values(),
        }
    )

    dataframe.to_csv(
        output_dir / "ground_truth_summary.csv",
        index=False,
    )


def print_summary(summary: dict) -> None:
    """
    Print the ground-truth summary.
    """

    print("\n" + "=" * 65)
    print(Fore.CYAN + "GROUND TRUTH ANALYSIS")
    print("=" * 65)

    for key, value in summary.items():
        print(f"{key:<35} : {value}")

    print("=" * 65 + "\n")


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Execute ground-truth analysis.
    """

    logger = ExecutionLogger("ground_truth_analysis")

    try:
        overview = load_overview()
        gradings = load_radiological_gradings()

        summary = analyze_ground_truth(
            overview,
            gradings,
        )

        save_summary(summary)
        print_summary(summary)

        logger.add(
            "Overview CSV",
            str(OVERVIEW_CSV),
        )

        logger.add(
            "Radiological Gradings CSV",
            str(RADIOLOGICAL_GRADINGS_CSV),
        )

        logger.add(
            "Slice Statistics CSV",
            str(SLICE_STATISTICS_CSV),
        )

        for key, value in summary.items():
            logger.add(key, value)

        logger.save("SUCCESS")

        print(
            Fore.GREEN
            + "Ground-truth analysis completed successfully."
        )

    except Exception as exception:
        logger.error(str(exception))
        logger.save("FAILED")

        print(
            Fore.RED
            + f"\nERROR: {exception}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()