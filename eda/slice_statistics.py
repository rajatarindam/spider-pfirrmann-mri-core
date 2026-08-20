"""
Slice Statistics

This module computes slice statistics for all Sagittal T2 MRI volumes.

Responsibilities:
- Read all T2 MRI volumes.
- Compute slice count for each study.
- Calculate summary statistics.
- Save per-study statistics.
- Save summary statistics.
- Generate execution log.

No visualization, preprocessing, unpacking, or model-related operations
are performed in this module.
"""

from statistics import mean, median, stdev
import sys

import pandas as pd
import SimpleITK as sitk
from colorama import Fore, init

from configs.paths import (
    OUTPUTS_DIR,
    RAW_IMAGES_DIR,
)
from utilities.logger import ExecutionLogger

init(autoreset=True)


# =============================================================================
# Helper Functions
# =============================================================================


def get_t2_files():
    """
    Return all Sagittal T2 MRI volumes.
    Excludes T2 SPACE.
    """

    t2_files = []

    for file in sorted(RAW_IMAGES_DIR.glob("*.mha")):
        filename = file.stem.lower()

        if "_t2_space" in filename:
            continue

        if "_t2" in filename:
            t2_files.append(file)

    return t2_files


def get_slice_count(image_path):
    """
    Return the number of slices in an MRI volume.

    The SPIDER dataset contains studies stored with different axis
    orientations. The slice dimension is therefore determined as the
    smallest image dimension.
    """

    image = sitk.ReadImage(str(image_path))

    return min(image.GetSize())


def compute_statistics(slice_counts):
    """
    Compute summary statistics.
    """

    return {
        "Total T2 Studies": len(slice_counts),
        "Minimum Slice Count": min(slice_counts),
        "Maximum Slice Count": max(slice_counts),
        "Mean Slice Count": round(mean(slice_counts), 2),
        "Median Slice Count": median(slice_counts),
        "Standard Deviation": round(
            stdev(slice_counts),
            2,
        ),
    }


def save_per_study_csv(records):
    """
    Save slice count for every T2 study.
    """

    output_dir = OUTPUTS_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        output_dir / "slice_statistics.csv",
        index=False,
    )


def save_summary_csv(summary):
    """
    Save summary statistics.
    """

    output_dir = OUTPUTS_DIR / "eda"

    dataframe = pd.DataFrame(
        {
            "Metric": summary.keys(),
            "Value": summary.values(),
        }
    )

    dataframe.to_csv(
        output_dir / "slice_statistics_summary.csv",
        index=False,
    )


def print_summary(summary):
    """
    Print summary statistics.
    """

    print("\n" + "=" * 60)
    print(Fore.CYAN + "SLICE STATISTICS")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key:<35} : {value}")

    print("=" * 60 + "\n")


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Execute slice statistics.
    """

    logger = ExecutionLogger("slice_statistics")

    try:

        t2_files = get_t2_files()

        records = []
        slice_counts = []

        for file in t2_files:

            slices = get_slice_count(file)

            slice_counts.append(slices)

            records.append(
                {
                    "MRI Filename": file.name,
                    "Slice Count": slices,
                }
            )

        summary = compute_statistics(slice_counts)

        save_per_study_csv(records)

        save_summary_csv(summary)

        print_summary(summary)

        logger.add(
            "Configuration",
            f"T2 Studies: {len(t2_files)}",
        )

        for key, value in summary.items():
            logger.add(key, value)

        logger.save("SUCCESS")

        print(
            Fore.GREEN
            + "Slice statistics completed successfully."
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