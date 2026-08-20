"""
Dataset Inventory

This module performs a structural inventory of the raw SPIDER dataset.

Responsibilities:
- Verify the expected dataset structure.
- Count MRI image volumes.
- Count segmentation mask volumes.
- Count MRI series types (T1, T2, T2 SPACE).
- Verify image-mask filename correspondence.
- Detect duplicate filenames.
- Generate an inventory summary CSV.
- Generate an execution log.

No image loading, preprocessing, visualization, or model-related operations
are performed in this module.
"""

from collections import Counter
from pathlib import Path
import sys

import pandas as pd
from colorama import Fore, Style, init

from configs.paths import (
    RAW_IMAGES_DIR,
    RAW_MASKS_DIR,
    OVERVIEW_CSV,
    RADIOLOGICAL_GRADINGS_CSV,
    OUTPUTS_DIR,
)
from utilities.logger import ExecutionLogger

init(autoreset=True)


# =============================================================================
# Helper Functions
# =============================================================================


def verify_dataset_structure() -> None:
    """
    Verify that the required dataset directories and files exist.

    Raises
    ------
    FileNotFoundError
        If any required directory or file is missing.
    """

    required_paths = [
        RAW_IMAGES_DIR,
        RAW_MASKS_DIR,
        OVERVIEW_CSV,
        RADIOLOGICAL_GRADINGS_CSV,
    ]

    missing = [path for path in required_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required dataset items:\n"
            + "\n".join(str(item) for item in missing)
        )


def get_mha_files(directory: Path) -> list[Path]:
    """
    Return all .mha files inside a directory.
    """

    return sorted(directory.glob("*.mha"))


def count_series_types(files: list[Path]) -> dict:
    """
    Count MRI series based on filename.
    """

    counts = {
        "T1": 0,
        "T2": 0,
        "T2 SPACE": 0,
    }

    for file in files:
        name = file.stem.lower()

        if "_t2_space" in name:
            counts["T2 SPACE"] += 1
        elif "_t2" in name:
            counts["T2"] += 1
        elif "_t1" in name:
            counts["T1"] += 1

    return counts


def detect_duplicates(files: list[Path]) -> int:
    """
    Detect duplicate filenames.
    """

    names = [file.name for file in files]
    duplicates = sum(count > 1 for count in Counter(names).values())

    return duplicates


def compare_image_and_mask_files(
    image_files: list[Path],
    mask_files: list[Path],
):
    """
    Compare image and mask filenames.
    """

    image_names = {file.name for file in image_files}
    mask_names = {file.name for file in mask_files}

    missing_masks = sorted(image_names - mask_names)
    missing_images = sorted(mask_names - image_names)

    return missing_images, missing_masks


def save_inventory_csv(summary: dict) -> None:
    """
    Save inventory summary as CSV.
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
        output_dir / "inventory_summary.csv",
        index=False,
    )


def print_summary(summary: dict) -> None:
    """
    Print inventory summary.
    """

    print("\n" + "=" * 60)
    print(Fore.CYAN + "SPIDER DATASET INVENTORY")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key:<35} : {value}")

    print("=" * 60 + "\n")


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Execute dataset inventory.
    """

    logger = ExecutionLogger("dataset_inventory")

    try:
        verify_dataset_structure()

        image_files = get_mha_files(RAW_IMAGES_DIR)
        mask_files = get_mha_files(RAW_MASKS_DIR)

        series_counts = count_series_types(image_files)

        missing_images, missing_masks = compare_image_and_mask_files(
            image_files,
            mask_files,
        )

        duplicate_images = detect_duplicates(image_files)
        duplicate_masks = detect_duplicates(mask_files)

        summary = {
            "Total Image Volumes": len(image_files),
            "Total Mask Volumes": len(mask_files),
            "Total T1 Series": series_counts["T1"],
            "Total T2 Series": series_counts["T2"],
            "Total T2 SPACE Series": series_counts["T2 SPACE"],
            "Missing Images": len(missing_images),
            "Missing Masks": len(missing_masks),
            "Duplicate Image Filenames": duplicate_images,
            "Duplicate Mask Filenames": duplicate_masks,
        }

        print_summary(summary)

        save_inventory_csv(summary)

        logger.add("Configuration", f"Images: {RAW_IMAGES_DIR}")
        logger.add("Configuration", f"Masks: {RAW_MASKS_DIR}")

        for key, value in summary.items():
            logger.add(key, value)

        logger.save("SUCCESS")

        print(
            Fore.GREEN
            + "Dataset inventory completed successfully."
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