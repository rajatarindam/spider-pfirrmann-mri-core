"""
Image Metadata Inspection

This module inspects all Sagittal T2 MRI volumes and reports
their metadata to help understand the dataset structure.

Responsibilities:
- Read all T2 MRI volumes.
- Extract image metadata.
- Save metadata as CSV.
- Print summary information.
- Generate execution log.

No visualization, preprocessing, or image modification is performed.
"""

import sys

import pandas as pd
import SimpleITK as sitk
from colorama import Fore, init

from configs.paths import OUTPUTS_DIR, RAW_IMAGES_DIR
from utilities.logger import ExecutionLogger

init(autoreset=True)


# =============================================================================
# Helper Functions
# =============================================================================


def get_t2_files():
    """
    Return all Sagittal T2 MRI volumes.
    """

    t2_files = []

    for file in sorted(RAW_IMAGES_DIR.glob("*.mha")):
        filename = file.stem.lower()

        if "_t2_space" in filename:
            continue

        if "_t2" in filename:
            t2_files.append(file)

    return t2_files


def extract_metadata(image_path):
    """
    Extract metadata from a single MRI volume.
    """

    image = sitk.ReadImage(str(image_path))

    return {
        "MRI Filename": image_path.name,
        "Size (X)": image.GetSize()[0],
        "Size (Y)": image.GetSize()[1],
        "Size (Z)": image.GetSize()[2],
        "Spacing": image.GetSpacing(),
        "Origin": image.GetOrigin(),
        "Direction": image.GetDirection(),
        "Pixel Type": image.GetPixelIDTypeAsString(),
        "Dimension": image.GetDimension(),
    }


def save_metadata(records):
    """
    Save metadata CSV.
    """

    output_dir = OUTPUTS_DIR / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(records)

    dataframe.to_csv(
        output_dir / "image_metadata.csv",
        index=False,
    )


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Execute image metadata inspection.
    """

    logger = ExecutionLogger("image_metadata")

    try:

        t2_files = get_t2_files()

        records = []

        for file in t2_files:
            records.append(extract_metadata(file))

        save_metadata(records)

        print("\n" + "=" * 70)
        print(Fore.CYAN + "IMAGE METADATA INSPECTION")
        print("=" * 70)

        print(f"Total T2 Studies : {len(records)}")

        if records:
            first = records[0]

            print("\nMetadata from first T2 study:")
            print(f"Size (X)     : {first['Size (X)']}")
            print(f"Size (Y)     : {first['Size (Y)']}")
            print(f"Size (Z)     : {first['Size (Z)']}")
            print(f"Spacing      : {first['Spacing']}")
            print(f"Pixel Type   : {first['Pixel Type']}")
            print(f"Dimension    : {first['Dimension']}")

        print("=" * 70 + "\n")

        logger.add("Total T2 Studies", len(records))
        logger.add("Output CSV", "outputs/eda/image_metadata.csv")

        logger.save("SUCCESS")

        print(
            Fore.GREEN
            + "Image metadata inspection completed successfully."
        )

    except Exception as exception:

        logger.error(str(exception))
        logger.save("FAILED")

        print(Fore.RED + f"\nERROR: {exception}")

        sys.exit(1)


if __name__ == "__main__":
    main()