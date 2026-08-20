"""
Disc Mask Visualization

Visualizes disc masks for three representative Sagittal T2 studies.

The purpose of this module is to inspect:
- Disc mask values
- Individual disc regions
- Number of visible discs
- Spatial relationship between disc masks and MRI anatomy

No preprocessing or dataset generation is performed.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from colorama import Fore, init

from configs.paths import OUTPUTS_DIR, RAW_IMAGES_DIR, RAW_MASKS_DIR
from utilities.logger import ExecutionLogger

init(autoreset=True)


# =============================================================================
# Configuration
# =============================================================================

PATIENT_IDS = [100, 101, 104]

OUTPUT_DIR = OUTPUTS_DIR / "eda" / "disc_mask_visualization"


# =============================================================================
# Helper Functions
# =============================================================================


def get_slice(image):
    """
    Extract the medial slice using the validated smallest-dimension rule.
    """

    size = image.GetSize()

    slice_axis = int(np.argmin(size))
    medial_index = size[slice_axis] // 2

    array = sitk.GetArrayFromImage(image)

    if slice_axis == 0:
        slice_image = array[:, :, medial_index]

    elif slice_axis == 1:
        slice_image = array[:, medial_index, :]

    else:
        slice_image = array[medial_index, :, :]

    return slice_image, slice_axis, medial_index


def normalize_for_display(image):
    """
    Normalize an MRI slice for visualization only.
    """

    image = image.astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum == minimum:
        return np.zeros_like(image)

    return (image - minimum) / (maximum - minimum)


def create_visualization(
    patient_id,
    mri_slice,
    disc_mask_slice,
    slice_index,
):
    """
    Create a landscape visualization of the disc mask.
    """

    mri_display = normalize_for_display(mri_slice)

    # Every non-zero value represents a foreground disc-mask region.
    mask_binary = disc_mask_slice > 0

    unique_values = np.unique(disc_mask_slice)
    nonzero_values = unique_values[unique_values != 0]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    # -------------------------------------------------------------------------
    # Panel 1: Raw disc mask
    # -------------------------------------------------------------------------

    axes[0].imshow(
        disc_mask_slice,
        cmap="nipy_spectral",
        interpolation="nearest",
    )

    axes[0].set_title("Disc Mask")
    axes[0].axis("off")

    # -------------------------------------------------------------------------
    # Panel 2: Binary disc mask
    # -------------------------------------------------------------------------

    axes[1].imshow(
        mask_binary,
        cmap="gray",
        interpolation="nearest",
    )

    axes[1].set_title("Binary Disc Mask")
    axes[1].axis("off")

    # -------------------------------------------------------------------------
    # Panel 3: MRI + disc mask
    # -------------------------------------------------------------------------

    axes[2].imshow(
        mri_display,
        cmap="gray",
    )

    axes[2].imshow(
        mask_binary,
        cmap="autumn",
        alpha=0.40,
        interpolation="nearest",
    )

    axes[2].set_title("MRI + Disc Mask")
    axes[2].axis("off")

    figure.suptitle(
        f"Patient {patient_id} | "
        f"Medial Slice Index: {slice_index} | "
        f"Non-zero Mask Values: {len(nonzero_values)}",
        fontsize=14,
    )

    figure.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{patient_id}_disc_mask_visualization.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path, unique_values, nonzero_values


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Generate disc-mask visualizations for three representative patients.
    """

    logger = ExecutionLogger(
        "disc_mask_visualization"
    )

    try:

        print("\n" + "=" * 70)
        print(
            Fore.CYAN
            + "DISC MASK VISUALIZATION"
        )
        print("=" * 70)

        for patient_id in PATIENT_IDS:

            image_path = (
                RAW_IMAGES_DIR
                / f"{patient_id}_t2.mha"
            )

            mask_path = (
                RAW_MASKS_DIR
                / f"{patient_id}_t2.mha"
            )

            if not image_path.exists():
                raise FileNotFoundError(
                    f"MRI file not found: {image_path}"
                )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Disc mask file not found: {mask_path}"
                )

            image = sitk.ReadImage(
                str(image_path)
            )

            disc_mask = sitk.ReadImage(
                str(mask_path)
            )

            mri_slice, image_axis, image_index = get_slice(
                image
            )

            mask_slice, mask_axis, mask_index = get_slice(
                disc_mask
            )

            if (
                image_axis != mask_axis
                or image_index != mask_index
            ):
                raise RuntimeError(
                    f"MRI and disc mask slice selection mismatch "
                    f"for patient {patient_id}."
                )

            if mri_slice.shape != mask_slice.shape:
                raise RuntimeError(
                    f"MRI and disc mask shapes do not match "
                    f"for patient {patient_id}: "
                    f"{mri_slice.shape} vs {mask_slice.shape}"
                )

            (
                output_path,
                unique_values,
                nonzero_values,
            ) = create_visualization(
                patient_id=patient_id,
                mri_slice=mri_slice,
                disc_mask_slice=mask_slice,
                slice_index=mask_index,
            )

            print(
                f"\nPatient {patient_id}"
            )

            print(
                f"  Medial slice index : {mask_index}"
            )

            print(
                f"  Unique mask values : "
                f"{unique_values.tolist()}"
            )

            print(
                f"  Non-zero values    : "
                f"{nonzero_values.tolist()}"
            )

            print(
                f"  Output             : "
                f"{output_path.name}"
            )

        print("\n" + "=" * 70)
        print(
            Fore.GREEN
            + "Disc mask visualization completed successfully."
        )
        print("=" * 70 + "\n")

        logger.add(
            "Patients Visualized",
            ", ".join(str(x) for x in PATIENT_IDS),
        )

        logger.add(
            "Output Directory",
            str(OUTPUT_DIR),
        )

        logger.save("SUCCESS")

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