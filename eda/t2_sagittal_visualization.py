"""
T2 Sagittal Slice Visualization

Visualizes the medial Sagittal T2 slice and corresponding segmentation
mask for three representative SPIDER patients.

Responsibilities:
- Select three retained T2 patients.
- Determine the medial slice.
- Extract the corresponding MRI and mask slices.
- Generate MRI, mask, and MRI-mask overlay views.
- Save one landscape figure per patient.

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

NUM_PATIENTS = 3
EXCLUDED_PATIENTS = {58}

OUTPUT_DIR = OUTPUTS_DIR / "eda" / "t2_visualization"


# =============================================================================
# Helper Functions
# =============================================================================


def get_t2_files():
    """
    Return valid Sagittal T2 image files after excluding patient 58.
    """

    t2_files = []

    for file in sorted(RAW_IMAGES_DIR.glob("*_t2.mha")):
        patient_id = int(file.stem.split("_")[0])

        if patient_id in EXCLUDED_PATIENTS:
            continue

        t2_files.append(file)

    return t2_files


def get_medial_slice(image):
    """
    Extract the medial slice from a 3D image.

    The SPIDER dataset contains different axis storage conventions.
    The validated slice dimension is the smallest image dimension.
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
    Normalize an MRI image only for visualization.
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
    mask_slice,
    slice_axis,
    slice_index,
):
    """
    Create and save a three-panel landscape visualization.
    """

    mri_display = normalize_for_display(mri_slice)

    mask_binary = mask_slice > 0

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    axes[0].imshow(
        mri_display,
        cmap="gray",
    )
    axes[0].set_title("Sagittal T2 MRI")
    axes[0].axis("off")

    axes[1].imshow(
        mask_binary,
        cmap="gray",
    )
    axes[1].set_title("Segmentation Mask")
    axes[1].axis("off")

    axes[2].imshow(
        mri_display,
        cmap="gray",
    )
    axes[2].imshow(
        mask_binary,
        cmap="autumn",
        alpha=0.35,
    )
    axes[2].set_title("MRI + Mask Overlay")
    axes[2].axis("off")

    figure.suptitle(
        f"Patient {patient_id} | "
        f"Slice Axis: {slice_axis} | "
        f"Medial Slice Index: {slice_index}",
        fontsize=14,
    )

    figure.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"{patient_id}_t2_medial_visualization.png"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Generate medial T2 visualizations for three patients.
    """

    logger = ExecutionLogger(
        "t2_sagittal_visualization"
    )

    try:

        t2_files = get_t2_files()

        if len(t2_files) < NUM_PATIENTS:
            raise RuntimeError(
                f"Only {len(t2_files)} valid T2 files were found. "
                f"{NUM_PATIENTS} are required."
            )

        selected_files = t2_files[:NUM_PATIENTS]

        print("\n" + "=" * 70)
        print(
            Fore.CYAN
            + "T2 SAGITTAL MEDIAL SLICE VISUALIZATION"
        )
        print("=" * 70)

        for image_path in selected_files:

            patient_id = int(
                image_path.stem.split("_")[0]
            )

            mask_path = (
                RAW_MASKS_DIR
                / image_path.name
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Corresponding mask not found: {mask_path}"
                )

            image = sitk.ReadImage(
                str(image_path)
            )

            mask = sitk.ReadImage(
                str(mask_path)
            )

            mri_slice, slice_axis, slice_index = (
                get_medial_slice(image)
            )

            mask_slice, mask_axis, mask_index = (
                get_medial_slice(mask)
            )

            if (
                slice_axis != mask_axis
                or slice_index != mask_index
            ):
                raise RuntimeError(
                    f"MRI and mask slice selection mismatch "
                    f"for patient {patient_id}."
                )

            if mri_slice.shape != mask_slice.shape:
                raise RuntimeError(
                    f"MRI and mask slice shapes do not match "
                    f"for patient {patient_id}: "
                    f"{mri_slice.shape} vs {mask_slice.shape}"
                )

            output_path = create_visualization(
                patient_id=patient_id,
                mri_slice=mri_slice,
                mask_slice=mask_slice,
                slice_axis=slice_axis,
                slice_index=slice_index,
            )

            print(
                f"Patient {patient_id:<5} "
                f"| Slice axis: {slice_axis} "
                f"| Medial index: {slice_index} "
                f"| Output: {output_path.name}"
            )

        print("=" * 70)
        print(
            Fore.GREEN
            + "T2 visualization completed successfully."
        )
        print("=" * 70 + "\n")

        logger.add(
            "Patients Visualized",
            NUM_PATIENTS,
        )

        logger.add(
            "Excluded Patients",
            ", ".join(
                str(patient)
                for patient in sorted(EXCLUDED_PATIENTS)
            ),
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