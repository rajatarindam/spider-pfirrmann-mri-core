"""
Unpack retained Sagittal T2 MRI volumes and disc masks from MHA to PNG.

This module:
- Processes only the 209 retained T2 studies.
- Excludes patient 58.
- Determines the slice axis independently for each volume using
  the smallest image dimension.
- Preserves the original slice order along that axis.
- Saves MRI slices as 16-bit PNG.
- Saves disc-mask slices as 8-bit PNG.
- Preserves MRI intensity information during PNG storage.
- Generates volume-level metadata describing the source representation.

No normalization, resizing, rotation, cropping, or slice selection
is performed in this module.
"""

import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from colorama import Fore, init
from PIL import Image

from configs.paths import (
    RAW_IMAGES_DIR,
    RAW_MASKS_DIR,
    UNPACKED_IMAGES_DIR,
    UNPACKED_MASKS_DIR,
    UNPACKED_METADATA_CSV,
)
from utilities.logger import ExecutionLogger


init(autoreset=True)


# =============================================================================
# Configuration
# =============================================================================

EXCLUDED_PATIENTS = {58}

IMAGE_PATTERN = "*_t2.mha"

MRI_OFFSET = 32768


# =============================================================================
# Helper Functions
# =============================================================================


def get_patient_id(file_path):
    """
    Extract the patient ID from an MRI filename.
    """

    return int(file_path.stem.split("_")[0])


def get_retained_t2_files():
    """
    Return all retained Sagittal T2 MRI files.
    """

    t2_files = []

    for file_path in sorted(
        RAW_IMAGES_DIR.glob(IMAGE_PATTERN)
    ):

        patient_id = get_patient_id(
            file_path
        )

        if patient_id in EXCLUDED_PATIENTS:
            continue

        t2_files.append(file_path)

    return t2_files


def determine_slice_axis(image):
    """
    Determine the slice axis from the smallest image dimension.

    SimpleITK image size is returned as:

        (X, Y, Z)

    The smallest dimension is treated as the slice dimension.

    Returns:
        slice_axis_index:
            0 = X
            1 = Y
            2 = Z

        slice_axis_name:
            "X", "Y", or "Z"

        slice_count:
            Number of slices along the selected axis.
    """

    size = image.GetSize()

    if len(size) != 3:
        raise ValueError(
            f"Expected a 3D image, found "
            f"dimension {len(size)}."
        )

    slice_axis_index = min(
        range(3),
        key=lambda axis: size[axis],
    )

    axis_names = {
        0: "X",
        1: "Y",
        2: "Z",
    }

    slice_axis_name = axis_names[
        slice_axis_index
    ]

    slice_count = size[
        slice_axis_index
    ]

    return (
        slice_axis_index,
        slice_axis_name,
        slice_count,
    )


def get_numpy_axis_from_sitk_axis(
    sitk_axis_index,
):
    """
    Convert a SimpleITK image axis index to the corresponding
    NumPy array axis returned by GetArrayFromImage().

    SimpleITK:
        (X, Y, Z)

    NumPy:
        (Z, Y, X)

    Therefore:

        X -> NumPy axis 2
        Y -> NumPy axis 1
        Z -> NumPy axis 0
    """

    axis_mapping = {
        0: 2,  # X -> NumPy axis 2
        1: 1,  # Y -> NumPy axis 1
        2: 0,  # Z -> NumPy axis 0
    }

    return axis_mapping[
        sitk_axis_index
    ]


def save_mri_slice(
    slice_array,
    output_path,
    pixel_type,
):
    """
    Save one MRI slice as a 16-bit PNG.

    Signed 16-bit MRI:
        Original int16 values are shifted by +32768
        for storage in uint16 PNG.

    Unsigned 16-bit MRI:
        Values are stored directly as uint16.

    Both approaches are lossless.
    """

    if pixel_type == sitk.sitkInt16:

        slice_array = slice_array.astype(
            np.int16
        )

        slice_uint16 = (
            slice_array.astype(
                np.int32
            )
            + MRI_OFFSET
        )

        if (
            slice_uint16.min() < 0
            or slice_uint16.max() > 65535
        ):
            raise ValueError(
                "Signed MRI conversion produced "
                "values outside uint16 range."
            )

        slice_uint16 = slice_uint16.astype(
            np.uint16
        )

    elif pixel_type == sitk.sitkUInt16:

        slice_uint16 = slice_array.astype(
            np.uint16
        )

    else:

        raise ValueError(
            "Unsupported MRI pixel type for "
            f"PNG conversion: "
            f"{sitk.GetPixelIDValueAsString(pixel_type)}"
        )

    Image.fromarray(
        slice_uint16,
        mode="I;16",
    ).save(
        output_path
    )


def save_disc_mask_slice(
    slice_array,
    output_path,
):
    """
    Save one disc-mask slice as an 8-bit PNG.

    The current disc-mask labels fit within uint8.
    """

    if slice_array.min() < 0:
        raise ValueError(
            "Disc mask contains negative values."
        )

    if slice_array.max() > 255:
        raise ValueError(
            "Disc mask contains values greater than 255."
        )

    slice_uint8 = slice_array.astype(
        np.uint8
    )

    Image.fromarray(
        slice_uint8,
        mode="L",
    ).save(
        output_path
    )


def save_mri_volume_slices(
    image,
    image_path,
    slice_axis_index,
    slice_count,
):
    """
    Extract and save all MRI slices along the selected slice axis.

    The original slice ordering is preserved.
    """

    array = sitk.GetArrayFromImage(
        image
    )

    expected_shape = tuple(
        reversed(
            image.GetSize()
        )
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected NumPy shape for "
            f"{image_path.name}: "
            f"expected {expected_shape}, "
            f"got {array.shape}"
        )

    numpy_axis = (
        get_numpy_axis_from_sitk_axis(
            slice_axis_index
        )
    )

    output_dir = (
        UNPACKED_IMAGES_DIR
        / image_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for slice_index in range(
        slice_count
    ):

        slice_array = np.take(
            array,
            slice_index,
            axis=numpy_axis,
        )

        output_path = (
            output_dir
            / f"slice_{slice_index:03d}.png"
        )

        save_mri_slice(
            slice_array=slice_array,
            output_path=output_path,
            pixel_type=image.GetPixelID(),
        )


def save_disc_mask_volume_slices(
    mask,
    mask_path,
    slice_axis_index,
    slice_count,
):
    """
    Extract and save all disc-mask slices along the same
    anatomical slice axis used for the MRI volume.
    """

    array = sitk.GetArrayFromImage(
        mask
    )

    expected_shape = tuple(
        reversed(
            mask.GetSize()
        )
    )

    if array.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected NumPy shape for "
            f"{mask_path.name}: "
            f"expected {expected_shape}, "
            f"got {array.shape}"
        )

    numpy_axis = (
        get_numpy_axis_from_sitk_axis(
            slice_axis_index
        )
    )

    output_dir = (
        UNPACKED_MASKS_DIR
        / mask_path.stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for slice_index in range(
        slice_count
    ):

        slice_array = np.take(
            array,
            slice_index,
            axis=numpy_axis,
        )

        output_path = (
            output_dir
            / f"slice_{slice_index:03d}.png"
        )

        save_disc_mask_slice(
            slice_array=slice_array,
            output_path=output_path,
        )


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Unpack all retained Sagittal T2 MRI and disc-mask volumes.
    """

    logger = ExecutionLogger(
        "unpack_t2_volumes"
    )

    metadata_records = []

    try:

        print("\n" + "=" * 70)
        print(
            Fore.CYAN
            + "T2 VOLUME UNPACKING"
        )
        print("=" * 70)

        t2_files = get_retained_t2_files()

        if len(t2_files) != 209:
            raise RuntimeError(
                f"Expected 209 retained T2 studies, "
                f"found {len(t2_files)}."
            )

        total_mri_slices = 0
        total_mask_slices = 0

        # ---------------------------------------------------------------------
        # Process each retained T2 study
        # ---------------------------------------------------------------------

        for index, image_path in enumerate(
            t2_files,
            start=1,
        ):

            patient_id = get_patient_id(
                image_path
            )

            mask_path = (
                RAW_MASKS_DIR
                / image_path.name
            )

            if not mask_path.exists():
                raise FileNotFoundError(
                    f"Corresponding disc mask not found "
                    f"for patient {patient_id}: "
                    f"{mask_path}"
                )

            # -----------------------------------------------------------------
            # Read MRI and disc mask
            # -----------------------------------------------------------------

            image = sitk.ReadImage(
                str(image_path)
            )

            disc_mask = sitk.ReadImage(
                str(mask_path)
            )

            # -----------------------------------------------------------------
            # Validate dimensions
            # -----------------------------------------------------------------

            if (
                image.GetSize()
                != disc_mask.GetSize()
            ):
                raise RuntimeError(
                    f"MRI/disc-mask dimension mismatch "
                    f"for patient {patient_id}: "
                    f"MRI={image.GetSize()}, "
                    f"Disc Mask={disc_mask.GetSize()}"
                )

            # -----------------------------------------------------------------
            # Validate 3D volumes
            # -----------------------------------------------------------------

            if image.GetDimension() != 3:
                raise RuntimeError(
                    f"Expected 3D MRI volume for "
                    f"patient {patient_id}, got "
                    f"{image.GetDimension()}D."
                )

            if disc_mask.GetDimension() != 3:
                raise RuntimeError(
                    f"Expected 3D disc-mask volume for "
                    f"patient {patient_id}, got "
                    f"{disc_mask.GetDimension()}D."
                )

            # -----------------------------------------------------------------
            # Determine slice axis independently for this study
            # -----------------------------------------------------------------

            (
                slice_axis_index,
                slice_axis_name,
                slice_count,
            ) = determine_slice_axis(
                image
            )

            # -----------------------------------------------------------------
            # Pixel type
            # -----------------------------------------------------------------

            pixel_type_name = (
                image.GetPixelIDTypeAsString()
            )

            # -----------------------------------------------------------------
            # Extract MRI
            # -----------------------------------------------------------------

            save_mri_volume_slices(
                image=image,
                image_path=image_path,
                slice_axis_index=slice_axis_index,
                slice_count=slice_count,
            )

            # -----------------------------------------------------------------
            # Extract corresponding disc mask
            # -----------------------------------------------------------------

            save_disc_mask_volume_slices(
                mask=disc_mask,
                mask_path=mask_path,
                slice_axis_index=slice_axis_index,
                slice_count=slice_count,
            )

            # -----------------------------------------------------------------
            # Record volume metadata
            # -----------------------------------------------------------------

            metadata_records.append(
                {
                    "Patient": patient_id,
                    "MRI filename": image_path.name,
                    "Original pixel type": pixel_type_name,
                    "Slice axis": slice_axis_name,
                    "Slice count": slice_count,
                }
            )

            total_mri_slices += slice_count
            total_mask_slices += slice_count

            print(
                f"[{index:03d}/209] "
                f"Patient {patient_id:<5} | "
                f"Size: {image.GetSize()} | "
                f"Slice axis: {slice_axis_name} | "
                f"Slices: {slice_count} | "
                f"Pixel type: {pixel_type_name}"
            )

        # ---------------------------------------------------------------------
        # Final slice-count validation
        # ---------------------------------------------------------------------

        if total_mri_slices != total_mask_slices:
            raise RuntimeError(
                "Total MRI and disc-mask slice counts "
                "do not match."
            )

        # ---------------------------------------------------------------------
        # Save volume metadata
        # ---------------------------------------------------------------------

        metadata_dataframe = pd.DataFrame(
            metadata_records
        )

        UNPACKED_METADATA_CSV.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_dataframe.to_csv(
            UNPACKED_METADATA_CSV,
            index=False,
        )

        # ---------------------------------------------------------------------
        # Final output
        # ---------------------------------------------------------------------

        print("\n" + "=" * 70)

        print(
            f"Retained T2 Studies     : "
            f"{len(t2_files)}"
        )

        print(
            f"Total MRI Slices        : "
            f"{total_mri_slices}"
        )

        print(
            f"Total Disc-Mask Slices  : "
            f"{total_mask_slices}"
        )

        print(
            f"Excluded Patients       : "
            f"{sorted(EXCLUDED_PATIENTS)}"
        )

        print(
            f"Volume Metadata         : "
            f"{UNPACKED_METADATA_CSV}"
        )

        print(
            f"Output Images           : "
            f"{UNPACKED_IMAGES_DIR}"
        )

        print(
            f"Output Disc Masks       : "
            f"{UNPACKED_MASKS_DIR}"
        )

        print("=" * 70)

        print(
            Fore.GREEN
            + "T2 volume unpacking completed successfully."
        )

        print("=" * 70 + "\n")

        # ---------------------------------------------------------------------
        # Execution log
        # ---------------------------------------------------------------------

        logger.add(
            "Retained T2 Studies",
            len(t2_files),
        )

        logger.add(
            "Excluded Patients",
            sorted(EXCLUDED_PATIENTS),
        )

        logger.add(
            "Total MRI Slices",
            total_mri_slices,
        )

        logger.add(
            "Total Disc-Mask Slices",
            total_mask_slices,
        )

        logger.add(
            "Volume Metadata CSV",
            str(UNPACKED_METADATA_CSV),
        )

        logger.add(
            "Output Images",
            str(UNPACKED_IMAGES_DIR),
        )

        logger.add(
            "Output Disc Masks",
            str(UNPACKED_MASKS_DIR),
        )

        logger.save(
            "SUCCESS"
        )

    except Exception as exception:

        logger.error(
            str(exception)
        )

        logger.save(
            "FAILED"
        )

        print(
            Fore.RED
            + f"\nERROR: {exception}"
        )

        sys.exit(1)


if __name__ == "__main__":
    main()