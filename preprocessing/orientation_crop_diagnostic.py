"""
Orientation and Crop Diagnostic for SPIDER IVD ROI Extraction

Purpose
-------
This diagnostic verifies the following pipeline on the ORIGINAL .mha files:

    Raw MRI MHA
        |
        +--> medial sagittal slice
        |
        +--> select IVD label
        |
        +--> map project IVD label -> SPIDER mask label
        |
        +--> extract exact IVD mask
        |
        +--> PCA orientation
        |
        +--> rotate MRI + mask
        |
        +--> bounding-box crop
        |
        +--> 15% padding on every side
        |
        +--> diagnostic visualization

IMPORTANT
---------
SPIDER mask label convention:

    0       = background
    1-25    = vertebrae
    100     = spinal canal
    101-125 = partially visible vertebrae
    201-225 = intervertebral discs

Therefore:

    Project IVD label 1 -> mask value 201
    Project IVD label 2 -> mask value 202
    ...
    Project IVD label N -> mask value 200 + N

This script intentionally DOES NOT:
    - create PNG datasets
    - perform the final resolution experiment
    - perform MRI-CORE preprocessing
    - perform model training
    - change intensity normalization
    - change the PCA method
    - change the 15% padding rule

It is only a diagnostic for IVD selection, orientation and cropping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
RAW_MASK_DIR = PROJECT_ROOT / "data" / "raw" / "masks"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "orientation_crop_diagnostic"
)

# Patient to inspect
PATIENT_ID = "101"

# We use the medial slice.
# This is 1-based because that is how we have been discussing slices.
# Patient 101 has 17 slices, therefore medial slice = 9.
MEDIAL_SLICE_NUMBER = 9

# Project-level IVD labels that we want to inspect.
# These correspond to the grading CSV's IVD labels.
IVD_LABELS = [1, 2, 3, 4, 5, 6]

# Padding applied independently to each side.
#
# Example:
#     width  = 100
#     horizontal padding = 15 px on left + 15 px on right
#
#     height = 80
#     vertical padding = 12 px on top + 12 px on bottom
#
PADDING_FRACTION = 0.15

# PCA line length for visualization.
PCA_LINE_LENGTH = 120

# Interpolation methods
MRI_INTERPOLATION = cv2.INTER_LINEAR
MASK_INTERPOLATION = cv2.INTER_NEAREST


# ============================================================================
# PATH HELPERS
# ============================================================================

def get_mri_path(patient_id: str) -> Path:
    """
    Return the original T2 MRI .mha path.
    """
    return RAW_IMAGE_DIR / f"{patient_id}_t2.mha"


def get_mask_path(patient_id: str) -> Path:
    """
    Return the original T2 segmentation mask .mha path.
    """
    return RAW_MASK_DIR / f"{patient_id}_t2.mha"


def get_output_dir(patient_id: str, slice_number: int) -> Path:
    """
    Create and return the output directory.
    """
    output_dir = (
        OUTPUT_ROOT
        / patient_id
        / f"slice_{slice_number:03d}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_dir


# ============================================================================
# SPIDER LABEL MAPPING
# ============================================================================

def ivd_project_label_to_mask_label(ivd_label: int) -> int:
    """
    Convert project/radiological IVD label to the original SPIDER
    segmentation-mask label.

    SPIDER convention:

        IVD 1 -> 201
        IVD 2 -> 202
        IVD 3 -> 203
        IVD 4 -> 204
        IVD 5 -> 205
        IVD 6 -> 206
        ...

    Parameters
    ----------
    ivd_label:
        Project-level IVD label.

    Returns
    -------
    int
        Actual pixel value used for that IVD in the segmentation mask.
    """

    if ivd_label < 1:
        raise ValueError(
            f"Invalid IVD label: {ivd_label}. "
            "IVD labels must be >= 1."
        )

    return 200 + ivd_label


# ============================================================================
# MHA LOADING
# ============================================================================

def load_mha(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    """
    Load an MHA file using SimpleITK.

    SimpleITK returns the array in:

        [z, y, x]

    ordering for a 3D image.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    return array, image


# ============================================================================
# SLICE AXIS DETECTION
# ============================================================================

def determine_slice_axis(
    size_xyz: Tuple[int, int, int]
) -> Tuple[int, str]:
    """
    Determine the slice axis using the smallest dimension.

    The project EDA established that the sagittal slice/depth dimension
    is not guaranteed to be X. Therefore we do NOT assume axis X.

    SimpleITK size:
        (X, Y, Z)

    We choose the smallest dimension as the slice/depth axis.

    Returns
    -------
    axis_index:
        0 = X
        1 = Y
        2 = Z

    axis_name:
        Human-readable axis name.
    """

    axis_index = int(np.argmin(size_xyz))

    axis_names = {
        0: "X",
        1: "Y",
        2: "Z",
    }

    return axis_index, axis_names[axis_index]


def extract_slice(
    volume: np.ndarray,
    sitk_image: sitk.Image,
    slice_number: int,
) -> Tuple[np.ndarray, int, str, int]:
    """
    Extract one sagittal slice.

    Parameters
    ----------
    volume:
        NumPy array returned by SimpleITK.
        Shape is [Z, Y, X].

    sitk_image:
        Original SimpleITK image.

    slice_number:
        1-based slice number.

    Returns
    -------
    slice_2d:
        2D slice.

    axis_index:
        Slice axis in SimpleITK XYZ coordinates.

    axis_name:
        X/Y/Z.

    zero_based_index:
        Actual zero-based slice index.
    """

    size_xyz = sitk_image.GetSize()

    axis_index, axis_name = determine_slice_axis(
        size_xyz
    )

    # SimpleITK NumPy array is [Z, Y, X].
    #
    # Convert the XYZ axis to the corresponding NumPy axis:
    #
    # SITK X -> NumPy axis 2
    # SITK Y -> NumPy axis 1
    # SITK Z -> NumPy axis 0

    sitk_to_numpy_axis = {
        0: 2,
        1: 1,
        2: 0,
    }

    numpy_axis = sitk_to_numpy_axis[axis_index]

    slice_count = volume.shape[numpy_axis]

    zero_based_index = slice_number - 1

    if not 0 <= zero_based_index < slice_count:
        raise IndexError(
            f"Slice {slice_number} is outside valid range "
            f"1-{slice_count}."
        )

    slice_2d = np.take(
        volume,
        zero_based_index,
        axis=numpy_axis,
    )

    return (
        np.asarray(slice_2d),
        axis_index,
        axis_name,
        zero_based_index,
    )


# ============================================================================
# IMAGE NORMALIZATION FOR VISUALIZATION ONLY
# ============================================================================

def normalize_for_display(
    image: np.ndarray
) -> np.ndarray:
    """
    Normalize an MRI slice to [0, 1] ONLY for matplotlib visualization.

    This does NOT represent the final training preprocessing.

    It does not modify the source MRI data.
    """

    image = image.astype(np.float32)

    finite_values = image[np.isfinite(image)]

    if finite_values.size == 0:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    min_value = finite_values.min()
    max_value = finite_values.max()

    if max_value <= min_value:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    normalized = (
        (image - min_value)
        / (max_value - min_value)
    )

    return np.clip(
        normalized,
        0.0,
        1.0,
    )


# ============================================================================
# IVD MASK EXTRACTION
# ============================================================================

def extract_ivd_mask(
    mask_slice: np.ndarray,
    ivd_label: int,
) -> Tuple[np.ndarray, int]:
    """
    Extract the exact IVD mask using the SPIDER mask-label convention.

    Example:

        ivd_label = 1
        mask_label = 201

        ivd_mask = mask_slice == 201
    """

    mask_label = ivd_project_label_to_mask_label(
        ivd_label
    )

    binary_mask = (
        mask_slice == mask_label
    ).astype(np.uint8)

    return binary_mask, mask_label


# ============================================================================
# PCA ORIENTATION
# ============================================================================

def calculate_pca_orientation(
    binary_mask: np.ndarray,
) -> Optional[Dict[str, object]]:
    """
    Calculate PCA orientation using ONLY the selected IVD mask pixels.

    Returns None if there are insufficient pixels.

    Returned dictionary contains:

        centroid
        angle_deg
        direction
        pixel_count
    """

    ys, xs = np.where(binary_mask > 0)

    if len(xs) < 2:
        return None

    points = np.column_stack(
        (
            xs.astype(np.float32),
            ys.astype(np.float32),
        )
    )

    centroid = points.mean(axis=0)

    centered = points - centroid

    covariance = np.cov(
        centered,
        rowvar=False,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    principal_index = int(
        np.argmax(eigenvalues)
    )

    direction = eigenvectors[
        :,
        principal_index,
    ]

    angle_deg = float(
        np.degrees(
            np.arctan2(
                direction[1],
                direction[0],
            )
        )
    )

    return {
        "centroid": centroid,
        "angle_deg": angle_deg,
        "direction": direction,
        "pixel_count": len(xs),
        "eigenvalues": eigenvalues,
    }


# ============================================================================
# ROTATION
# ============================================================================

def rotate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    angle_deg: float,
    centroid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate MRI and mask around the IVD centroid.

    MRI:
        INTER_LINEAR

    Mask:
        INTER_NEAREST

    This keeps the segmentation labels discrete.
    """

    height, width = image.shape[:2]

    center = (
        float(centroid[0]),
        float(centroid[1]),
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center=center,
        angle=angle_deg,
        scale=1.0,
    )

    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=MRI_INTERPOLATION,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    rotated_mask = cv2.warpAffine(
        mask,
        rotation_matrix,
        (width, height),
        flags=MASK_INTERPOLATION,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return (
        rotated_image,
        rotated_mask,
    )


# ============================================================================
# BOUNDING BOX
# ============================================================================

def get_mask_bounding_box(
    binary_mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Return bounding box of the selected IVD mask.

    Returns:

        x_min
        y_min
        x_max
        y_max

    """

    ys, xs = np.where(binary_mask > 0)

    if len(xs) == 0:
        return None

    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())

    return (
        x_min,
        y_min,
        x_max,
        y_max,
    )


# ============================================================================
# PADDED CROP
# ============================================================================

def crop_with_padding(
    image: np.ndarray,
    mask: np.ndarray,
    padding_fraction: float = 0.15,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Tuple[int, int, int, int]],
]:
    """
    Crop around the selected IVD mask.

    Padding is applied independently to all four sides.

    If the IVD bounding box is:

        width  = W
        height = H

    then:

        horizontal padding = 0.15 * W
        vertical padding   = 0.15 * H

    applied to EACH side.
    """

    bbox = get_mask_bounding_box(mask)

    if bbox is None:
        return None, None, None

    x_min, y_min, x_max, y_max = bbox

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    pad_x = int(
        round(
            bbox_width
            * padding_fraction
        )
    )

    pad_y = int(
        round(
            bbox_height
            * padding_fraction
        )
    )

    image_height, image_width = image.shape[:2]

    crop_x_min = max(
        0,
        x_min - pad_x,
    )

    crop_y_min = max(
        0,
        y_min - pad_y,
    )

    crop_x_max = min(
        image_width - 1,
        x_max + pad_x,
    )

    crop_y_max = min(
        image_height - 1,
        y_max + pad_y,
    )

    cropped_image = image[
        crop_y_min : crop_y_max + 1,
        crop_x_min : crop_x_max + 1,
    ]

    cropped_mask = mask[
        crop_y_min : crop_y_max + 1,
        crop_x_min : crop_x_max + 1,
    ]

    crop_box = (
        crop_x_min,
        crop_y_min,
        crop_x_max,
        crop_y_max,
    )

    return (
        cropped_image,
        cropped_mask,
        crop_box,
    )


# ============================================================================
# DRAW PCA AXIS
# ============================================================================

def create_pca_overlay(
    image: np.ndarray,
    pca_result: Dict[str, object],
    line_length: int = 120,
) -> np.ndarray:
    """
    Create an RGB diagnostic image showing the PCA axis.
    """

    display = normalize_for_display(image)

    display_rgb = np.stack(
        [
            display,
            display,
            display,
        ],
        axis=-1,
    )

    display_rgb = (
        display_rgb * 255
    ).astype(np.uint8)

    centroid = np.asarray(
        pca_result["centroid"],
        dtype=np.float32,
    )

    direction = np.asarray(
        pca_result["direction"],
        dtype=np.float32,
    )

    cx = int(round(centroid[0]))
    cy = int(round(centroid[1]))

    dx = float(direction[0])
    dy = float(direction[1])

    x1 = int(
        round(
            cx - dx * line_length
        )
    )

    y1 = int(
        round(
            cy - dy * line_length
        )
    )

    x2 = int(
        round(
            cx + dx * line_length
        )
    )

    y2 = int(
        round(
            cy + dy * line_length
        )
    )

    # PCA axis
    cv2.line(
        display_rgb,
        (x1, y1),
        (x2, y2),
        (255, 255, 0),
        2,
    )

    # Centroid
    cv2.circle(
        display_rgb,
        (cx, cy),
        5,
        (255, 0, 0),
        -1,
    )

    return display_rgb


# ============================================================================
# MASK OVERLAY
# ============================================================================

def create_mask_overlay(
    image: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Create an MRI + selected IVD mask overlay.

    The selected IVD is displayed in green.
    """

    display = normalize_for_display(image)

    rgb = np.stack(
        [
            display,
            display,
            display,
        ],
        axis=-1,
    )

    mask_bool = binary_mask > 0

    # Green overlay
    rgb[mask_bool, 0] = (
        rgb[mask_bool, 0] * 0.25
    )

    rgb[mask_bool, 1] = (
        rgb[mask_bool, 1] * 0.25
        + 0.75
    )

    rgb[mask_bool, 2] = (
        rgb[mask_bool, 2] * 0.25
    )

    return np.clip(
        rgb,
        0.0,
        1.0,
    )


# ============================================================================
# SAVE IMAGE
# ============================================================================

def save_rgb_image(
    image: np.ndarray,
    path: Path,
) -> None:
    """
    Save RGB image using matplotlib.
    """

    plt.imsave(
        str(path),
        np.clip(
            image,
            0.0,
            1.0,
        ),
    )


def save_gray_image(
    image: np.ndarray,
    path: Path,
) -> None:
    """
    Save grayscale image for diagnostic purposes.
    """

    display = normalize_for_display(image)

    plt.imsave(
        str(path),
        display,
        cmap="gray",
    )


# ============================================================================
# COMPLETE DIAGNOSTIC PANEL
# ============================================================================

def create_diagnostic_panel(
    patient_id: str,
    slice_number: int,
    ivd_label: int,
    mask_label: int,
    original_image: np.ndarray,
    original_mask: np.ndarray,
    pca_overlay: np.ndarray,
    rotated_image: np.ndarray,
    rotated_mask: np.ndarray,
    cropped_image: np.ndarray,
    cropped_mask: np.ndarray,
    angle_deg: float,
    output_path: Path,
) -> None:
    """
    Create the six-panel diagnostic visualization.
    """

    original_display = normalize_for_display(
        original_image
    )

    mask_display = (
        original_mask > 0
    ).astype(np.float32)

    rotated_display = normalize_for_display(
        rotated_image
    )

    cropped_display = normalize_for_display(
        cropped_image
    )

    cropped_overlay = create_mask_overlay(
        cropped_image,
        cropped_mask,
    )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
    )

    fig.suptitle(
        (
            f"Patient {patient_id} | "
            f"Slice {slice_number} | "
            f"IVD Label {ivd_label} | "
            f"Mask Label {mask_label}"
        ),
        fontsize=16,
    )

    # ----------------------------------------------------------------------
    # Original MRI
    # ----------------------------------------------------------------------

    axes[0, 0].imshow(
        original_display,
        cmap="gray",
    )

    axes[0, 0].set_title(
        "Original MRI"
    )

    axes[0, 0].axis("off")

    # ----------------------------------------------------------------------
    # Exact IVD mask
    # ----------------------------------------------------------------------

    axes[0, 1].imshow(
        mask_display,
        cmap="gray",
    )

    axes[0, 1].set_title(
        f"IVD Mask | Label {mask_label}"
    )

    axes[0, 1].axis("off")

    # ----------------------------------------------------------------------
    # PCA overlay
    # ----------------------------------------------------------------------

    axes[0, 2].imshow(
        pca_overlay
    )

    axes[0, 2].set_title(
        f"PCA Axis | Angle {angle_deg:.2f}°"
    )

    axes[0, 2].axis("off")

    # ----------------------------------------------------------------------
    # Rotated MRI + mask
    # ----------------------------------------------------------------------

    rotated_overlay = create_mask_overlay(
        rotated_image,
        rotated_mask,
    )

    axes[1, 0].imshow(
        rotated_overlay
    )

    axes[1, 0].set_title(
        "Rotated MRI + IVD Mask"
    )

    axes[1, 0].axis("off")

    # ----------------------------------------------------------------------
    # Cropped MRI
    # ----------------------------------------------------------------------

    axes[1, 1].imshow(
        cropped_display,
        cmap="gray",
    )

    axes[1, 1].set_title(
        (
            f"Rectangular Crop | "
            f"{cropped_image.shape[1]} × "
            f"{cropped_image.shape[0]}"
        )
    )

    axes[1, 1].axis("off")

    # ----------------------------------------------------------------------
    # Final crop + mask
    # ----------------------------------------------------------------------

    axes[1, 2].imshow(
        cropped_overlay
    )

    axes[1, 2].set_title(
        "Final Crop + IVD Mask"
    )

    axes[1, 2].axis("off")

    plt.tight_layout()

    fig.savefig(
        str(output_path),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# FULL-PATIENT MEDIAL-SLICE VISUALIZATION
# ============================================================================

def create_full_mask_visualization(
    patient_id: str,
    slice_number: int,
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    output_path: Path,
) -> None:
    """
    Create a basic full medial-slice visualization.

    This is useful for verifying that the selected medial slice
    and the original segmentation mask are correct before
    inspecting individual IVDs.
    """

    image_display = normalize_for_display(
        image_slice
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
    )

    fig.suptitle(
        (
            f"Patient {patient_id} | "
            f"Slice Axis: medial | "
            f"Medial Slice Index: {slice_number}"
        ),
        fontsize=16,
    )

    axes[0].imshow(
        image_display,
        cmap="gray",
    )

    axes[0].set_title(
        "Sagittal T2 MRI"
    )

    axes[0].axis("off")

    axes[1].imshow(
        mask_slice,
        cmap="gray",
    )

    axes[1].set_title(
        "Segmentation Mask"
    )

    axes[1].axis("off")

    axes[2].imshow(
        image_display,
        cmap="gray",
    )

    axes[2].imshow(
        mask_slice > 0,
        alpha=0.30,
        cmap="autumn",
    )

    axes[2].set_title(
        "MRI + Mask Overlay"
    )

    axes[2].axis("off")

    plt.tight_layout()

    fig.savefig(
        str(output_path),
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 70)
    print("ORIENTATION AND CROP DIAGNOSTIC")
    print("=" * 70)

    # ----------------------------------------------------------------------
    # Paths
    # ----------------------------------------------------------------------

    mri_path = get_mri_path(
        PATIENT_ID
    )

    mask_path = get_mask_path(
        PATIENT_ID
    )

    output_dir = get_output_dir(
        PATIENT_ID,
        MEDIAL_SLICE_NUMBER,
    )

    print(
        f"Patient             : {PATIENT_ID}"
    )

    print(
        f"MRI MHA             : {mri_path}"
    )

    print(
        f"Disc Mask MHA       : {mask_path}"
    )

    # ----------------------------------------------------------------------
    # Load original MHA files
    # ----------------------------------------------------------------------

    mri_volume, mri_sitk = load_mha(
        mri_path
    )

    mask_volume, mask_sitk = load_mha(
        mask_path
    )

    # ----------------------------------------------------------------------
    # Basic shape verification
    # ----------------------------------------------------------------------

    if mri_volume.shape != mask_volume.shape:
        raise ValueError(
            "MRI and mask volume dimensions do not match.\n"
            f"MRI shape  : {mri_volume.shape}\n"
            f"Mask shape : {mask_volume.shape}"
        )

    volume_size = mri_sitk.GetSize()

    # ----------------------------------------------------------------------
    # Determine slice axis
    # ----------------------------------------------------------------------

    axis_index, axis_name = determine_slice_axis(
        volume_size
    )

    # Number of slices along selected axis
    numpy_axis = {
        0: 2,
        1: 1,
        2: 0,
    }[axis_index]

    slice_count = mri_volume.shape[
        numpy_axis
    ]

    # ----------------------------------------------------------------------
    # Extract medial slice
    # ----------------------------------------------------------------------

    if MEDIAL_SLICE_NUMBER < 1:
        raise ValueError(
            "MEDIAL_SLICE_NUMBER must be >= 1."
        )

    if MEDIAL_SLICE_NUMBER > slice_count:
        raise ValueError(
            f"Requested medial slice "
            f"{MEDIAL_SLICE_NUMBER}, but only "
            f"{slice_count} slices exist."
        )

    image_slice, _, _, zero_based_index = (
        extract_slice(
            mri_volume,
            mri_sitk,
            MEDIAL_SLICE_NUMBER,
        )
    )

    mask_slice, _, _, _ = extract_slice(
        mask_volume,
        mask_sitk,
        MEDIAL_SLICE_NUMBER,
    )

    # ----------------------------------------------------------------------
    # Print volume information
    # ----------------------------------------------------------------------

    print(
        f"Volume Size         : {volume_size}"
    )

    print(
        f"Slice Axis          : {axis_name}"
    )

    print(
        f"Slice Axis Index    : {axis_index}"
    )

    print(
        f"Slice Count         : {slice_count}"
    )

    print(
        f"Medial Slice        : "
        f"{MEDIAL_SLICE_NUMBER}"
    )

    print(
        f"Native Slice Shape  : "
        f"{image_slice.shape}"
    )

    # ----------------------------------------------------------------------
    # Inspect mask values
    # ----------------------------------------------------------------------

    unique_values = np.unique(
        mask_slice
    )

    print(
        f"Mask Unique Values  : "
        f"{unique_values.tolist()}"
    )

    # ----------------------------------------------------------------------
    # Verify which requested IVD labels exist
    # ----------------------------------------------------------------------

    available_ivds: List[int] = []

    for ivd_label in IVD_LABELS:

        mask_label = (
            ivd_project_label_to_mask_label(
                ivd_label
            )
        )

        if np.any(
            mask_slice == mask_label
        ):
            available_ivds.append(
                ivd_label
            )

    print(
        f"IVD Labels Found    : "
        f"{available_ivds}"
    )

    # ----------------------------------------------------------------------
    # Save full medial slice visualization
    # ----------------------------------------------------------------------

    full_visualization_path = (
        output_dir
        / "medial_slice_mask_visualization.png"
    )

    create_full_mask_visualization(
        patient_id=PATIENT_ID,
        slice_number=MEDIAL_SLICE_NUMBER,
        image_slice=image_slice,
        mask_slice=mask_slice,
        output_path=full_visualization_path,
    )

    print(
        f"Full Visualization  : "
        f"{full_visualization_path}"
    )

    print()
    print("-" * 70)
    print("PROCESSING IVDs")
    print("-" * 70)

    # ----------------------------------------------------------------------
    # Process each IVD
    # ----------------------------------------------------------------------

    processed_count = 0

    for ivd_label in IVD_LABELS:

        # --------------------------------------------------------------
        # Convert project label -> actual SPIDER mask value
        # --------------------------------------------------------------

        mask_label = (
            ivd_project_label_to_mask_label(
                ivd_label
            )
        )

        # --------------------------------------------------------------
        # Extract exact IVD mask
        # --------------------------------------------------------------

        binary_mask, actual_mask_label = (
            extract_ivd_mask(
                mask_slice,
                ivd_label,
            )
        )

        pixel_count = int(
            np.count_nonzero(
                binary_mask
            )
        )

        if pixel_count == 0:

            print(
                f"IVD {ivd_label} | "
                f"Mask {actual_mask_label} | "
                f"NOT FOUND"
            )

            continue

        # --------------------------------------------------------------
        # PCA
        # --------------------------------------------------------------

        pca_result = calculate_pca_orientation(
            binary_mask
        )

        if pca_result is None:

            print(
                f"IVD {ivd_label} | "
                f"Mask {actual_mask_label} | "
                f"PCA FAILED"
            )

            continue

        angle_deg = float(
            pca_result["angle_deg"]
        )

        centroid = np.asarray(
            pca_result["centroid"]
        )

        # --------------------------------------------------------------
        # Rotate
        # --------------------------------------------------------------

        rotated_image, rotated_mask = (
            rotate_image_and_mask(
                image=image_slice,
                mask=binary_mask,
                angle_deg=-angle_deg,
                centroid=centroid,
            )
        )

        # --------------------------------------------------------------
        # Crop with 15% padding
        # --------------------------------------------------------------

        (
            cropped_image,
            cropped_mask,
            crop_box,
        ) = crop_with_padding(
            image=rotated_image,
            mask=rotated_mask,
            padding_fraction=PADDING_FRACTION,
        )

        if (
            cropped_image is None
            or cropped_mask is None
            or crop_box is None
        ):

            print(
                f"IVD {ivd_label} | "
                f"Mask {actual_mask_label} | "
                f"CROP FAILED"
            )

            continue

        # --------------------------------------------------------------
        # Create PCA overlay
        # --------------------------------------------------------------

        pca_overlay = create_pca_overlay(
            image=image_slice,
            pca_result=pca_result,
            line_length=PCA_LINE_LENGTH,
        )

        # --------------------------------------------------------------
        # Create output paths
        # --------------------------------------------------------------

        panel_path = (
            output_dir
            / (
                f"ivd_{ivd_label:02d}"
                f"_mask_{actual_mask_label}"
                f"_diagnostic.png"
            )
        )

        mask_path_out = (
            output_dir
            / (
                f"ivd_{ivd_label:02d}"
                f"_mask_{actual_mask_label}"
                f"_binary_mask.png"
            )
        )

        crop_path = (
            output_dir
            / (
                f"ivd_{ivd_label:02d}"
                f"_mask_{actual_mask_label}"
                f"_crop.png"
            )
        )

        # --------------------------------------------------------------
        # Save binary mask
        # --------------------------------------------------------------

        save_gray_image(
            binary_mask,
            mask_path_out,
        )

        # --------------------------------------------------------------
        # Save cropped MRI diagnostic image
        # --------------------------------------------------------------

        save_gray_image(
            cropped_image,
            crop_path,
        )

        # --------------------------------------------------------------
        # Save six-panel diagnostic
        # --------------------------------------------------------------

        create_diagnostic_panel(
            patient_id=PATIENT_ID,
            slice_number=MEDIAL_SLICE_NUMBER,
            ivd_label=ivd_label,
            mask_label=actual_mask_label,
            original_image=image_slice,
            original_mask=binary_mask,
            pca_overlay=pca_overlay,
            rotated_image=rotated_image,
            rotated_mask=rotated_mask,
            cropped_image=cropped_image,
            cropped_mask=cropped_mask,
            angle_deg=angle_deg,
            output_path=panel_path,
        )

        # --------------------------------------------------------------
        # Console output
        # --------------------------------------------------------------

        crop_height, crop_width = (
            cropped_image.shape
        )

        print(
            f"IVD {ivd_label} "
            f"| Mask {actual_mask_label} "
            f"| Pixels: {pixel_count:5d} "
            f"| Angle: {angle_deg:8.2f}° "
            f"| Crop: "
            f"{crop_width} × {crop_height}"
        )

        processed_count += 1

    # ----------------------------------------------------------------------
    # Completion
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ORIENTATION AND CROP DIAGNOSTIC COMPLETED"
    )
    print("=" * 70)

    print(
        f"IVDs Requested     : "
        f"{len(IVD_LABELS)}"
    )

    print(
        f"IVDs Processed     : "
        f"{processed_count}"
    )

    print(
        f"Output Directory   : "
        f"{output_dir}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This run uses the ORIGINAL .mha files."
    )

    print(
        "No unpacked PNG dataset is used."
    )

    print(
        "The only anatomical-label change is:"
    )

    print(
        "Project IVD 1-6 -> SPIDER mask 201-206."
    )

    print(
        "PCA, 15% padding and interpolation "
        "have otherwise been left unchanged."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()