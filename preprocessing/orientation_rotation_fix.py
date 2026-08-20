"""
IVD ROTATION FIX DIAGNOSTIC

Purpose
-------
Diagnose and visually verify the IVD orientation correction before
introducing any resolution experiments.

Pipeline
--------
1. Load original MRI .mha
2. Load original SPIDER segmentation-mask .mha
3. Automatically determine slice axis using the LOWEST volume dimension
4. Extract the medial sagittal slice
5. Map project IVD labels 1-6 to SPIDER mask labels 201-206
6. Calculate PCA orientation from each IVD mask
7. Normalize PCA orientation to [-90°, +90°)
8. Apply the smallest corrective rotation
9. Rotate MRI and mask using the same transformation
10. Apply 15% padding around the rotated IVD bounding box
11. Save diagnostic visualizations

IMPORTANT
---------
Slice-axis rule:
    The axis with the LOWEST dimension value is selected as the slice axis.

PCA rule:
    PCA gives an AXIS, not a directional vector.

Therefore:
    -143.29° and +36.71° represent the same anatomical axis.

We normalize the PCA angle to [-90°, +90°) before calculating
the corrective rotation.

No resolution resizing is performed here.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.ndimage import rotate


# ============================================================================
# CONFIGURATION
# ============================================================================

PATIENT_ID = "101"

MRI_PATH = Path(
    r"D:\BAAA project\spider-pfirrmann-mri-core\data\raw\images\101_t2.mha"
)

MASK_PATH = Path(
    r"D:\BAAA project\spider-pfirrmann-mri-core\data\raw\masks\101_t2.mha"
)

OUTPUT_ROOT = Path(
    r"D:\BAAA project\spider-pfirrmann-mri-core"
    r"\outputs\orientation_rotation_fix"
)

# Project IVD label -> SPIDER mask label
IVD_TO_MASK = {
    1: 201,
    2: 202,
    3: 203,
    4: 204,
    5: 205,
    6: 206,
}

# 15% padding on each side of the rotated IVD bounding box.
PADDING_PERCENT = 0.40


# ============================================================================
# LOAD MHA
# ============================================================================

def load_mha(path: Path) -> np.ndarray:
    """
    Load an MHA volume using SimpleITK.
    """

    image = sitk.ReadImage(str(path))

    array = sitk.GetArrayFromImage(image)

    return np.asarray(array)


# ============================================================================
# SLICE AXIS SELECTION
# ============================================================================

def determine_slice_axis(volume: np.ndarray) -> int:
    """
    Select the slice axis using the project's agreed rule:

        The axis with the LOWEST dimension value is the slice axis.

    Example
    -------
    volume.shape = (352, 384, 17)

        axis 0 -> 352
        axis 1 -> 384
        axis 2 -> 17

    Therefore:

        slice_axis = 2
    """

    slice_axis = int(
        np.argmin(volume.shape)
    )

    return slice_axis


# ============================================================================
# EXTRACT SLICE
# ============================================================================

def get_slice(
    volume: np.ndarray,
    axis: int,
    index: int,
) -> np.ndarray:
    """
    Extract one 2D slice from a 3D volume.
    """

    if axis == 0:
        return volume[index, :, :]

    elif axis == 1:
        return volume[:, index, :]

    elif axis == 2:
        return volume[:, :, index]

    else:
        raise ValueError(
            f"Invalid slice axis: {axis}"
        )


# ============================================================================
# PCA ANGLE NORMALIZATION
# ============================================================================

def normalize_pca_angle(
    angle: float,
) -> float:
    """
    Normalize a PCA AXIS angle into [-90°, +90°).

    PCA represents an axis.

    Therefore:

        theta
        theta + 180°
        theta - 180°

    all represent the same physical axis.

    Examples
    --------
    -143.29° -> +36.71°
    -166.10° -> +13.90°
    -178.40° ->  +1.60°
     171.74° ->  -8.26°
     165.64° -> -14.36°
     168.69° -> -11.31°
    """

    normalized = (
        (angle + 90.0) % 180.0
    ) - 90.0

    return normalized


def calculate_applied_rotation(
    pca_angle: float,
):
    """
    Calculate the corrective rotation required to make
    the IVD principal axis horizontal.

    IMPORTANT:
    scipy.ndimage.rotate uses the opposite visual
    rotation convention from the image-coordinate angle
    returned by our PCA calculation.

    Therefore the corrective rotation must use the
    NORMALIZED angle directly, not its negative.

    Example:

        Raw PCA:
            -143.29°

        Normalize:
            +36.71°

        Correct scipy rotation:
            +36.71°

    After rotation, the IVD principal axis should be
    approximately 0° (horizontal).
    """

    normalized_angle = normalize_pca_angle(
        pca_angle
    )

    # IMPORTANT:
    # Do NOT negate this angle.
    applied_rotation = normalized_angle

    return (
        normalized_angle,
        applied_rotation,
    )


# ============================================================================
# PCA CALCULATION
# ============================================================================

def calculate_pca_angle(
    mask: np.ndarray,
):
    """
    Calculate the principal axis of a binary IVD mask.

    Returns
    -------
    angle_deg
        Raw PCA angle in degrees.

    centroid
        (x, y) centroid.

    principal_vector
        Principal PCA direction vector.
    """

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) < 2:
        return (
            None,
            None,
            None,
        )

    points = np.column_stack(
        (xs, ys)
    ).astype(
        np.float64
    )

    centroid = points.mean(
        axis=0
    )

    centered = (
        points - centroid
    )

    covariance = np.cov(
        centered,
        rowvar=False,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    principal_vector = eigenvectors[
        :,
        np.argmax(eigenvalues),
    ]

    vx, vy = principal_vector

    angle_rad = np.arctan2(
        vy,
        vx,
    )

    angle_deg = np.degrees(
        angle_rad
    )

    return (
        angle_deg,
        centroid,
        principal_vector,
    )


# ============================================================================
# ROTATION
# ============================================================================

def rotate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    angle: float,
):
    """
    Rotate MRI and IVD mask using exactly the same angle.

    MRI:
        Linear interpolation.

    Mask:
        Nearest-neighbour interpolation.

    reshape=True keeps the complete rotated image.
    """

    rotated_image = rotate(
        image,
        angle=angle,
        reshape=True,
        order=1,
        mode="constant",
        cval=0,
        prefilter=True,
    )

    rotated_mask = rotate(
        mask.astype(np.uint8),
        angle=angle,
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    return (
        rotated_image,
        rotated_mask,
    )


# ============================================================================
# CROPPING
# ============================================================================

def extract_padded_crop(
    rotated_image: np.ndarray,
    rotated_mask: np.ndarray,
    padding_percent: float = 0.40,
):
    """
    Extract the rectangular bounding box around the rotated IVD mask.

    Padding is applied on all four sides.
    """

    ys, xs = np.where(
        rotated_mask > 0
    )

    if len(xs) == 0:
        return (
            None,
            None,
            None,
        )

    x_min = int(xs.min())
    x_max = int(xs.max())

    y_min = int(ys.min())
    y_max = int(ys.max())

    width = (
        x_max - x_min + 1
    )

    height = (
        y_max - y_min + 1
    )

    pad_x = int(
        np.ceil(
            width * padding_percent
        )
    )

    pad_y = int(
        np.ceil(
            height * padding_percent
        )
    )

    x0 = max(
        0,
        x_min - pad_x,
    )

    x1 = min(
        rotated_image.shape[1],
        x_max + pad_x + 1,
    )

    y0 = max(
        0,
        y_min - pad_y,
    )

    y1 = min(
        rotated_image.shape[0],
        y_max + pad_y + 1,
    )

    cropped_image = rotated_image[
        y0:y1,
        x0:x1,
    ]

    cropped_mask = rotated_mask[
        y0:y1,
        x0:x1,
    ]

    crop_info = {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "crop_x0": x0,
        "crop_x1": x1,
        "crop_y0": y0,
        "crop_y1": y1,
        "width": cropped_image.shape[1],
        "height": cropped_image.shape[0],
    }

    return (
        cropped_image,
        cropped_mask,
        crop_info,
    )


# ============================================================================
# DISPLAY NORMALIZATION
# ============================================================================

def normalize_for_display(
    image: np.ndarray,
) -> np.ndarray:
    """
    Normalize an image to [0, 1] for visualization only.
    """

    image = image.astype(
        np.float32
    )

    min_val = np.min(image)
    max_val = np.max(image)

    if max_val <= min_val:
        return np.zeros_like(
            image
        )

    return (
        image - min_val
    ) / (
        max_val - min_val
    )


# ============================================================================
# MASK OVERLAY
# ============================================================================

def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
):
    """
    Create MRI + IVD mask visualization.

    IVD mask is displayed in green.
    """

    image_norm = normalize_for_display(
        image
    )

    rgb = np.stack(
        [
            image_norm,
            image_norm,
            image_norm,
        ],
        axis=-1,
    )

    mask_bool = (
        mask > 0
    )

    alpha = 0.55

    rgb[mask_bool, 0] *= (
        1.0 - alpha
    )

    rgb[mask_bool, 1] = (
        rgb[mask_bool, 1]
        * (1.0 - alpha)
        + alpha
    )

    rgb[mask_bool, 2] *= (
        1.0 - alpha
    )

    return np.clip(
        rgb,
        0.0,
        1.0,
    )


# ============================================================================
# DIAGNOSTIC FIGURE
# ============================================================================

def save_diagnostic_figure(
    patient_id,
    slice_index,
    ivd_label,
    mask_label,
    original_image,
    original_mask,
    pca_angle,
    normalized_angle,
    applied_rotation,
    rotated_image,
    rotated_mask,
    cropped_image,
    cropped_mask,
    crop_info,
    output_path,
):
    """
    Save the six-panel rotation diagnostic.
    """

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 11),
    )

    # ----------------------------------------------------------------------
    # Original MRI
    # ----------------------------------------------------------------------

    axes[0, 0].imshow(
        normalize_for_display(
            original_image
        ),
        cmap="gray",
    )

    axes[0, 0].set_title(
        "Original MRI"
    )

    axes[0, 0].axis("off")

    # ----------------------------------------------------------------------
    # IVD Mask
    # ----------------------------------------------------------------------

    axes[0, 1].imshow(
        original_mask,
        cmap="gray",
    )

    axes[0, 1].set_title(
        f"IVD Mask | Label {mask_label}"
    )

    axes[0, 1].axis("off")

    # ----------------------------------------------------------------------
    # PCA Axis
    # ----------------------------------------------------------------------

    axes[0, 2].imshow(
        normalize_for_display(
            original_image
        ),
        cmap="gray",
    )

    ys, xs = np.where(
        original_mask > 0
    )

    if len(xs) > 0:

        points = np.column_stack(
            (xs, ys)
        ).astype(
            np.float64
        )

        centroid = points.mean(
            axis=0
        )

        centered = (
            points - centroid
        )

        covariance = np.cov(
            centered,
            rowvar=False,
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        principal_vector = eigenvectors[
            :,
            np.argmax(eigenvalues),
        ]

        cx, cy = centroid

        vx, vy = principal_vector

        length = (
            max(original_image.shape)
            * 0.20
        )

        x1 = (
            cx - vx * length
        )

        y1 = (
            cy - vy * length
        )

        x2 = (
            cx + vx * length
        )

        y2 = (
            cy + vy * length
        )

        axes[0, 2].plot(
            [x1, x2],
            [y1, y2],
            color="yellow",
            linewidth=2,
        )

        axes[0, 2].scatter(
            [cx],
            [cy],
            color="red",
            s=40,
        )

    axes[0, 2].set_title(
        f"PCA Axis | {pca_angle:.2f}°"
    )

    axes[0, 2].axis("off")

    # ----------------------------------------------------------------------
    # Rotated MRI + IVD Mask
    # ----------------------------------------------------------------------

    overlay = create_mask_overlay(
        rotated_image,
        rotated_mask,
    )

    axes[1, 0].imshow(
        overlay
    )

    axes[1, 0].set_title(
        f"Rotated MRI + IVD Mask\n"
        f"Applied Rotation = "
        f"{applied_rotation:.2f}°"
    )

    axes[1, 0].axis("off")

    # ----------------------------------------------------------------------
    # Cropped MRI
    # ----------------------------------------------------------------------

    axes[1, 1].imshow(
        normalize_for_display(
            cropped_image
        ),
        cmap="gray",
    )

    axes[1, 1].set_title(
        f"15% Padded Crop | "
        f"{crop_info['width']} × "
        f"{crop_info['height']}"
    )

    axes[1, 1].axis("off")

    # ----------------------------------------------------------------------
    # Cropped MRI + Mask
    # ----------------------------------------------------------------------

    crop_overlay = create_mask_overlay(
        cropped_image,
        cropped_mask,
    )

    axes[1, 2].imshow(
        crop_overlay
    )

    axes[1, 2].set_title(
        "Cropped IVD + Mask"
    )

    axes[1, 2].axis("off")

    # ----------------------------------------------------------------------
    # Main title
    # ----------------------------------------------------------------------

    fig.suptitle(
        f"Patient {patient_id} | "
        f"Slice {slice_index + 1} | "
        f"IVD Label {ivd_label} | "
        f"Mask Label {mask_label}\n"
        f"PCA Angle = {pca_angle:.2f}° | "
        f"Normalized = {normalized_angle:.2f}° | "
        f"Applied Rotation = "
        f"{applied_rotation:.2f}°",
        fontsize=16,
    )

    plt.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ]
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print("IVD ROTATION FIX DIAGNOSTIC")
    print("=" * 70)

    # ----------------------------------------------------------------------
    # Load MRI
    # ----------------------------------------------------------------------

    print(
        f"Patient             : {PATIENT_ID}"
    )

    print(
        f"MRI MHA             : {MRI_PATH}"
    )

    print(
        f"Mask MHA            : {MASK_PATH}"
    )

    mri_volume = load_mha(
        MRI_PATH
    )

    mask_volume = load_mha(
        MASK_PATH
    )

    print(
        f"Volume Size         : "
        f"{mri_volume.shape}"
    )

    # ----------------------------------------------------------------------
    # AUTOMATIC SLICE AXIS
    #
    # PROJECT RULE:
    # Choose the axis with the LOWEST dimension value.
    # ----------------------------------------------------------------------

    slice_axis = determine_slice_axis(
        mri_volume
    )

    print(
        f"Dimension Values     : "
        f"{mri_volume.shape}"
    )

    print(
        f"Lowest Dimension     : "
        f"{mri_volume.shape[slice_axis]}"
    )

    print(
        f"Slice Axis          : "
        f"{slice_axis}"
    )

    slice_count = (
        mri_volume.shape[
            slice_axis
        ]
    )

    print(
        f"Slice Count         : "
        f"{slice_count}"
    )

    # ----------------------------------------------------------------------
    # MEDIAL SLICE
    # ----------------------------------------------------------------------

    medial_slice_index = (
        slice_count // 2
    )

    print(
        f"Medial Slice Index  : "
        f"{medial_slice_index}"
    )

    print(
        f"Medial Slice        : "
        f"{medial_slice_index + 1}"
    )

    # ----------------------------------------------------------------------
    # Extract medial slice
    # ----------------------------------------------------------------------

    mri_slice = get_slice(
        mri_volume,
        slice_axis,
        medial_slice_index,
    )

    mask_slice = get_slice(
        mask_volume,
        slice_axis,
        medial_slice_index,
    )

    print(
        f"Native Slice Shape  : "
        f"{mri_slice.shape}"
    )

    # ----------------------------------------------------------------------
    # Mask information
    # ----------------------------------------------------------------------

    unique_values = np.unique(
        mask_slice
    )

    print(
        f"Mask Unique Values  : "
        f"{unique_values.tolist()}"
    )

    print(
        f"IVD Labels          : "
        f"{list(IVD_TO_MASK.keys())}"
    )

    # ----------------------------------------------------------------------
    # Output directory
    # ----------------------------------------------------------------------

    output_dir = (
        OUTPUT_ROOT
        / PATIENT_ID
        / f"slice_{medial_slice_index + 1:03d}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Output Directory    : "
        f"{output_dir}"
    )

    print()
    print("-" * 70)
    print("PROCESSING IVDS")
    print("-" * 70)

    processed = 0

    # ----------------------------------------------------------------------
    # PROCESS EACH IVD
    # ----------------------------------------------------------------------

    for ivd_label, mask_label in IVD_TO_MASK.items():

        # --------------------------------------------------------------
        # Extract respective SPIDER IVD mask
        # --------------------------------------------------------------

        ivd_mask = (
            mask_slice == mask_label
        )

        pixel_count = int(
            np.sum(ivd_mask)
        )

        # --------------------------------------------------------------
        # Skip missing IVD
        # --------------------------------------------------------------

        if pixel_count == 0:

            print(
                f"IVD {ivd_label} | "
                f"Mask {mask_label} | "
                f"SKIPPED | "
                f"Pixels: {pixel_count}"
            )

            continue

        # --------------------------------------------------------------
        # PCA
        # --------------------------------------------------------------

        (
            pca_angle,
            centroid,
            principal_vector,
        ) = calculate_pca_angle(
            ivd_mask
        )

        if pca_angle is None:

            print(
                f"IVD {ivd_label} | "
                f"Mask {mask_label} | "
                f"SKIPPED | "
                f"Invalid PCA"
            )

            continue

        # --------------------------------------------------------------
        # ROTATION FIX
        #
        # PCA gives an axis.
        # Normalize to [-90°, +90°).
        # Then apply the negative of that angle.
        # --------------------------------------------------------------

        (
            normalized_angle,
            applied_rotation,
        ) = calculate_applied_rotation(
            pca_angle
        )

        # --------------------------------------------------------------
        # Rotate MRI + mask
        # --------------------------------------------------------------

        (
            rotated_image,
            rotated_mask,
        ) = rotate_image_and_mask(
            mri_slice,
            ivd_mask,
            applied_rotation,
        )

        # --------------------------------------------------------------
        # Crop
        # --------------------------------------------------------------

        (
            cropped_image,
            cropped_mask,
            crop_info,
        ) = extract_padded_crop(
            rotated_image,
            rotated_mask,
            PADDING_PERCENT,
        )

        if cropped_image is None:

            print(
                f"IVD {ivd_label} | "
                f"Mask {mask_label} | "
                f"FAILED | "
                f"Empty rotated mask"
            )

            continue

        # --------------------------------------------------------------
        # Save diagnostic figure
        # --------------------------------------------------------------

        output_path = (
            output_dir
            / f"ivd_{ivd_label:02d}"
            f"_mask_{mask_label}"
            f"_rotation_diagnostic.png"
        )

        save_diagnostic_figure(
            patient_id=PATIENT_ID,
            slice_index=medial_slice_index,
            ivd_label=ivd_label,
            mask_label=mask_label,
            original_image=mri_slice,
            original_mask=ivd_mask,
            pca_angle=pca_angle,
            normalized_angle=normalized_angle,
            applied_rotation=applied_rotation,
            rotated_image=rotated_image,
            rotated_mask=rotated_mask,
            cropped_image=cropped_image,
            cropped_mask=cropped_mask,
            crop_info=crop_info,
            output_path=output_path,
        )

        # --------------------------------------------------------------
        # Console output
        # --------------------------------------------------------------

        print(
            f"IVD {ivd_label} | "
            f"Mask {mask_label} | "
            f"Pixels: {pixel_count:6d} | "
            f"PCA: {pca_angle:8.2f}° | "
            f"Normalized: "
            f"{normalized_angle:8.2f}° | "
            f"Rotation: "
            f"{applied_rotation:8.2f}° | "
            f"Crop: "
            f"{crop_info['width']} × "
            f"{crop_info['height']}"
        )

        processed += 1

    # ----------------------------------------------------------------------
    # COMPLETION
    # ----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("IVD ROTATION FIX DIAGNOSTIC COMPLETED")
    print("=" * 70)

    print(
        f"IVDs Requested     : "
        f"{len(IVD_TO_MASK)}"
    )

    print(
        f"IVDs Processed     : "
        f"{processed}"
    )

    print(
        f"Output Directory   : "
        f"{output_dir}"
    )

    print()
    print("IMPORTANT:")
    print(
        "1. Original .mha files were used directly."
    )
    print(
        "2. No unpacked PNG dataset was used."
    )
    print(
        "3. IVD 1-6 use SPIDER mask labels 201-206."
    )
    print(
        "4. Slice axis is automatically selected "
        "using the LOWEST volume dimension."
    )
    print(
        "5. PCA is calculated from each individual IVD mask."
    )
    print(
        "6. PCA axis is normalized to [-90°, +90°)."
    )
    print(
        "7. The smallest corrective rotation is applied."
    )
    print(
        "8. MRI uses linear interpolation."
    )
    print(
        "9. IVD mask uses nearest-neighbour interpolation."
    )
    print(
        "10. 15% padding is applied to the rotated "
        "IVD bounding box."
    )
    print(
        "11. No resolution resizing is performed."
    )


if __name__ == "__main__":
    main()