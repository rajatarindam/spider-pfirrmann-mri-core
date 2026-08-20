"""
DISC ROI RESOLUTION EXPERIMENT - DIRECT SQUARE STRETCHING

Experiment:
    1. Load original MHA MRI and mask.
    2. Select medial slice.
    3. Extract each IVD mask.
    4. Calculate PCA orientation independently for each IVD.
    5. Normalize PCA angle to [-90, +90).
    6. Apply the validated corrective rotation.
    7. Extract the same 30%-padded native rectangular ROI
       used in resolution_experiment.py.
    8. DO NOT preserve aspect ratio.
    9. Directly resize the rectangular ROI to square resolutions.
   10. Generate 1024x1024 representations.
   11. Save raw and upscaled representations.
   12. Generate comparison figures.

IMPORTANT:
    This is an EXPERIMENT ONLY.

    Unlike the current pipeline:
        rectangular ROI
            -> aspect-ratio preserving resize
            -> black square padding

    this experiment does:
        rectangular ROI
            -> direct resize to square

    Therefore anatomical proportions will intentionally be distorted.
"""

from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "raw" / "masks"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "resolution_experiment_stretched"
)

PATIENT_ID = "101"
MRI_NAME = "101_t2.mha"

# Same resolutions as the previous experiment.
RESOLUTIONS = [
    512,
    768,
    1024,
    1280,
    1536,
]

FINAL_UPSCALED_SIZE = 1024

# Keep the same ROI definition as the current experiment.
PADDING_PERCENT = 0.30

# IVD labels in the SPIDER mask.
IVD_TO_MASK = {
    1: 201,
    2: 202,
    3: 203,
    4: 204,
    5: 205,
    6: 206,
}


# ============================================================================
# LOAD MHA
# ============================================================================

def load_mha(path: Path) -> np.ndarray:
    """
    Load MHA file and return NumPy array.
    """
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)
    return array


# ============================================================================
# MEDIAL SLICE
# ============================================================================

def get_medial_slice(volume: np.ndarray):
    """
    Select the medial slice along the lowest-dimension axis.

    Expected volume:
        (H, W, S)

    Returns:
        slice_axis
        medial_index
        medial_slice
    """

    slice_axis = int(np.argmin(volume.shape))

    medial_index = volume.shape[slice_axis] // 2

    medial_slice = np.take(
        volume,
        medial_index,
        axis=slice_axis,
    )

    return slice_axis, medial_index, medial_slice


# ============================================================================
# PCA ORIENTATION
# ============================================================================

def calculate_pca_angle(binary_mask: np.ndarray):
    """
    Calculate the principal axis angle of the IVD mask.

    Returns angle in degrees in the range approximately [-180, 180).
    """

    ys, xs = np.where(binary_mask > 0)

    if len(xs) < 2:
        raise ValueError("Not enough pixels for PCA.")

    points = np.column_stack((xs, ys)).astype(np.float32)

    centroid = points.mean(axis=0)

    centered = points - centroid

    covariance = np.cov(centered.T)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]

    angle = np.degrees(
        np.arctan2(
            principal_vector[1],
            principal_vector[0],
        )
    )

    return float(angle), centroid


# ============================================================================
# NORMALIZE PCA ANGLE
# ============================================================================

def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-90, +90).
    """

    while angle >= 90:
        angle -= 180

    while angle < -90:
        angle += 180

    return angle


# ============================================================================
# ROTATION
# ============================================================================

def rotate_image(
    image: np.ndarray,
    angle: float,
    interpolation: int,
) -> np.ndarray:
    """
    Rotate image around its center while keeping the full image canvas.
    """

    height, width = image.shape[:2]

    center = (
        width / 2.0,
        height / 2.0,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0,
    )

    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return rotated


# ============================================================================
# PADDED ROI EXTRACTION
# ============================================================================

def extract_padded_crop(
    rotated_image: np.ndarray,
    rotated_mask: np.ndarray,
    padding_percent: float = 0.30,
):
    """
    Extract the same 30%-padded rectangular ROI used by the
    normal resolution experiment.

    IMPORTANT:
        This function DOES NOT make the crop square.

    The output retains its natural rectangular aspect ratio.
    """

    ys, xs = np.where(rotated_mask > 0)

    if len(xs) == 0:
        raise ValueError("IVD mask is empty after rotation.")

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    bbox_width = xmax - xmin + 1
    bbox_height = ymax - ymin + 1

    pad_x = int(round(bbox_width * padding_percent))
    pad_y = int(round(bbox_height * padding_percent))

    crop_xmin = max(0, xmin - pad_x)
    crop_xmax = min(
        rotated_image.shape[1],
        xmax + pad_x + 1,
    )

    crop_ymin = max(0, ymin - pad_y)
    crop_ymax = min(
        rotated_image.shape[0],
        ymax + pad_y + 1,
    )

    image_crop = rotated_image[
        crop_ymin:crop_ymax,
        crop_xmin:crop_xmax,
    ]

    mask_crop = rotated_mask[
        crop_ymin:crop_ymax,
        crop_xmin:crop_xmax,
    ]

    return image_crop, mask_crop


# ============================================================================
# DIRECT SQUARE RESIZE
# ============================================================================

def resize_direct_square(
    image: np.ndarray,
    size: int,
    interpolation: int,
) -> np.ndarray:
    """
    DIRECTLY resize rectangular image to size x size.

    Aspect ratio is intentionally NOT preserved.

    This is the key operation of Experiment 2.
    """

    return cv2.resize(
        image,
        (size, size),
        interpolation=interpolation,
    )


# ============================================================================
# NORMALIZATION FOR SAVING
# ============================================================================

def normalize_for_png(image: np.ndarray) -> np.ndarray:
    """
    Normalize MRI image to uint8 for PNG output.
    """

    image = image.astype(np.float32)

    minimum = image.min()
    maximum = image.max()

    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.uint8)

    normalized = (
        (image - minimum)
        / (maximum - minimum)
        * 255.0
    )

    return normalized.astype(np.uint8)


# ============================================================================
# SAVE IMAGE
# ============================================================================

def save_png(
    path: Path,
    image: np.ndarray,
):
    """
    Save image as PNG.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(path),
        image,
    )


# ============================================================================
# CREATE COMPARISON FIGURE
# ============================================================================

def create_comparison(
    representations,
    title,
    output_path,
):
    """
    Create comparison figure.

    representations:
        list of (name, image, mask)
    """

    count = len(representations)

    fig, axes = plt.subplots(
        2,
        count,
        figsize=(4 * count, 7),
    )

    if count == 1:
        axes = np.array(axes).reshape(2, 1)

    for i, (name, image, mask) in enumerate(representations):

        # --------------------------------------------------------------
        # MRI
        # --------------------------------------------------------------

        axes[0, i].imshow(
            image,
            cmap="gray",
        )

        axes[0, i].set_title(
            f"MRI\n{name}"
        )

        axes[0, i].axis("off")

        # --------------------------------------------------------------
        # MASK OVERLAY
        # --------------------------------------------------------------

        axes[1, i].imshow(
            image,
            cmap="gray",
        )

        overlay = np.ma.masked_where(
            mask == 0,
            mask,
        )

        axes[1, i].imshow(
            overlay,
            cmap="Greens",
            alpha=0.75,
        )

        axes[1, i].set_title(
            f"IVD Mask\n{name}"
        )

        axes[1, i].axis("off")

    plt.suptitle(
        title,
        fontsize=14,
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print("DISC ROI RESOLUTION EXPERIMENT - DIRECT SQUARE STRETCHING")
    print("=" * 78)

    image_path = IMAGE_DIR / MRI_NAME
    mask_path = MASK_DIR / MRI_NAME

    print(f"Patient             : {PATIENT_ID}")
    print(f"MRI MHA             : {image_path}")
    print(f"Disc Mask MHA       : {mask_path}")

    # ------------------------------------------------------------------------
    # LOAD VOLUMES
    # ------------------------------------------------------------------------

    image_volume = load_mha(image_path)
    mask_volume = load_mha(mask_path)

    print(f"Volume Size         : {image_volume.shape}")

    # ------------------------------------------------------------------------
    # MEDIAL SLICE
    # ------------------------------------------------------------------------

    slice_axis, medial_index, image_slice = get_medial_slice(
        image_volume
    )

    _, _, mask_slice = get_medial_slice(
        mask_volume
    )

    medial_slice_number = medial_index + 1

    print(f"Slice Axis          : {slice_axis}")
    print(f"Slice Count         : {image_volume.shape[slice_axis]}")
    print(f"Medial Slice Index  : {medial_index}")
    print(f"Medial Slice        : {medial_slice_number}")
    print(f"Native Slice Shape  : {image_slice.shape}")

    print(
        f"Mask Unique Values  : "
        f"{np.unique(mask_slice)}"
    )

    # ------------------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------------------

    slice_dir = (
        OUTPUT_ROOT
        / PATIENT_ID
        / f"slice_{medial_slice_number:03d}"
    )

    print(f"Output Directory    : {slice_dir}")

    # ------------------------------------------------------------------------
    # PROCESS IVDS
    # ------------------------------------------------------------------------

    print()
    print("-" * 78)
    print("PROCESSING IVDS")
    print("-" * 78)

    processed_count = 0

    for ivd_label, mask_label in IVD_TO_MASK.items():

        binary_mask = (
            mask_slice == mask_label
        ).astype(np.uint8)

        pixel_count = int(
            np.count_nonzero(binary_mask)
        )

        if pixel_count == 0:
            print(
                f"IVD {ivd_label} | "
                f"Mask {mask_label} | "
                f"SKIPPED: mask not found"
            )
            continue

        # --------------------------------------------------------------
        # PCA
        # --------------------------------------------------------------

        pca_angle, centroid = calculate_pca_angle(
            binary_mask
        )

        normalized_angle = normalize_angle(
            pca_angle
        )

        applied_rotation = normalized_angle

        # --------------------------------------------------------------
        # ROTATE MRI
        # --------------------------------------------------------------

        rotated_image = rotate_image(
            image_slice,
            applied_rotation,
            cv2.INTER_LINEAR,
        )

        # --------------------------------------------------------------
        # ROTATE MASK
        # --------------------------------------------------------------

        rotated_mask = rotate_image(
            binary_mask,
            applied_rotation,
            cv2.INTER_NEAREST,
        )

        rotated_mask = (
            rotated_mask > 0
        ).astype(np.uint8)

        # --------------------------------------------------------------
        # EXTRACT SAME 30%-PADDED RECTANGULAR ROI
        # --------------------------------------------------------------

        native_image, native_mask = extract_padded_crop(
            rotated_image,
            rotated_mask,
            PADDING_PERCENT,
        )

        native_height, native_width = native_image.shape[:2]

        print(
            f"IVD {ivd_label} | "
            f"Mask {mask_label} | "
            f"Pixels: {pixel_count:5d} | "
            f"PCA: {pca_angle:8.2f}° | "
            f"Normalized: {normalized_angle:8.2f}° | "
            f"Rotation: {applied_rotation:8.2f}° | "
            f"Native Crop: "
            f"{native_width} × {native_height}"
        )

        # --------------------------------------------------------------
        # OUTPUT DIRECTORIES
        # --------------------------------------------------------------

        label_dir = (
            slice_dir
            / f"label_{ivd_label}"
        )

        raw_dir = label_dir / "raw"
        upscaled_dir = label_dir / "upscaled"
        raw_comparison_dir = (
            label_dir / "raw_comparison"
        )
        upscaled_comparison_dir = (
            label_dir / "upscaled_comparison"
        )

        for directory in [
            raw_dir,
            upscaled_dir,
            raw_comparison_dir,
            upscaled_comparison_dir,
        ]:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        # --------------------------------------------------------------
        # REPRESENTATION STORAGE
        # --------------------------------------------------------------

        raw_representations = []
        upscaled_representations = []

        # --------------------------------------------------------------
        # NATIVE
        # --------------------------------------------------------------

        native_png = normalize_for_png(
            native_image
        )

        save_png(
            raw_dir / "native.png",
            native_png,
        )

        save_png(
            raw_dir / "native_mask.png",
            native_mask * 255,
        )

        raw_representations.append(
            (
                "native",
                native_png,
                native_mask,
            )
        )

        # --------------------------------------------------------------
        # RESOLUTION EXPERIMENT
        # --------------------------------------------------------------

        for resolution in RESOLUTIONS:

            # ----------------------------------------------------------
            # DIRECT SQUARE STRETCHING
            # ----------------------------------------------------------

            resized_image = resize_direct_square(
                native_image,
                resolution,
                cv2.INTER_LINEAR,
            )

            resized_mask = resize_direct_square(
                native_mask,
                resolution,
                cv2.INTER_NEAREST,
            )

            resized_mask = (
                resized_mask > 0
            ).astype(np.uint8)

            resized_png = normalize_for_png(
                resized_image
            )

            # ----------------------------------------------------------
            # SAVE RAW REPRESENTATION
            # ----------------------------------------------------------

            save_png(
                raw_dir / f"{resolution}.png",
                resized_png,
            )

            save_png(
                raw_dir / f"{resolution}_mask.png",
                resized_mask * 255,
            )

            raw_representations.append(
                (
                    str(resolution),
                    resized_png,
                    resized_mask,
                )
            )

            # ----------------------------------------------------------
            # UPSCALE EVERYTHING TO 1024 × 1024
            #
            # This lets us compare all representations at the same
            # final display/input size.
            # ----------------------------------------------------------

            if resolution == FINAL_UPSCALED_SIZE:

                final_image = resized_png
                final_mask = resized_mask

            else:

                final_image = cv2.resize(
                    resized_png,
                    (
                        FINAL_UPSCALED_SIZE,
                        FINAL_UPSCALED_SIZE,
                    ),
                    interpolation=cv2.INTER_CUBIC,
                )

                final_mask = cv2.resize(
                    resized_mask,
                    (
                        FINAL_UPSCALED_SIZE,
                        FINAL_UPSCALED_SIZE,
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

                final_mask = (
                    final_mask > 0
                ).astype(np.uint8)

            save_png(
                upscaled_dir / f"{resolution}.png",
                final_image,
            )

            save_png(
                upscaled_dir / f"{resolution}_mask.png",
                final_mask * 255,
            )

            upscaled_representations.append(
                (
                    str(resolution),
                    final_image,
                    final_mask,
                )
            )

        # --------------------------------------------------------------
        # NATIVE UPSCALED TO 1024
        # --------------------------------------------------------------

        native_upscaled = cv2.resize(
            native_png,
            (
                FINAL_UPSCALED_SIZE,
                FINAL_UPSCALED_SIZE,
            ),
            interpolation=cv2.INTER_CUBIC,
        )

        native_mask_upscaled = cv2.resize(
            native_mask,
            (
                FINAL_UPSCALED_SIZE,
                FINAL_UPSCALED_SIZE,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        native_mask_upscaled = (
            native_mask_upscaled > 0
        ).astype(np.uint8)

        save_png(
            upscaled_dir / "native.png",
            native_upscaled,
        )

        save_png(
            upscaled_dir / "native_mask.png",
            native_mask_upscaled * 255,
        )

        # Put native at the beginning of the upscaled comparison.
        upscaled_representations.insert(
            0,
            (
                "native",
                native_upscaled,
                native_mask_upscaled,
            )
        )

        # --------------------------------------------------------------
        # RAW COMPARISON
        # --------------------------------------------------------------

        raw_title = (
            f"Patient {PATIENT_ID} | "
            f"IVD Label {ivd_label} | "
            f"RAW DIRECT-SQUARE STRETCHING\n"
            f"Native Crop: "
            f"{native_width} × {native_height} | "
            f"PCA: {pca_angle:.2f}° | "
            f"Rotation: {applied_rotation:.2f}° | "
            f"Padding: {PADDING_PERCENT * 100:.0f}%"
        )

        create_comparison(
            raw_representations,
            raw_title,
            raw_comparison_dir / "comparison.png",
        )

        # --------------------------------------------------------------
        # UPSCALED COMPARISON
        # --------------------------------------------------------------

        upscaled_title = (
            f"Patient {PATIENT_ID} | "
            f"IVD Label {ivd_label} | "
            f"ALL REPRESENTATIONS AT "
            f"{FINAL_UPSCALED_SIZE} × "
            f"{FINAL_UPSCALED_SIZE}\n"
            f"PCA: {pca_angle:.2f}° | "
            f"Rotation: {applied_rotation:.2f}° | "
            f"Padding: {PADDING_PERCENT * 100:.0f}% | "
            f"DIRECT SQUARE STRETCH"
        )

        create_comparison(
            upscaled_representations,
            upscaled_title,
            upscaled_comparison_dir / "comparison.png",
        )

        processed_count += 1

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 78)
    print("DIRECT-SQUARE-STRETCHING EXPERIMENT COMPLETED")
    print("=" * 78)

    print(f"IVDs Requested     : {len(IVD_TO_MASK)}")
    print(f"IVDs Processed     : {processed_count}")
    print(f"Slice Axis         : {slice_axis}")
    print(f"Medial Slice       : {medial_slice_number}")
    print(f"Padding             : {PADDING_PERCENT * 100:.0f}%")

    print(
        "Raw Resolutions    : "
        "native + 512 + 768 + 1024 + 1280 + 1536"
    )

    print(
        "Upscaled Resolution: "
        "all representations -> 1024 × 1024"
    )

    print(f"Output Directory   : {slice_dir}")

    print()
    print("IMPORTANT:")
    print("1. Same MHA input files were used.")
    print("2. Same medial slice selection was used.")
    print("3. Same IVD labels 1-6 -> masks 201-206.")
    print("4. PCA is calculated independently for every IVD.")
    print("5. Same PCA normalization and rotation are used.")
    print("6. Same 30% padded rectangular ROI is extracted.")
    print("7. Aspect ratio is intentionally NOT preserved.")
    print("8. Rectangular ROI is directly resized to square.")
    print("9. No black square padding is added.")
    print("10. Anatomical proportions are intentionally distorted.")
    print("11. This experiment is for visual comparison only.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()