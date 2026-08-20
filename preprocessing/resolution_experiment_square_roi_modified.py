"""
SPIDER Pfirrmann MRI-CORE
DISC ROI RESOLUTION EXPERIMENT

Purpose
-------
Compare different preprocessing resolutions for the same square IVD ROI.

The experiment uses the ORIGINAL .mha files directly.

Pipeline
--------
RAW MHA
   |
   +--> Select lowest-dimension axis
   |
   +--> Select medial slice
   |
   +--> Extract IVD mask
   |
   +--> PCA orientation
   |
   +--> Corrective rotation
   |
   +--> 30% padding
   |
   +--> Square native IVD crop
        (expand shorter dimension using ORIGINAL MRI pixels)
            |
            +--> 512 x 512
            +--> 768 x 768
            +--> 1024 x 1024
            +--> 1280 x 1280
            +--> 1536 x 1536

All RAW representations are already square:
    preserve anatomy
    + no black padding
    + no anisotropic stretching

UPSCALED
--------
Each RAW representation is converted to 1024 x 1024
with a square-to-square resize.

Output structure
----------------

outputs/
└── resolution_experiment/
    └── 101/
        └── slice_009/
            ├── label_1/
            │   ├── raw/
            │   │   ├── native.png
            │   │   ├── 512.png
            │   │   ├── 768.png
            │   │   ├── 1024.png
            │   │   ├── 1280.png
            │   │   └── 1536.png
            │   │
            │   ├── upscaled/
            │   │   ├── native.png
            │   │   ├── 512.png
            │   │   ├── 768.png
            │   │   ├── 1024.png
            │   │   ├── 1280.png
            │   │   └── 1536.png
            │   │
            │   ├── raw_comparison/
            │   │   └── comparison.png
            │   │
            │   └── upscaled_comparison/
            │       └── comparison.png
            │
            └── ...
"""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import rotate


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "images"
)

RAW_MASKS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "masks"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "resolution_experiment_square_roi"
)


# ============================================================================
# TEST PATIENT
# ============================================================================

PATIENT_ID = "101"


MRI_FILE = (
    RAW_IMAGES_DIR
    / f"{PATIENT_ID}_t2.mha"
)

MASK_FILE = (
    RAW_MASKS_DIR
    / f"{PATIENT_ID}_t2.mha"
)


# ============================================================================
# IVD LABEL MAPPING
# ============================================================================

# Project IVD label -> SPIDER mask label
IVD_TO_MASK = {
    1: 201,
    2: 202,
    3: 203,
    4: 204,
    5: 205,
    6: 206,
}


# ============================================================================
# RESOLUTION EXPERIMENT
# ============================================================================

TARGET_RESOLUTIONS = [
    512,
    768,
    1024,
    1280,
    1536,
]


# ============================================================================
# PADDING
# ============================================================================

# 30% padding on EACH side of the rotated IVD bounding box.
PADDING_PERCENT = 0.01

# Experiment 3: square native ROI using existing MRI pixels.
# No black padding is used to create the square ROI.
SQUARE_NATIVE_ROI = True


# ============================================================================
# INTERPOLATION
# ============================================================================

# MRI:
# continuous image data -> linear interpolation for geometric rotation.
MRI_ROTATION_ORDER = 1

# Mask:
# categorical labels -> nearest-neighbour interpolation.
MASK_ROTATION_ORDER = 0


# ============================================================================
# MHA LOADING
# ============================================================================

def load_mha(
    path: Path,
) -> np.ndarray:
    """
    Load an MHA volume using SimpleITK.
    """

    image = sitk.ReadImage(
        str(path)
    )

    array = sitk.GetArrayFromImage(
        image
    )

    return np.asarray(array)


# ============================================================================
# SLICE AXIS
# ============================================================================

def select_slice_axis(
    volume: np.ndarray,
) -> int:
    """
    Select the slice axis using the agreed project rule:

        LOWEST dimension value = slice axis

    Example:

        (352, 384, 17)

        lowest = 17

        therefore:

        slice_axis = 2
    """

    return int(
        np.argmin(
            volume.shape
        )
    )


# ============================================================================
# SLICE EXTRACTION
# ============================================================================

def extract_slice(
    volume: np.ndarray,
    slice_axis: int,
    slice_index: int,
) -> np.ndarray:
    """
    Extract one 2D slice from a 3D volume.
    """

    return np.take(
        volume,
        slice_index,
        axis=slice_axis,
    )


# ============================================================================
# IMAGE INTENSITY NORMALIZATION
# ============================================================================

def normalize_to_uint8(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert image intensities to 8-bit [0, 255].

    This normalization is for the image representation used in
    this experiment.

    The image is NOT treated as a categorical mask.
    """

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    finite = image[
        np.isfinite(image)
    ]

    if finite.size == 0:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    min_value = float(
        finite.min()
    )

    max_value = float(
        finite.max()
    )

    if max_value <= min_value:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    normalized = (
        image - min_value
    ) / (
        max_value - min_value
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return (
        normalized * 255.0
    ).round().astype(
        np.uint8
    )


# ============================================================================
# PCA
# ============================================================================

def calculate_pca_angle(
    binary_mask: np.ndarray,
) -> Tuple[
    float,
    Tuple[float, float],
]:
    """
    Calculate the principal axis of an IVD mask.

    Returns:

        angle_deg
        centroid
    """

    ys, xs = np.where(
        binary_mask > 0
    )

    if len(xs) < 2:
        raise ValueError(
            "Not enough IVD mask pixels for PCA."
        )

    points = np.column_stack(
        (
            xs.astype(np.float64),
            ys.astype(np.float64),
        )
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

    eigenvalues, eigenvectors = (
        np.linalg.eigh(
            covariance
        )
    )

    principal_vector = (
        eigenvectors[
            :,
            np.argmax(
                eigenvalues
            ),
        ]
    )

    vx = principal_vector[0]
    vy = principal_vector[1]

    angle_rad = np.arctan2(
        vy,
        vx,
    )

    angle_deg = np.degrees(
        angle_rad
    )

    return (
        float(angle_deg),
        (
            float(centroid[0]),
            float(centroid[1]),
        ),
    )


# ============================================================================
# PCA ANGLE NORMALIZATION
# ============================================================================

def normalize_pca_angle(
    angle_deg: float,
) -> float:
    """
    Normalize the undirected PCA axis to:

        [-90°, +90°)

    PCA has a 180° axis ambiguity.
    """

    normalized = (
        (angle_deg + 90.0)
        % 180.0
    ) - 90.0

    return float(
        normalized
    )


# ============================================================================
# ROTATION
# ============================================================================

def rotate_image(
    image: np.ndarray,
    angle_deg: float,
    order: int,
) -> np.ndarray:
    """
    Rotate an image.

    MRI:
        order=1 -> linear

    Mask:
        order=0 -> nearest neighbour
    """

    rotated = rotate(
        image,
        angle=angle_deg,
        reshape=True,
        order=order,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    return rotated


def rotate_mri_and_mask(
    mri: np.ndarray,
    mask: np.ndarray,
    rotation_angle: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Apply the SAME geometric rotation to MRI and mask.
    """

    rotated_mri = rotate_image(
        mri,
        rotation_angle,
        MRI_ROTATION_ORDER,
    )

    rotated_mask = rotate_image(
        mask,
        rotation_angle,
        MASK_ROTATION_ORDER,
    )

    rotated_mask = (
        rotated_mask > 0
    ).astype(
        np.uint8
    )

    return (
        rotated_mri,
        rotated_mask,
    )


# ============================================================================
# PADDED IVD CROP -> SQUARE NATIVE ROI
# ============================================================================

def extract_padded_crop(
    rotated_image: np.ndarray,
    rotated_mask: np.ndarray,
    padding_percent: float = PADDING_PERCENT,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Tuple[int, int, int, int],
]:
    """
    Crop the rotated IVD using its mask bounding box.

    Padding is applied independently on all four sides first.

    The padded rectangle is then converted to a SQUARE ROI by expanding
    the shorter dimension using pixels that already exist in the rotated
    MRI image.

    IMPORTANT:
        - The MRI is NOT stretched to make it square.
        - No black pixels are added to make the ROI square.
        - The square is shifted when necessary so it remains inside the
          rotated MRI canvas.
    """

    ys, xs = np.where(
        rotated_mask > 0
    )

    if len(xs) == 0:
        raise ValueError(
            "Rotated IVD mask is empty."
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
        round(
            width
            * padding_percent
        )
    )

    pad_y = int(
        round(
            height
            * padding_percent
        )
    )

    # First create the padded rectangular bounding box.
    padded_x_min = max(
        0,
        x_min - pad_x,
    )

    padded_x_max = min(
        rotated_image.shape[1],
        x_max + pad_x + 1,
    )

    padded_y_min = max(
        0,
        y_min - pad_y,
    )

    padded_y_max = min(
        rotated_image.shape[0],
        y_max + pad_y + 1,
    )

    padded_width = (
        padded_x_max - padded_x_min
    )

    padded_height = (
        padded_y_max - padded_y_min
    )

    if padded_width <= 0 or padded_height <= 0:
        raise ValueError(
            "Invalid padded IVD bounding box."
        )

    # ------------------------------------------------------------------------
    # Convert padded rectangle to square.
    #
    # The square is centered on the padded rectangle. If it crosses an image
    # boundary, shift it back into the image instead of adding black padding.
    # ------------------------------------------------------------------------

    square_side = max(
        padded_width,
        padded_height,
    )

    image_height, image_width = (
        rotated_image.shape[:2]
    )

    if square_side > image_width or square_side > image_height:
        raise ValueError(
            "The requested square ROI is larger than the rotated MRI canvas. "
            f"Square side: {square_side}, "
            f"MRI size: {image_width} x {image_height}. "
            "A square ROI without padding or clipping cannot be created."
        )

    padded_center_x = (
        padded_x_min + padded_x_max - 1
    ) / 2.0

    padded_center_y = (
        padded_y_min + padded_y_max - 1
    ) / 2.0

    square_x_min = int(
        round(
            padded_center_x
            - (square_side - 1) / 2.0
        )
    )

    square_y_min = int(
        round(
            padded_center_y
            - (square_side - 1) / 2.0
        )
    )

    square_x_max = (
        square_x_min + square_side
    )

    square_y_max = (
        square_y_min + square_side
    )

    # ------------------------------------------------------------------------
    # Shift horizontally into the valid MRI canvas.
    # ------------------------------------------------------------------------

    if square_x_min < 0:
        shift = -square_x_min

        square_x_min += shift
        square_x_max += shift

    if square_x_max > image_width:
        shift = (
            square_x_max - image_width
        )

        square_x_min -= shift
        square_x_max -= shift

    # ------------------------------------------------------------------------
    # Shift vertically into the valid MRI canvas.
    # ------------------------------------------------------------------------

    if square_y_min < 0:
        shift = -square_y_min

        square_y_min += shift
        square_y_max += shift

    if square_y_max > image_height:
        shift = (
            square_y_max - image_height
        )

        square_y_min -= shift
        square_y_max -= shift

    # Final safety clipping.
    square_x_min = max(
        0,
        square_x_min,
    )

    square_y_min = max(
        0,
        square_y_min,
    )

    square_x_max = min(
        image_width,
        square_x_max,
    )

    square_y_max = min(
        image_height,
        square_y_max,
    )

    final_width = (
        square_x_max - square_x_min
    )

    final_height = (
        square_y_max - square_y_min
    )

    if final_width != final_height:
        raise RuntimeError(
            "Failed to construct a square ROI without black padding. "
            f"Final ROI: {final_width} x {final_height}"
        )

    square_bbox = (
        square_x_min,
        square_y_min,
        square_x_max,
        square_y_max,
    )

    cropped_image = (
        rotated_image[
            square_y_min:square_y_max,
            square_x_min:square_x_max,
        ]
    )

    cropped_mask = (
        rotated_mask[
            square_y_min:square_y_max,
            square_x_min:square_x_max,
        ]
    )

    return (
        cropped_image,
        cropped_mask,
        square_bbox,
    )


# ============================================================================
# SIMPLE SQUARE RESIZE
# ============================================================================

def resize_square(
    image: np.ndarray,
    target_size: int,
    interpolation: int,
) -> np.ndarray:
    """
    Resize an already-square ROI to target_size x target_size.

    Because the input is square and the output is square, this operation
    does not introduce aspect-ratio distortion.
    """

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            "Cannot resize an empty image."
        )

    if height != width:
        raise ValueError(
            "resize_square() received a non-square image: "
            f"{width} x {height}"
        )

    return cv2.resize(
        image,
        (
            target_size,
            target_size,
        ),
        interpolation=interpolation,
    )


# ============================================================================
# LEGACY ASPECT-RATIO-PRESERVING RESIZE + SQUARE PADDING HELPERS
# ============================================================================

def resize_and_pad_square(
    image: np.ndarray,
    target_size: int,
    interpolation: int,
) -> np.ndarray:
    """
    Resize an image while PRESERVING aspect ratio, then place it
    inside a target_size x target_size square canvas.

    Example:

        Native crop:
            400 x 100

        Target:
            1024

        Resize:
            1024 x 256

        Then:
            1024 x 1024

        with padding above/below.

    The image is NEVER stretched independently in X and Y.
    """

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            "Cannot resize an empty image."
        )

    scale = min(
        target_size / width,
        target_size / height,
    )

    new_width = max(
        1,
        int(round(width * scale)),
    )

    new_height = max(
        1,
        int(round(height * scale)),
    )

    resized = cv2.resize(
        image,
        (
            new_width,
            new_height,
        ),
        interpolation=interpolation,
    )

    # ------------------------------------------------------------------------
    # Create square canvas
    # ------------------------------------------------------------------------

    if image.ndim == 2:

        canvas = np.zeros(
            (
                target_size,
                target_size,
            ),
            dtype=resized.dtype,
        )

    else:

        canvas = np.zeros(
            (
                target_size,
                target_size,
                image.shape[2],
            ),
            dtype=resized.dtype,
        )

    # ------------------------------------------------------------------------
    # Center the resized image
    # ------------------------------------------------------------------------

    x_offset = (
        target_size
        - new_width
    ) // 2

    y_offset = (
        target_size
        - new_height
    ) // 2

    canvas[
        y_offset:
        y_offset + new_height,
        x_offset:
        x_offset + new_width,
    ] = resized

    return canvas


# ============================================================================
# SAVE PNG
# ============================================================================

def save_png(
    path: Path,
    image: np.ndarray,
) -> None:
    """
    Save a grayscale PNG.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if image.dtype != np.uint8:
        image = normalize_to_uint8(
            image
        )

    success = cv2.imwrite(
        str(path),
        image,
    )

    if not success:
        raise IOError(
            f"Could not save image:\n{path}"
        )


# ============================================================================
# MASK RESIZE
# ============================================================================

def resize_mask_and_pad(
    mask: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """
    Resize an IVD mask with aspect ratio preserved and square padding.

    Nearest-neighbour interpolation is mandatory for categorical masks.
    """

    return resize_and_pad_square(
        mask,
        target_size,
        cv2.INTER_NEAREST,
    ).astype(
        np.uint8
    )


# ============================================================================
# MASK OVERLAY FOR COMPARISON
# ============================================================================

def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Create a visualization overlay.

    MRI is grayscale.
    IVD mask is shown in green.
    """

    if image.dtype != np.uint8:
        image = normalize_to_uint8(
            image
        )

    if mask.shape != image.shape:
        raise ValueError(
            "Image and mask dimensions "
            "do not match for overlay."
        )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB,
    )

    foreground = (
        mask > 0
    )

    if np.any(foreground):

        green = np.zeros(
            (
                np.count_nonzero(
                    foreground
                ),
                3,
            ),
            dtype=np.float32,
        )

        green[:, 1] = 255.0

        original = (
            rgb[foreground]
            .astype(np.float32)
        )

        blended = (
            0.45 * original
            + 0.55 * green
        )

        rgb[foreground] = (
            np.clip(
                blended,
                0,
                255,
            )
            .astype(np.uint8)
        )

    return rgb


# ============================================================================
# COMPARISON FIGURE
# ============================================================================

def create_comparison_panel(
    images: Dict[str, np.ndarray],
    masks: Dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    """
    Create a comparison panel.

    Top row:
        MRI crops

    Bottom row:
        MRI + IVD mask overlay

    Each image is displayed in its actual square canvas.
    """

    names = list(
        images.keys()
    )

    number_of_images = len(
        names
    )

    figure, axes = plt.subplots(
        2,
        number_of_images,
        figsize=(
            4 * number_of_images,
            8,
        ),
    )

    if number_of_images == 1:
        axes = np.asarray(
            axes
        ).reshape(
            2,
            1,
        )

    for column, name in enumerate(
        names
    ):

        image = images[name]
        mask = masks[name]

        # --------------------------------------------------------------
        # MRI
        # --------------------------------------------------------------

        axes[
            0,
            column,
        ].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255,
        )

        axes[
            0,
            column,
        ].set_title(
            f"MRI\n{name}"
        )

        axes[
            0,
            column,
        ].axis("off")

        # --------------------------------------------------------------
        # Overlay
        # --------------------------------------------------------------

        overlay = create_overlay(
            image,
            mask,
        )

        axes[
            1,
            column,
        ].imshow(
            overlay
        )

        axes[
            1,
            column,
        ].set_title(
            f"IVD Mask\n{name}"
        )

        axes[
            1,
            column,
        ].axis("off")

    figure.suptitle(
        title,
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================================
# CREATE DIRECTORY STRUCTURE
# ============================================================================

def create_label_directories(
    patient_output_dir: Path,
    ivd_label: int,
) -> Dict[str, Path]:
    """
    Create the requested directory structure for one IVD.
    """

    label_dir = (
        patient_output_dir
        / f"label_{ivd_label}"
    )

    directories = {
        "raw": (
            label_dir
            / "raw"
        ),
        "upscaled": (
            label_dir
            / "upscaled"
        ),
        "raw_comparison": (
            label_dir
            / "raw_comparison"
        ),
        "upscaled_comparison": (
            label_dir
            / "upscaled_comparison"
        ),
    }

    for directory in (
        directories.values()
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    return directories


# ============================================================================
# PROCESS ONE IVD
# ============================================================================

def process_ivd(
    mri_slice: np.ndarray,
    mask_slice: np.ndarray,
    ivd_label: int,
    patient_output_dir: Path,
) -> bool:
    """
    Process one IVD.

    Returns:
        True  -> processed
        False -> skipped
    """

    mask_label = (
        IVD_TO_MASK[
            ivd_label
        ]
    )

    # ------------------------------------------------------------------------
    # Extract ONLY the requested IVD
    # ------------------------------------------------------------------------

    ivd_mask = (
        mask_slice == mask_label
    ).astype(
        np.uint8
    )

    pixel_count = int(
        np.count_nonzero(
            ivd_mask
        )
    )

    if pixel_count == 0:

        print(
            f"IVD {ivd_label} | "
            f"Mask {mask_label} | "
            f"SKIPPED | "
            f"Pixels: 0"
        )

        return False

    # ------------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------------

    pca_angle, centroid = (
        calculate_pca_angle(
            ivd_mask
        )
    )

    normalized_angle = (
        normalize_pca_angle(
            pca_angle
        )
    )

    # ------------------------------------------------------------------------
    # VALIDATED ROTATION
    #
    # This is the direction validated with orientation_rotation_fix.py.
    #
    # Example:
    #
    # PCA:
    #     -143.29°
    #
    # Normalized:
    #     +36.71°
    #
    # Corrective rotation:
    #     +36.71°
    # ------------------------------------------------------------------------

    rotation_angle = (
        normalized_angle
    )

    # ------------------------------------------------------------------------
    # ROTATE
    # ------------------------------------------------------------------------

    rotated_mri, rotated_mask = (
        rotate_mri_and_mask(
            mri=mri_slice,
            mask=ivd_mask,
            rotation_angle=rotation_angle,
        )
    )

    # ------------------------------------------------------------------------
    # CROP
    # ------------------------------------------------------------------------

    (
        native_crop,
        native_crop_mask,
        bbox,
    ) = extract_padded_crop(
        rotated_image=rotated_mri,
        rotated_mask=rotated_mask,
        padding_percent=PADDING_PERCENT,
    )

    native_height, native_width = (
        native_crop.shape[:2]
    )

    # ------------------------------------------------------------------------
    # NORMALIZE NATIVE SQUARE ROI TO 8-BIT
    # ------------------------------------------------------------------------

    native_uint8 = (
        normalize_to_uint8(
            native_crop
        )
    )

    if native_height != native_width:
        raise RuntimeError(
            "Experiment 3 requires a square native ROI, but got "
            f"{native_width} x {native_height}."
        )

    # ------------------------------------------------------------------------
    # DIRECTORIES
    # ------------------------------------------------------------------------

    directories = (
        create_label_directories(
            patient_output_dir,
            ivd_label,
        )
    )

    # ========================================================================
    # RAW FOLDER
    # ========================================================================

    raw_images = {}
    raw_masks = {}

    # ------------------------------------------------------------------------
    # Native
    #
    # IMPORTANT:
    # Experiment 3 makes the native ROI square using ORIGINAL MRI pixels.
    # ------------------------------------------------------------------------

    raw_images[
        "native"
    ] = native_uint8.copy()

    raw_masks[
        "native"
    ] = native_crop_mask.copy()

    save_png(
        directories["raw"]
        / "native.png",
        raw_images["native"],
    )

    # ------------------------------------------------------------------------
    # Target resolutions
    #
    # Native ROI is already square.
    # Therefore a square-to-square resize preserves aspect ratio
    # without any black padding.
    # ------------------------------------------------------------------------

    for resolution in (
        TARGET_RESOLUTIONS
    ):

        image_resized = (
            resize_square(
                native_uint8,
                resolution,
                cv2.INTER_CUBIC,
            )
        )

        mask_resized = (
            resize_square(
                native_crop_mask,
                resolution,
                cv2.INTER_NEAREST,
            ).astype(np.uint8)
        )

        key = str(
            resolution
        )

        raw_images[
            key
        ] = image_resized

        raw_masks[
            key
        ] = mask_resized

        save_png(
            directories["raw"]
            / f"{resolution}.png",
            image_resized,
        )

    # ========================================================================
    # RAW COMPARISON
    # ========================================================================

    create_comparison_panel(
        images=raw_images,
        masks=raw_masks,
        output_path=(
            directories[
                "raw_comparison"
            ]
            / "comparison.png"
        ),
        title=(
            f"Patient {PATIENT_ID} | "
            f"IVD Label {ivd_label} | "
            f"RAW Resolution Comparison\n"
            f"Native Crop: "
            f"{native_width} × "
            f"{native_height} | "
            f"PCA: {pca_angle:.2f}° | "
            f"Rotation: {rotation_angle:.2f}° | "
            f"Padding: "
            f"{PADDING_PERCENT * 100:.0f}%"
        ),
    )

    # ========================================================================
    # UPSCALED FOLDER
    # ========================================================================

    upscaled_images = {}
    upscaled_masks = {}

    # ------------------------------------------------------------------------
    # Native -> 1024 x 1024
    #
    # Native ROI is already square, so this is a direct
    # square-to-square resize.
    # ------------------------------------------------------------------------

    upscaled_images[
        "native"
    ] = resize_square(
        native_uint8,
        1024,
        cv2.INTER_CUBIC,
    )

    upscaled_masks[
        "native"
    ] = resize_square(
        native_crop_mask,
        1024,
        cv2.INTER_NEAREST,
    ).astype(np.uint8)

    save_png(
        directories["upscaled"]
        / "native.png",
        upscaled_images["native"],
    )

    # ------------------------------------------------------------------------
    # Every RAW resolution -> 1024 x 1024
    # ------------------------------------------------------------------------

    for resolution in (
        TARGET_RESOLUTIONS
    ):

        key = str(
            resolution
        )

        raw_image = raw_images[
            key
        ]

        raw_mask = raw_masks[
            key
        ]

        if resolution == 1024:

            # Already 1024 x 1024.
            final_image = (
                raw_image.copy()
            )

            final_mask = (
                raw_mask.copy()
            )

        else:

            # Every RAW image is already square.
            # Therefore this remains a square-to-square resize.
            final_image = resize_square(
                raw_image,
                1024,
                cv2.INTER_CUBIC,
            )

            final_mask = resize_square(
                raw_mask,
                1024,
                cv2.INTER_NEAREST,
            )

            final_mask = (
                final_mask > 0
            ).astype(
                np.uint8
            )

        upscaled_images[
            key
        ] = final_image

        upscaled_masks[
            key
        ] = final_mask

        save_png(
            directories["upscaled"]
            / f"{resolution}.png",
            final_image,
        )

    # ========================================================================
    # UPSCALED COMPARISON
    # ========================================================================

    create_comparison_panel(
        images=upscaled_images,
        masks=upscaled_masks,
        output_path=(
            directories[
                "upscaled_comparison"
            ]
            / "comparison.png"
        ),
        title=(
            f"Patient {PATIENT_ID} | "
            f"IVD Label {ivd_label} | "
            f"ALL REPRESENTATIONS AT 1024 × 1024\n"
            f"PCA: {pca_angle:.2f}° | "
            f"Rotation: {rotation_angle:.2f}° | "
            f"Padding: "
            f"{PADDING_PERCENT * 100:.0f}%"
        ),
    )

    # ------------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------------

    print(
        f"IVD {ivd_label} | "
        f"Mask {mask_label} | "
        f"Pixels: {pixel_count:5d} | "
        f"PCA: {pca_angle:8.2f}° | "
        f"Normalized: {normalized_angle:8.2f}° | "
        f"Rotation: {rotation_angle:8.2f}° | "
        f"Native Crop: "
        f"{native_width} × "
        f"{native_height}"
    )

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 78)
    print("DISC ROI SQUARE RESOLUTION EXPERIMENT")
    print("=" * 78)

    print(
        f"Patient             : "
        f"{PATIENT_ID}"
    )

    print(
        f"MRI MHA             : "
        f"{MRI_FILE}"
    )

    print(
        f"Disc Mask MHA       : "
        f"{MASK_FILE}"
    )

    # ------------------------------------------------------------------------
    # Validate input files
    # ------------------------------------------------------------------------

    if not MRI_FILE.exists():

        raise FileNotFoundError(
            f"MRI MHA not found:\n"
            f"{MRI_FILE}"
        )

    if not MASK_FILE.exists():

        raise FileNotFoundError(
            f"Mask MHA not found:\n"
            f"{MASK_FILE}"
        )

    # ------------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------------

    mri_volume = load_mha(
        MRI_FILE
    )

    mask_volume = load_mha(
        MASK_FILE
    )

    if (
        mri_volume.shape
        != mask_volume.shape
    ):

        raise ValueError(
            "MRI and mask dimensions "
            "do not match.\n"
            f"MRI : {mri_volume.shape}\n"
            f"Mask: {mask_volume.shape}"
        )

    # ------------------------------------------------------------------------
    # Slice axis
    # ------------------------------------------------------------------------

    slice_axis = (
        select_slice_axis(
            mri_volume
        )
    )

    slice_count = (
        mri_volume.shape[
            slice_axis
        ]
    )

    print(
        f"Volume Size         : "
        f"{mri_volume.shape}"
    )

    print(
        f"Slice Axis          : "
        f"{slice_axis}"
    )

    print(
        f"Slice Count         : "
        f"{slice_count}"
    )

    # ------------------------------------------------------------------------
    # Medial slice
    # ------------------------------------------------------------------------

    medial_index = (
        slice_count // 2
    )

    slice_number = (
        medial_index + 1
    )

    mri_slice = (
        extract_slice(
            mri_volume,
            slice_axis,
            medial_index,
        )
    )

    mask_slice = (
        extract_slice(
            mask_volume,
            slice_axis,
            medial_index,
        )
    )

    print(
        f"Medial Slice Index  : "
        f"{medial_index}"
    )

    print(
        f"Medial Slice        : "
        f"{slice_number}"
    )

    print(
        f"Native Slice Shape  : "
        f"{mri_slice.shape}"
    )

    # ------------------------------------------------------------------------
    # Mask values
    # ------------------------------------------------------------------------

    unique_values = np.unique(
        mask_slice
    )

    print(
        f"Mask Unique Values  : "
        f"{unique_values.tolist()}"
    )

    print(
        f"IVD Labels Found    : "
        f"{list(IVD_TO_MASK.keys())}"
    )

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------

    patient_output_dir = (
        OUTPUT_ROOT
        / PATIENT_ID
        / f"slice_{slice_number:03d}"
    )

    patient_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Output Directory    : "
        f"{patient_output_dir}"
    )

    print()
    print("-" * 78)
    print("PROCESSING IVDS")
    print("-" * 78)

    processed = 0

    # ------------------------------------------------------------------------
    # Process IVD 1-6
    # ------------------------------------------------------------------------

    for ivd_label in range(
        1,
        7,
    ):

        success = process_ivd(
            mri_slice=mri_slice,
            mask_slice=mask_slice,
            ivd_label=ivd_label,
            patient_output_dir=patient_output_dir,
        )

        if success:
            processed += 1

    # ------------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("SQUARE ROI RESOLUTION EXPERIMENT COMPLETED")
    print("=" * 78)

    print(
        f"IVDs Requested     : 6"
    )

    print(
        f"IVDs Processed     : "
        f"{processed}"
    )

    print(
        f"Slice Axis         : "
        f"{slice_axis}"
    )

    print(
        f"Medial Slice       : "
        f"{slice_number}"
    )

    print(
        f"Padding             : "
        f"{PADDING_PERCENT * 100:.0f}%"
    )

    print(
        "Raw Resolutions     : "
        "native square + 512 + 768 + "
        "1024 + 1280 + 1536"
    )

    print(
        "Upscaled Resolution : "
        "all representations -> 1024 × 1024"
    )

    print(
        f"Output Directory   : "
        f"{patient_output_dir}"
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
        "3. Slice axis = lowest volume dimension."
    )

    print(
        "4. IVD labels 1-6 map to SPIDER masks 201-206."
    )

    print(
        "5. PCA is calculated independently for every IVD."
    )

    print(
        "6. PCA axis is normalized to [-90°, +90°)."
    )

    print(
        "7. Validated corrective rotation is applied."
    )

    print(
        "8. MRI rotation uses linear interpolation."
    )

    print(
        "9. IVD mask rotation uses nearest-neighbour."
    )

    print(
        "10. 30% padding is applied on each side."
    )

    print(
        "11. The native ROI is converted to a square using ORIGINAL MRI pixels."
    )

    print(
        "12. No black padding is added to create the native square ROI."
    )

    print(
        "13. RAW representations are square-to-square resizes."
    )

    print(
        "14. UPSCALED images are all 1024 × 1024."
    )

    print(
        "15. No anisotropic rectangular-to-square stretching is performed."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()