"""
Production dataset generator for SPIDER Pfirrmann MRI-CORE classification.

Finalized preprocessing decisions:
- Source: original .mha files only
- Slice axis: lowest of the 3 volume dimensions
- Medial slice: ceil(N / 2), using 1-based slice numbering
- Offsets: -2, -1, 0, +1, +2
- IVD labels: 1..N map to SPIDER disc-mask labels 201..(200+N)
- Orientation: PCA on each individual IVD mask
- Rotation: validated corrective rotation
- Padding: 10% on each side before square construction
- Square ROI: expand the shorter dimension using existing rotated MRI pixels
- No black padding
- No anisotropic rectangular-to-square stretching
- MRI: per-crop min-max normalization -> uint8 grayscale -> 1024x1024
- Disc mask: cropped at native resolution, nearest-neighbour, NOT resized to 1024
- Master GT: one row per generated disc sample

Run from project root:
    python -m preprocessing.build_training_dataset

Small test:
    python -m preprocessing.build_training_dataset --patients 100 101

Limit number of patients:
    python -m preprocessing.build_training_dataset --max-patients 3

Force regeneration:
    python -m preprocessing.build_training_dataset --patients 100 101 --overwrite
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import rotate


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "images"
RAW_MASKS_DIR = PROJECT_ROOT / "data" / "raw" / "masks"
RADIOLOGICAL_GT = PROJECT_ROOT / "data" / "raw" / "radiological_gradings.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_OUTPUT_DIR = PROCESSED_DIR / "images"
MASK_OUTPUT_DIR = PROCESSED_DIR / "masks"

MASTER_GT_PATH = PROCESSED_DIR / "master_gt.csv"
ERROR_LOG_PATH = PROCESSED_DIR / "generation_errors.csv"


# ============================================================================
# FINALIZED CONFIGURATION
# ============================================================================

OFFSETS = (-2, -1, 0, 1, 2)

PADDING_PERCENT = 0.10

FINAL_IMAGE_SIZE = 1024

# SPIDER disc-mask convention used in the project:
# project IVD label 1 -> mask label 201
# project IVD label 2 -> mask label 202
# ...
def ivd_to_mask_label(ivd_label: int) -> int:
    return 200 + int(ivd_label)


# numpy volume axis -> anatomical/storage axis name used in metadata
AXIS_NAMES = {
    0: "Z",
    1: "Y",
    2: "X",
}


# ============================================================================
# HELPERS
# ============================================================================

def load_mha(path: Path) -> np.ndarray:
    """Read an MHA volume while preserving its native numeric values."""
    if not path.exists():
        raise FileNotFoundError(f"MHA file not found: {path}")

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image)

    if array.ndim != 3:
        raise ValueError(
            f"Expected a 3-D MHA volume, got shape {array.shape} from {path}"
        )

    return array


def select_lowest_dimension_axis(volume: np.ndarray) -> int:
    """
    Select the slice axis using the project's finalized rule:
    choose the axis having the smallest dimension.
    """
    return int(np.argmin(volume.shape))


def medial_slice_number(slice_count: int) -> int:
    """
    Return the medial slice using 1-based numbering.

    Examples:
        18 -> 9
        19 -> 10
        17 -> 9

    This is ceil(N / 2).
    """
    return int(math.ceil(slice_count / 2.0))


def extract_slice(volume: np.ndarray, axis: int, slice_number: int) -> np.ndarray:
    """Extract a 1-based slice number along the selected numpy axis."""
    index = slice_number - 1

    if index < 0 or index >= volume.shape[axis]:
        raise IndexError(
            f"Slice {slice_number} is outside axis {axis} with "
            f"{volume.shape[axis]} slices."
        )

    return np.take(volume, index, axis=axis)


def pca_orientation(mask: np.ndarray) -> tuple[float, float]:
    """
    Calculate the PCA orientation of one binary IVD mask.

    Returns:
        pca_angle_degrees:
            Raw PCA axis angle in [-90, 90) after axis normalization.
        normalized_angle_degrees:
            Same normalized anatomical orientation used for correction.

    The PCA axis is an undirected line, so theta and theta +/- 180 degrees
    represent the same axis.
    """
    ys, xs = np.nonzero(mask)

    if len(xs) < 2:
        raise ValueError("Not enough IVD mask pixels for PCA.")

    points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    centered = points - points.mean(axis=0, keepdims=True)

    covariance = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal_vector = eigenvectors[:, int(np.argmax(eigenvalues))]

    angle = math.degrees(math.atan2(principal_vector[1], principal_vector[0]))

    # Normalize the undirected PCA axis into [-90, 90).
    normalized = ((angle + 90.0) % 180.0) - 90.0

    return float(angle), float(normalized)


def rotate_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    rotation_angle: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate MRI and mask with identical geometry.

    MRI:
        linear interpolation, continuous intensity image.

    Mask:
        nearest-neighbour interpolation, categorical labels.
    """
    rotated_image = rotate(
        image.astype(np.float32),
        rotation_angle,
        reshape=True,
        order=1,
        mode="nearest",
        prefilter=True,
    )

    rotated_mask = rotate(
        mask.astype(np.uint16),
        rotation_angle,
        reshape=True,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    )

    return rotated_image, rotated_mask


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return x_min, y_min, x_max, y_max for non-zero mask pixels."""
    ys, xs = np.nonzero(mask)

    if len(xs) == 0:
        raise ValueError("Mask contains no foreground pixels.")

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )


def crop_square_from_rotated_image(
    rotated_image: np.ndarray,
    rotated_mask: np.ndarray,
    padding_percent: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Create the final square ROI.

    Important:
    - padding is applied to the IVD bounding box
    - the square is then created by expanding the shorter dimension
    - expansion uses existing rotated MRI pixels
    - no black padding is introduced
    - no anisotropic stretching is performed
    """
    x_min, y_min, x_max, y_max = bounding_box(rotated_mask)

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    pad_x = max(1, int(round(width * padding_percent)))
    pad_y = max(1, int(round(height * padding_percent)))

    padded_x_min = x_min - pad_x
    padded_x_max = x_max + pad_x
    padded_y_min = y_min - pad_y
    padded_y_max = y_max + pad_y

    # Clamp the padded rectangle to the actual rotated image.
    h, w = rotated_image.shape[:2]

    padded_x_min = max(0, padded_x_min)
    padded_y_min = max(0, padded_y_min)
    padded_x_max = min(w - 1, padded_x_max)
    padded_y_max = min(h - 1, padded_y_max)

    padded_width = padded_x_max - padded_x_min + 1
    padded_height = padded_y_max - padded_y_min + 1

    square_side = max(padded_width, padded_height)

    # Center the square on the padded ROI.
    center_x = (padded_x_min + padded_x_max) / 2.0
    center_y = (padded_y_min + padded_y_max) / 2.0

    square_x_min = int(round(center_x - (square_side - 1) / 2.0))
    square_y_min = int(round(center_y - (square_side - 1) / 2.0))

    square_x_max = square_x_min + square_side - 1
    square_y_max = square_y_min + square_side - 1

    # Shift the square back into the image if necessary.
    if square_x_min < 0:
        shift = -square_x_min
        square_x_min += shift
        square_x_max += shift

    if square_y_min < 0:
        shift = -square_y_min
        square_y_min += shift
        square_y_max += shift

    if square_x_max >= w:
        shift = square_x_max - (w - 1)
        square_x_min -= shift
        square_x_max -= shift

    if square_y_max >= h:
        shift = square_y_max - (h - 1)
        square_y_min -= shift
        square_y_max -= shift

    # Final safety check.
    if (
        square_x_min < 0
        or square_y_min < 0
        or square_x_max >= w
        or square_y_max >= h
    ):
        raise ValueError(
            "Could not fit square ROI inside rotated image. "
            f"Image={w}x{h}, requested square={square_side}."
        )

    crop_image = rotated_image[
        square_y_min : square_y_max + 1,
        square_x_min : square_x_max + 1,
    ]

    crop_mask = rotated_mask[
        square_y_min : square_y_max + 1,
        square_x_min : square_x_max + 1,
    ]

    if crop_image.shape[0] != crop_image.shape[1]:
        raise RuntimeError(
            f"Square ROI check failed: {crop_image.shape}"
        )

    if crop_mask.shape != crop_image.shape:
        raise RuntimeError(
            f"MRI/mask ROI mismatch: MRI={crop_image.shape}, "
            f"mask={crop_mask.shape}"
        )

    # Confirm that the complete rotated IVD mask remains inside the crop.
    local_mask = crop_mask > 0
    if not np.any(local_mask):
        raise RuntimeError("IVD disappeared from the final square crop.")

    metadata = {
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "native_bbox_width": width,
        "native_bbox_height": height,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "padded_width": padded_width,
        "padded_height": padded_height,
        "square_side": square_side,
        "final_roi_width": int(crop_image.shape[1]),
        "final_roi_height": int(crop_image.shape[0]),
    }

    return crop_image, crop_mask, metadata


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Per-crop min-max normalization to uint8.

    Original numeric MRI values are first converted to float32.
    The finite crop range is mapped to [0, 255].
    """
    image = np.asarray(image, dtype=np.float32)

    finite = np.isfinite(image)

    if not np.any(finite):
        raise ValueError("MRI crop contains no finite intensity values.")

    values = image[finite]
    min_value = float(values.min())
    max_value = float(values.max())

    if max_value <= min_value:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = (image - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)

    return np.rint(normalized * 255.0).astype(np.uint8)


def save_grayscale_png(array: np.ndarray, path: Path) -> None:
    """Save a 2-D uint8 array as grayscale PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim != 2:
        raise ValueError(f"Expected 2-D grayscale array, got {array.shape}")

    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    """
    Save the native-resolution categorical disc mask.

    The mask contains only the selected IVD label and background.
    We preserve its categorical identity as uint16 because the original
    SPIDER mask labels are 201, 202, ... and should not be converted to
    intensity values.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got {mask.shape}")

    Image.fromarray(mask.astype(np.uint16), mode="I;16").save(path)


def load_ground_truth(path: Path) -> pd.DataFrame:
    """Load and validate radiological_gradings.csv."""
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")

    gt = pd.read_csv(path)

    required = {
        "Patient",
        "IVD label",
        "Pfirrman grade",
    }

    missing = sorted(required - set(gt.columns))
    if missing:
        raise ValueError(
            f"Missing required GT columns: {missing}\n"
            f"Available columns: {gt.columns.tolist()}"
        )

    gt = gt.copy()

    gt["Patient"] = pd.to_numeric(gt["Patient"], errors="coerce")
    gt["IVD label"] = pd.to_numeric(gt["IVD label"], errors="coerce")
    gt["Pfirrman grade"] = pd.to_numeric(
        gt["Pfirrman grade"],
        errors="coerce",
    )

    gt = gt.dropna(
        subset=["Patient", "IVD label", "Pfirrman grade"]
    ).copy()

    gt["Patient"] = gt["Patient"].astype(int)
    gt["IVD label"] = gt["IVD label"].astype(int)
    gt["Pfirrman grade"] = gt["Pfirrman grade"].astype(int)

    invalid_grade = ~gt["Pfirrman grade"].isin([1, 2, 3, 4, 5])
    if invalid_grade.any():
        bad = gt.loc[invalid_grade, ["Patient", "IVD label", "Pfirrman grade"]]
        raise ValueError(
            "Invalid Pfirrmann grades found:\n"
            f"{bad.to_string(index=False)}"
        )

    duplicate_pairs = gt.duplicated(
        subset=["Patient", "IVD label"],
        keep=False,
    )

    if duplicate_pairs.any():
        dup = gt.loc[
            duplicate_pairs,
            ["Patient", "IVD label", "Pfirrman grade"],
        ]
        raise ValueError(
            "Duplicate Patient + IVD label records found in GT:\n"
            f"{dup.to_string(index=False)}"
        )

    return gt


def patient_ids_from_gt(gt: pd.DataFrame) -> list[int]:
    return sorted(gt["Patient"].unique().tolist())


def resolve_patient_mha(directory: Path, patient_id: int) -> Path:
    """
    Resolve the project's current filename convention:
        {patient}_t2.mha
    """
    path = directory / f"{patient_id}_t2.mha"

    if path.exists():
        return path

    raise FileNotFoundError(
        f"T2 MHA not found for Patient {patient_id}: {path}"
    )


def find_disc_labels_in_slice(mask_slice: np.ndarray) -> list[int]:
    """
    Find project IVD labels from the radiological GT that are actually
    present in this slice.

    A project IVD label L corresponds to mask value 200 + L.
    """
    unique_values = set(np.unique(mask_slice).tolist())

    labels = []
    for value in sorted(unique_values):
        if 201 <= value <= 299:
            labels.append(int(value - 200))

    return labels


def crop_single_ivd(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    ivd_label: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Rotate/crop one IVD using its own SPIDER mask label.
    """
    mask_label = ivd_to_mask_label(ivd_label)
    binary_mask = (mask_slice == mask_label).astype(np.uint8)

    pixel_count = int(np.count_nonzero(binary_mask))

    if pixel_count == 0:
        raise ValueError(
            f"IVD {ivd_label} / mask {mask_label} is absent in this slice."
        )

    pca_angle, normalized_angle = pca_orientation(binary_mask)

    # Validated project convention:
    # rotate by the normalized PCA angle so the IVD long axis is horizontal.
    rotation_angle = normalized_angle

    rotated_image, rotated_mask_full = rotate_image_and_mask(
        image_slice,
        binary_mask,
        rotation_angle,
    )

    crop_image, crop_mask, roi_meta = crop_square_from_rotated_image(
        rotated_image,
        rotated_mask_full,
        PADDING_PERCENT,
    )

    roi_meta.update(
        {
            "mask_label": mask_label,
            "mask_pixel_count": pixel_count,
            "pca_angle": pca_angle,
            "normalized_angle": normalized_angle,
            "rotation_angle": rotation_angle,
        }
    )

    return crop_image, crop_mask, roi_meta


# ============================================================================
# PATIENT PROCESSING
# ============================================================================

def process_patient(
    patient_id: int,
    gt_patient: pd.DataFrame,
    overwrite: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Process one patient.

    Returns:
        rows: successful Master GT rows
        errors: per-sample/per-patient errors
    """
    rows: list[dict] = []
    errors: list[dict] = []

    image_path = resolve_patient_mha(RAW_IMAGES_DIR, patient_id)
    mask_path = resolve_patient_mha(RAW_MASKS_DIR, patient_id)

    image_volume = load_mha(image_path)
    mask_volume = load_mha(mask_path)

    if image_volume.shape != mask_volume.shape:
        raise ValueError(
            f"MRI/mask volume mismatch for Patient {patient_id}: "
            f"MRI={image_volume.shape}, mask={mask_volume.shape}"
        )

    slice_axis = select_lowest_dimension_axis(image_volume)
    slice_axis_name = AXIS_NAMES[slice_axis]
    slice_count = int(image_volume.shape[slice_axis])

    medial = medial_slice_number(slice_count)

    # All GT discs expected for this patient.
    gt_disc_map = {
        int(row["IVD label"]): int(row["Pfirrman grade"])
        for _, row in gt_patient.iterrows()
    }

    number_of_discs = len(gt_disc_map)

    print(
        f"\nPatient {patient_id} | "
        f"Volume={image_volume.shape} | "
        f"Slice axis={slice_axis_name} | "
        f"Slices={slice_count} | "
        f"Medial={medial} | "
        f"GT discs={number_of_discs}"
    )

    for offset in OFFSETS:
        slice_number = medial + offset

        if slice_number < 1 or slice_number > slice_count:
            errors.append(
                {
                    "Patient_ID": patient_id,
                    "Offset": offset,
                    "Slice_Number": slice_number,
                    "Error_Type": "SliceOutOfRange",
                    "Error": (
                        f"Selected slice {slice_number} outside "
                        f"1..{slice_count}"
                    ),
                }
            )
            continue

        image_slice = extract_slice(
            image_volume,
            slice_axis,
            slice_number,
        )
        mask_slice = extract_slice(
            mask_volume,
            slice_axis,
            slice_number,
        )

        present_labels = find_disc_labels_in_slice(mask_slice)

        # Only process labels that exist in both the source GT and this mask.
        process_labels = [
            label for label in present_labels
            if label in gt_disc_map
        ]

        for ivd_label in sorted(process_labels):
            grade = gt_disc_map[ivd_label]

            image_out = (
                IMAGE_OUTPUT_DIR
                / str(patient_id)
                / f"slice_{slice_number:03d}"
                / f"label_{ivd_label}.png"
            )

            mask_out = (
                MASK_OUTPUT_DIR
                / str(patient_id)
                / f"slice_{slice_number:03d}"
                / f"label_{ivd_label}.png"
            )

            if (
                image_out.exists()
                and mask_out.exists()
                and not overwrite
            ):
                print(
                    f"  Slice {slice_number:03d} | "
                    f"IVD {ivd_label} | already exists, skipped"
                )
                continue

            try:
                crop_image_float, crop_mask, roi_meta = crop_single_ivd(
                    image_slice=image_slice,
                    mask_slice=mask_slice,
                    ivd_label=ivd_label,
                )

                # Final intensity conversion happens ONLY after the ROI
                # has been selected. This is the finalized uint8 workflow.
                crop_image_uint8 = normalize_to_uint8(crop_image_float)

                # Ensure the final MRI is exactly 1024x1024.
                if crop_image_uint8.shape != (
                    FINAL_IMAGE_SIZE,
                    FINAL_IMAGE_SIZE,
                ):
                    resized = Image.fromarray(
                        crop_image_uint8,
                        mode="L",
                    ).resize(
                        (FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE),
                        resample=Image.Resampling.BICUBIC,
                    )
                    crop_image_uint8 = np.asarray(
                        resized,
                        dtype=np.uint8,
                    )

                # IMPORTANT:
                # Mask is deliberately NOT resized to 1024x1024.
                # It remains at its native cropped resolution.
                save_grayscale_png(crop_image_uint8, image_out)
                save_mask_png(crop_mask, mask_out)

                row = {
                    "Sample_ID": f"P{patient_id}_S{slice_number:03d}_IVD{ivd_label}",
                    "Filename": image_out.name,
                    "Patient_ID": patient_id,
                    "MRI_Filename": image_path.name,
                    "Mask_Filename": mask_path.name,
                    "Slice_Count": slice_count,
                    "Slice_Axis": slice_axis_name,
                    "Slice_Axis_Index": slice_axis,
                    "Native_Slice_Height": int(image_slice.shape[0]),
                    "Native_Slice_Width": int(image_slice.shape[1]),
                    "Medial_Slice_Number": medial,
                    "Offset": offset,
                    "Slice_Number": slice_number,
                    "Number_of_Discs": number_of_discs,
                    "IVD_Label": ivd_label,
                    "Mask_Label": ivd_to_mask_label(ivd_label),
                    "Pfirrmann_Grade": grade,
                    "PCA_Angle": roi_meta["pca_angle"],
                    "Normalized_Angle": roi_meta["normalized_angle"],
                    "Rotation_Angle": roi_meta["rotation_angle"],
                    "x_min": roi_meta["x_min"],
                    "y_min": roi_meta["y_min"],
                    "x_max": roi_meta["x_max"],
                    "y_max": roi_meta["y_max"],
                    "Native_BBox_Width": roi_meta["native_bbox_width"],
                    "Native_BBox_Height": roi_meta["native_bbox_height"],
                    "Pad_X": roi_meta["pad_x"],
                    "Pad_Y": roi_meta["pad_y"],
                    "Padded_Width": roi_meta["padded_width"],
                    "Padded_Height": roi_meta["padded_height"],
                    "Native_ROI_Width": roi_meta["final_roi_width"],
                    "Native_ROI_Height": roi_meta["final_roi_height"],
                    "Final_ROI_Width": int(crop_image_uint8.shape[1]),
                    "Final_ROI_Height": int(crop_image_uint8.shape[0]),
                    "MRI_Output_Path": str(image_out),
                    "Mask_Output_Path": str(mask_out),
                    "MRI_Dtype": str(crop_image_uint8.dtype),
                    "MRI_Channels": 1,
                    "MRI_Intensity_Normalization": "per-crop min-max [0,255]",
                    "Mask_Resolution": (
                        f"{crop_mask.shape[1]}x{crop_mask.shape[0]}"
                    ),
                }

                rows.append(row)

                print(
                    f"  Slice {slice_number:03d} | "
                    f"IVD {ivd_label} | "
                    f"Grade {grade} | "
                    f"Native ROI {crop_image_float.shape[1]}x"
                    f"{crop_image_float.shape[0]} | "
                    f"Final 1024x1024"
                )

            except Exception as exc:
                errors.append(
                    {
                        "Patient_ID": patient_id,
                        "Offset": offset,
                        "Slice_Number": slice_number,
                        "IVD_Label": ivd_label,
                        "Error_Type": type(exc).__name__,
                        "Error": str(exc),
                    }
                )

    return rows, errors


# ============================================================================
# MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build finalized SPIDER cropped-disc training dataset."
    )

    parser.add_argument(
        "--patients",
        nargs="+",
        type=int,
        default=None,
        help="Specific patient IDs to process.",
    )

    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Process only the first N patients from the GT.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed samples.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 78)
    print("SPIDER PFIRRMANN TRAINING DATASET GENERATION")
    print("=" * 78)

    print(f"Radiological GT : {RADIOLOGICAL_GT}")
    print(f"Raw MRI         : {RAW_IMAGES_DIR}")
    print(f"Raw masks       : {RAW_MASKS_DIR}")
    print(f"Images output   : {IMAGE_OUTPUT_DIR}")
    print(f"Masks output    : {MASK_OUTPUT_DIR}")
    print(f"Master GT       : {MASTER_GT_PATH}")

    print("\nFinal configuration:")
    print(f"  Slice axis rule     : lowest volume dimension")
    print(f"  Offsets             : {OFFSETS}")
    print(f"  Padding             : {PADDING_PERCENT * 100:.0f}%")
    print(f"  Final MRI size      : {FINAL_IMAGE_SIZE} x {FINAL_IMAGE_SIZE}")
    print(f"  MRI format          : uint8 grayscale")
    print(f"  MRI normalization   : per-crop min-max")
    print(f"  Mask format         : native cropped resolution")
    print(f"  Black padding       : NO")
    print(f"  Anisotropic stretch : NO")

    gt = load_ground_truth(RADIOLOGICAL_GT)

    available_patients = patient_ids_from_gt(gt)

    if args.patients is not None:
        requested = sorted(set(args.patients))
        patients = [p for p in requested if p in available_patients]

        missing = [p for p in requested if p not in available_patients]
        if missing:
            print(f"\nWARNING: Patients absent from radiological GT: {missing}")
    else:
        patients = available_patients

    if args.max_patients is not None:
        patients = patients[: args.max_patients]

    print(f"\nPatients selected: {len(patients)}")

    if not patients:
        raise RuntimeError("No patients selected for processing.")

    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_errors: list[dict] = []

    for index, patient_id in enumerate(patients, start=1):
        print(
            f"\n[{index:03d}/{len(patients):03d}] "
            f"PROCESSING PATIENT {patient_id}"
        )

        gt_patient = gt[gt["Patient"] == patient_id].copy()

        try:
            rows, errors = process_patient(
                patient_id=patient_id,
                gt_patient=gt_patient,
                overwrite=args.overwrite,
            )

            all_rows.extend(rows)
            all_errors.extend(errors)

        except Exception as exc:
            error = {
                "Patient_ID": patient_id,
                "Offset": "",
                "Slice_Number": "",
                "IVD_Label": "",
                "Error_Type": type(exc).__name__,
                "Error": str(exc),
            }

            all_errors.append(error)

            print(
                f"  PATIENT FAILED: {type(exc).__name__}: {exc}"
            )

    # Save Master GT.
    if all_rows:
        master_df = pd.DataFrame(all_rows)

        # Stable ordering for reproducibility.
        master_df = master_df.sort_values(
            by=[
                "Patient_ID",
                "Slice_Number",
                "IVD_Label",
            ]
        ).reset_index(drop=True)

        master_df.to_csv(
            MASTER_GT_PATH,
            index=False,
        )
    else:
        # Still create a correctly structured empty CSV.
        pd.DataFrame().to_csv(
            MASTER_GT_PATH,
            index=False,
        )

    # Save errors separately.
    if all_errors:
        error_df = pd.DataFrame(all_errors)
        error_df.to_csv(
            ERROR_LOG_PATH,
            index=False,
        )

    print("\n" + "=" * 78)
    print("DATASET GENERATION SUMMARY")
    print("=" * 78)
    print(f"Patients processed        : {len(patients)}")
    print(f"Generated disc samples    : {len(all_rows)}")
    print(f"Errors / skipped samples  : {len(all_errors)}")
    print(f"Master GT                 : {MASTER_GT_PATH}")

    if all_errors:
        print(f"Error log                 : {ERROR_LOG_PATH}")

    print("\nIMPORTANT:")
    print("1. The original .mha files were used directly.")
    print("2. No unpacked PNG dataset is used.")
    print("3. Slice axis = lowest volume dimension.")
    print("4. Medial slice = ceil(N / 2), using 1-based numbering.")
    print("5. Five slices are generated using offsets -2,-1,0,+1,+2.")
    print("6. IVD label L maps to SPIDER disc-mask label 200 + L.")
    print("7. PCA is calculated independently for every IVD.")
    print("8. Validated corrective rotation is applied.")
    print("9. 10% padding is applied before square construction.")
    print("10. Square ROI uses existing rotated MRI pixels.")
    print("11. No black padding is introduced.")
    print("12. No anisotropic stretching is performed.")
    print("13. MRI crop is min-max normalized per crop and saved as uint8.")
    print("14. Final MRI image is 1024x1024 grayscale.")
    print("15. Disc masks remain at native cropped resolution.")
    print("16. Master GT contains one row per generated disc sample.")
    print("17. Train/validation/test splitting is NOT performed by this script.")
    print("18. Patient-wise splitting will be performed after validation.")


if __name__ == "__main__":
    main()