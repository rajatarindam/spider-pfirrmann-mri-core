"""
Dataset Validation

Performs final validation of the retained Sagittal T2 dataset before
starting dataset preparation.

Validation includes:
- Retained T2 patient count
- MRI/disc-mask filename matching
- MRI/disc-mask dimension matching
- Valid IVD label range
- Missing Pfirrman grades
- Invalid Pfirrman grades
- Orphan ground-truth records
- Medial slice extraction feasibility

No preprocessing or dataset generation is performed.
"""

import sys

import pandas as pd
import SimpleITK as sitk
from colorama import Fore, init

from configs.paths import (
    OUTPUTS_DIR,
    RAW_IMAGES_DIR,
    RAW_MASKS_DIR,
    RADIOLOGICAL_GRADINGS_CSV,
)
from utilities.logger import ExecutionLogger

init(autoreset=True)


# =============================================================================
# Configuration
# =============================================================================

EXCLUDED_PATIENTS = {58}

MIN_IVD_LABEL = 1
MAX_IVD_LABEL = 9

MIN_PFIRRMANN_GRADE = 1
MAX_PFIRRMANN_GRADE = 5

OUTPUT_DIR = OUTPUTS_DIR / "eda" / "dataset_validation"


# =============================================================================
# Helper Functions
# =============================================================================


def get_patient_id(file_path):
    """
    Extract the patient ID from a T2 filename.
    """

    return int(file_path.stem.split("_")[0])


def get_retained_t2_files():
    """
    Return retained Sagittal T2 MRI files after excluding patient 58.
    """

    t2_files = []

    for file_path in sorted(RAW_IMAGES_DIR.glob("*_t2.mha")):

        patient_id = get_patient_id(file_path)

        if patient_id in EXCLUDED_PATIENTS:
            continue

        t2_files.append(file_path)

    return t2_files


def validate_image_mask_filenames(t2_files):
    """
    Check whether every retained MRI has a corresponding disc mask.
    """

    image_names = {
        file_path.name
        for file_path in t2_files
    }

    mask_names = {
        file_path.name
        for file_path in RAW_MASKS_DIR.glob("*_t2.mha")
        if get_patient_id(file_path) not in EXCLUDED_PATIENTS
    }

    missing_masks = image_names - mask_names
    extra_masks = mask_names - image_names

    mismatch_count = (
        len(missing_masks)
        + len(extra_masks)
    )

    return (
        mismatch_count,
        missing_masks,
        extra_masks,
    )


def validate_dimensions(t2_files):
    """
    Check whether every MRI volume and its corresponding disc mask
    have identical dimensions.
    """

    mismatches = []

    for image_path in t2_files:

        mask_path = RAW_MASKS_DIR / image_path.name

        if not mask_path.exists():
            continue

        image = sitk.ReadImage(
            str(image_path)
        )

        disc_mask = sitk.ReadImage(
            str(mask_path)
        )

        if image.GetSize() != disc_mask.GetSize():

            mismatches.append(
                (
                    image_path.name,
                    image.GetSize(),
                    disc_mask.GetSize(),
                )
            )

    return mismatches


def validate_medial_slice_extraction(t2_files):
    """
    Verify that a valid medial slice index can be calculated for
    every retained T2 volume.

    The smallest image dimension is treated as the slice dimension,
    consistent with the validated dataset structure.
    """

    failures = []

    for image_path in t2_files:

        try:

            image = sitk.ReadImage(
                str(image_path)
            )

            size = image.GetSize()

            if len(size) != 3:

                failures.append(
                    (
                        image_path.name,
                        f"Expected 3D image, found "
                        f"dimension {len(size)}",
                    )
                )

                continue

            slice_axis = min(
                range(len(size)),
                key=lambda axis: size[axis],
            )

            slice_count = size[slice_axis]

            if slice_count <= 0:

                failures.append(
                    (
                        image_path.name,
                        "Invalid slice count",
                    )
                )

                continue

            medial_index = slice_count // 2

            if (
                medial_index < 0
                or medial_index >= slice_count
            ):

                failures.append(
                    (
                        image_path.name,
                        f"Invalid medial index "
                        f"{medial_index}",
                    )
                )

        except Exception as exception:

            failures.append(
                (
                    image_path.name,
                    str(exception),
                )
            )

    return failures


def validate_ground_truth(t2_files):
    """
    Validate radiological ground-truth records.

    Valid disc records must:
    - Belong to a retained T2 patient.
    - Have IVD labels from 1 through 9.
    - Have Pfirrman grades from 1 through 5.

    IVD label 0 is a non-disc record and is therefore excluded
    from the valid disc records.
    """

    if not RADIOLOGICAL_GRADINGS_CSV.exists():
        raise FileNotFoundError(
            "Radiological grading file not found: "
            f"{RADIOLOGICAL_GRADINGS_CSV}"
        )

    dataframe = pd.read_csv(
        RADIOLOGICAL_GRADINGS_CSV
    )

    # -------------------------------------------------------------------------
    # Validate expected CSV schema
    # -------------------------------------------------------------------------

    required_columns = {
        "Patient",
        "IVD label",
        "Pfirrman grade",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns in radiological "
            "grading CSV: "
            f"{sorted(missing_columns)}"
        )

    # -------------------------------------------------------------------------
    # Convert relevant columns to numeric
    # -------------------------------------------------------------------------

    dataframe["Patient"] = pd.to_numeric(
        dataframe["Patient"],
        errors="coerce",
    )

    dataframe["IVD label"] = pd.to_numeric(
        dataframe["IVD label"],
        errors="coerce",
    )

    dataframe["Pfirrman grade"] = pd.to_numeric(
        dataframe["Pfirrman grade"],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Retained patient IDs
    # -------------------------------------------------------------------------

    retained_patient_ids = {
        get_patient_id(file_path)
        for file_path in t2_files
    }

    # -------------------------------------------------------------------------
    # Valid disc records
    #
    # IVD label 0 is intentionally excluded because it does not
    # represent an actual disc.
    # -------------------------------------------------------------------------

    valid_disc_records = dataframe[
        dataframe["IVD label"].between(
            MIN_IVD_LABEL,
            MAX_IVD_LABEL,
        )
        & dataframe["Patient"].isin(
            retained_patient_ids
        )
    ].copy()

    # -------------------------------------------------------------------------
    # Invalid IVD labels
    #
    # Values 0-9 are allowed in the raw CSV.
    # 0 is a non-disc record.
    # 1-9 are valid disc labels.
    # Anything outside 0-9 is invalid.
    # -------------------------------------------------------------------------

    invalid_ivd_records = dataframe[
        dataframe["IVD label"].notna()
        & ~dataframe["IVD label"].between(
            0,
            MAX_IVD_LABEL,
        )
    ]

    # -------------------------------------------------------------------------
    # Missing Pfirrman grades
    # -------------------------------------------------------------------------

    missing_grade_records = valid_disc_records[
        valid_disc_records["Pfirrman grade"].isna()
    ]

    # -------------------------------------------------------------------------
    # Invalid Pfirrman grades
    # -------------------------------------------------------------------------

    invalid_grade_records = valid_disc_records[
        valid_disc_records["Pfirrman grade"].notna()
        & ~valid_disc_records["Pfirrman grade"].between(
            MIN_PFIRRMANN_GRADE,
            MAX_PFIRRMANN_GRADE,
        )
    ]

    # -------------------------------------------------------------------------
    # Orphan ground-truth records
    #
    # A valid disc record is considered orphaned if its patient does
    # not belong to the retained T2 patient cohort.
    # -------------------------------------------------------------------------

    orphan_records = valid_disc_records[
        ~valid_disc_records["Patient"].isin(
            retained_patient_ids
        )
    ]

    return {
        "invalid_ivd_count": len(
            invalid_ivd_records
        ),
        "missing_grade_count": len(
            missing_grade_records
        ),
        "invalid_grade_count": len(
            invalid_grade_records
        ),
        "orphan_record_count": len(
            orphan_records
        ),
        "valid_disc_records": len(
            valid_disc_records
        ),
    }


# =============================================================================
# Main
# =============================================================================


def main():
    """
    Run complete dataset validation.
    """

    logger = ExecutionLogger(
        "dataset_validation"
    )

    try:

        print("\n" + "=" * 60)
        print(
            Fore.CYAN
            + "DATASET VALIDATION"
        )
        print("=" * 60)

        # ---------------------------------------------------------------------
        # Get retained T2 files
        # ---------------------------------------------------------------------

        t2_files = get_retained_t2_files()

        retained_patient_ids = {
            get_patient_id(file_path)
            for file_path in t2_files
        }

        # ---------------------------------------------------------------------
        # Validate MRI / disc-mask filenames
        # ---------------------------------------------------------------------

        (
            filename_mismatch_count,
            missing_masks,
            extra_masks,
        ) = validate_image_mask_filenames(
            t2_files
        )

        # ---------------------------------------------------------------------
        # Validate MRI / disc-mask dimensions
        # ---------------------------------------------------------------------

        dimension_mismatches = validate_dimensions(
            t2_files
        )

        # ---------------------------------------------------------------------
        # Validate medial slice extraction
        # ---------------------------------------------------------------------

        medial_slice_failures = (
            validate_medial_slice_extraction(
                t2_files
            )
        )

        # ---------------------------------------------------------------------
        # Validate ground truth
        # ---------------------------------------------------------------------

        ground_truth = validate_ground_truth(
            t2_files
        )

        # ---------------------------------------------------------------------
        # Determine final validation status
        # ---------------------------------------------------------------------

        validation_passed = (
            len(t2_files) == 209
            and len(retained_patient_ids) == 209
            and filename_mismatch_count == 0
            and len(dimension_mismatches) == 0
            and len(medial_slice_failures) == 0
            and ground_truth["invalid_ivd_count"] == 0
            and ground_truth["missing_grade_count"] == 0
            and ground_truth["invalid_grade_count"] == 0
        )

        # ---------------------------------------------------------------------
        # Console output
        # ---------------------------------------------------------------------

        print(
            f"T2 Patients Checked              : "
            f"{len(retained_patient_ids)}"
        )

        print(
            f"Image Volumes                    : "
            f"{len(t2_files)}"
        )

        print(
            f"Disc Mask Volumes                : "
            f"{len(t2_files) - len(missing_masks)}"
        )

        print(
            f"Image/Mask Filename Mismatch     : "
            f"{filename_mismatch_count}"
        )

        print(
            f"Image/Mask Dimension Mismatch    : "
            f"{len(dimension_mismatches)}"
        )

        print(
            f"Invalid IVD Labels               : "
            f"{ground_truth['invalid_ivd_count']}"
        )

        print(
            f"Missing Pfirrman Grades          : "
            f"{ground_truth['missing_grade_count']}"
        )

        print(
            f"Invalid Pfirrman Grades          : "
            f"{ground_truth['invalid_grade_count']}"
        )

        print(
            f"Orphan GT Records                : "
            f"{ground_truth['orphan_record_count']}"
        )

        print(
            f"Medial Slice Extraction Failures : "
            f"{len(medial_slice_failures)}"
        )

        print(
            f"Valid Disc Records               : "
            f"{ground_truth['valid_disc_records']}"
        )

        status_text = (
            "PASSED"
            if validation_passed
            else "FAILED"
        )

        print(
            f"Validation Status                : "
            f"{status_text}"
        )

        print("=" * 60)

        # ---------------------------------------------------------------------
        # Print detailed failures
        # ---------------------------------------------------------------------

        if missing_masks:

            print("\nMissing Disc Masks:")

            for filename in sorted(
                missing_masks
            ):
                print(f"  {filename}")

        if extra_masks:

            print("\nExtra Disc Masks:")

            for filename in sorted(
                extra_masks
            ):
                print(f"  {filename}")

        if dimension_mismatches:

            print("\nDimension Mismatches:")

            for (
                filename,
                image_size,
                mask_size,
            ) in dimension_mismatches:

                print(
                    f"  {filename} | "
                    f"MRI: {image_size} | "
                    f"Disc Mask: {mask_size}"
                )

        if medial_slice_failures:

            print("\nMedial Slice Extraction Failures:")

            for (
                filename,
                reason,
            ) in medial_slice_failures:

                print(
                    f"  {filename} | {reason}"
                )

        # ---------------------------------------------------------------------
        # Execution log
        # ---------------------------------------------------------------------

        logger.add(
            "T2 Patients Checked",
            len(retained_patient_ids),
        )

        logger.add(
            "Image Volumes",
            len(t2_files),
        )

        logger.add(
            "Disc Mask Volumes",
            len(t2_files) - len(missing_masks),
        )

        logger.add(
            "Image/Mask Filename Mismatch",
            filename_mismatch_count,
        )

        logger.add(
            "Image/Mask Dimension Mismatch",
            len(dimension_mismatches),
        )

        logger.add(
            "Invalid IVD Labels",
            ground_truth["invalid_ivd_count"],
        )

        logger.add(
            "Missing Pfirrman Grades",
            ground_truth["missing_grade_count"],
        )

        logger.add(
            "Invalid Pfirrman Grades",
            ground_truth["invalid_grade_count"],
        )

        logger.add(
            "Orphan GT Records",
            ground_truth["orphan_record_count"],
        )

        logger.add(
            "Medial Slice Extraction Failures",
            len(medial_slice_failures),
        )

        logger.add(
            "Valid Disc Records",
            ground_truth["valid_disc_records"],
        )

        logger.add(
            "Validation Status",
            status_text,
        )

        logger.save(
            "SUCCESS"
            if validation_passed
            else "FAILED"
        )

        if not validation_passed:
            sys.exit(1)

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