import sys
from pathlib import Path

import pandas as pd

# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import (
    MASTER_GT_CSV,
    DATASET_CSV,
)


# =============================================================================
# PATH CONVERSION
# =============================================================================

def convert_path(value):
    """
    Convert a project-local absolute Windows path into a
    project-relative POSIX path.

    Example:

    D:\\BAAA project\\spider-pfirrmann-mri-core\\data\\processed\\images\\1\\...
    
    becomes:

    data/processed/images/1/...
    """

    if pd.isna(value):
        return value

    value = str(value).strip()

    if not value:
        return value

    # Normalize Windows separators.
    normalized = value.replace("\\", "/")

    # Absolute Windows path.
    project_root = PROJECT_ROOT.as_posix().rstrip("/")

    if normalized.lower().startswith(
        project_root.lower() + "/"
    ):
        relative_path = normalized[
            len(project_root) + 1:
        ]

        return relative_path

    # Already relative.
    return normalized


# =============================================================================
# PROCESS ONE CSV
# =============================================================================

def process_csv(csv_path: Path):

    print("=" * 70)
    print(f"PROCESSING: {csv_path.name}")
    print("=" * 70)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    required_columns = [
        "MRI_Output_Path",
        "Mask_Output_Path",
    ]

    for column in required_columns:

        if column not in df.columns:
            raise ValueError(
                f"Required column '{column}' "
                f"not found in {csv_path.name}"
            )

    # -------------------------------------------------------------------------
    # Show example before conversion
    # -------------------------------------------------------------------------

    print()
    print("Before conversion:")

    for column in required_columns:

        print(f"{column}:")
        print(f"  {df[column].iloc[0]}")

    # -------------------------------------------------------------------------
    # Convert only the two path columns
    # -------------------------------------------------------------------------

    for column in required_columns:

        df[column] = df[column].apply(
            convert_path
        )

    # -------------------------------------------------------------------------
    # Verify conversion
    # -------------------------------------------------------------------------

    absolute_remaining = {}

    for column in required_columns:

        remaining = df[column].astype(str).str.match(
            r"^[A-Za-z]:/"
        ).sum()

        absolute_remaining[column] = remaining

    # -------------------------------------------------------------------------
    # Show example after conversion
    # -------------------------------------------------------------------------

    print()
    print("After conversion:")

    for column in required_columns:

        print(f"{column}:")
        print(f"  {df[column].iloc[0]}")

    print()
    print("Absolute paths remaining:")

    for column, count in absolute_remaining.items():

        print(
            f"  {column}: {count}"
        )

    # -------------------------------------------------------------------------
    # Safety check
    # -------------------------------------------------------------------------

    if any(
        count > 0
        for count in absolute_remaining.values()
    ):
        raise RuntimeError(
            f"Some absolute paths remain in {csv_path.name}. "
            f"CSV was NOT overwritten."
        )

    # -------------------------------------------------------------------------
    # Save directly to original file
    # -------------------------------------------------------------------------

    df.to_csv(
        csv_path,
        index=False
    )

    print()
    print(f"Saved directly to:")
    print(f"  {csv_path}")

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 70)
    print("CONVERT DATASET PATHS TO PROJECT-RELATIVE PATHS")
    print("=" * 70)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print()

    # -------------------------------------------------------------------------
    # Process original files directly.
    #
    # The *.absolute_backup.csv files remain untouched.
    # -------------------------------------------------------------------------

    process_csv(
        MASTER_GT_CSV
    )

    process_csv(
        DATASET_CSV
    )

    print("=" * 70)
    print("PATH CONVERSION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()