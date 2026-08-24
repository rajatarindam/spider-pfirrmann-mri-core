import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# =============================================================================
# PROJECT ROOT
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# NEW FINAL DATASET PATH
# =============================================================================
# This constant will be added to configs/paths.py for the new experiment.
# The original training/dataset.py remains untouched.
from configs.paths import NEW_FINAL_DATASET_CSV


# =============================================================================
# PFIRRMANN DATASET - NEW FINAL EXPERIMENT
# =============================================================================
class PfirrmannDatasetNewFinal(Dataset):
    """
    Dataset for single-disc Pfirrmann grading using the NEW FINAL dataset.

    Each row in dataset_updated_pfirrmann.csv represents one disc sample.

    Input:
        1024 x 1024 uint8 grayscale PNG

    Output:
        image: [3, 1024, 1024] float32 tensor in [0, 1]
        label: integer 0-4
    """

    GRADE_TO_LABEL = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
    }

    LABEL_TO_GRADE = {
        0: 1,
        1: 2,
        2: 3,
        3: 4,
        4: 5,
    }

    def __init__(
        self,
        csv_path=NEW_FINAL_DATASET_CSV,
        split=None,
    ):
        super().__init__()

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found:\n{self.csv_path}"
            )

        self.df = pd.read_csv(self.csv_path)

        # ------------------------------------------------------------------
        # Required columns
        # ------------------------------------------------------------------
        required_columns = [
            "MRI_Output_Path",
            "Pfirrmann_Grade",
        ]

        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(
                    f"Required column '{column}' not found in dataset CSV."
                )

        # ------------------------------------------------------------------
        # Split filtering
        # ------------------------------------------------------------------
        if split is not None:

            if "Split" not in self.df.columns:
                raise ValueError(
                    "Split was requested, but the CSV does not contain "
                    "a 'Split' column."
                )

            split = split.lower()

            valid_splits = {"train", "val", "test"}

            if split not in valid_splits:
                raise ValueError(
                    f"Invalid split '{split}'. "
                    f"Expected one of {valid_splits}."
                )

            self.df = self.df[
                self.df["Split"].astype(str).str.lower() == split
            ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split: {split}"
            )

        # ------------------------------------------------------------------
        # Validate Pfirrmann grades
        # ------------------------------------------------------------------
        grades = pd.to_numeric(
            self.df["Pfirrmann_Grade"],
            errors="coerce"
        )

        if grades.isna().any():
            raise ValueError(
                "Dataset contains invalid Pfirrmann grades."
            )

        invalid_grades = sorted(
            set(grades.astype(int))
            - set(self.GRADE_TO_LABEL.keys())
        )

        if invalid_grades:
            raise ValueError(
                f"Invalid Pfirrmann grades found: {invalid_grades}"
            )

        self.df["Pfirrmann_Grade"] = grades.astype(int)

        print(
            f"PfirrmannDatasetNewFinal | "
            f"Split: {split or 'all':5s} | "
            f"Samples: {len(self.df)}"
        )

    # =========================================================================
    # DATASET LENGTH
    # =========================================================================
    def __len__(self):
        return len(self.df)

    # =========================================================================
    # LOAD IMAGE
    # =========================================================================
    def _load_image(self, relative_path):

        image_path = PROJECT_ROOT / str(relative_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        # ------------------------------------------------------------------
        # Load as grayscale
        # ------------------------------------------------------------------
        image = Image.open(image_path).convert("L")

        # ------------------------------------------------------------------
        # Validate resolution
        # ------------------------------------------------------------------
        if image.size != (1024, 1024):
            raise ValueError(
                f"Expected image size (1024, 1024), "
                f"got {image.size} for:\n{image_path}"
            )

        # ------------------------------------------------------------------
        # Convert to uint8 NumPy array
        # ------------------------------------------------------------------
        import numpy as np

        image = np.asarray(image, dtype=np.uint8)

        # ------------------------------------------------------------------
        # uint8 -> float32 [0, 1]
        # ------------------------------------------------------------------
        image = image.astype(np.float32) / 255.0

        # ------------------------------------------------------------------
        # Grayscale -> 3 channels
        # ------------------------------------------------------------------
        image = np.stack([image, image, image], axis=0)

        # ------------------------------------------------------------------
        # NumPy -> PyTorch
        # ------------------------------------------------------------------
        return torch.from_numpy(image)

    # =========================================================================
    # GET ONE SAMPLE
    # =========================================================================
    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = self._load_image(row["MRI_Output_Path"])

        grade = int(row["Pfirrmann_Grade"])

        label = self.GRADE_TO_LABEL[grade]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label