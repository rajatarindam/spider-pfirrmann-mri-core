"""
Model 4 Comprehensive Test Set Inference and Reporting Pipeline

Features:
    - Runs full test-set inference (731 single-slice disc crops) on the exact Model 4 architecture (PfirrmannClassifierV2).
    - Checkpoint Flexibility:
        * Defaults to 'best_model.pt' in outputs/model_4/checkpoints/ (or candidate dirs).
        * Can be switched to 'latest_checkpoint.pt' or any custom checkpoint path using --checkpoint.
    - Configurable batch size: Default is 6 (--batch_size 6).
    - Consolidated Output:
        All results and figures are automatically written into a dedicated 'results/' subfolder:
            results/
                prediction_results.csv              (19 columns with all probabilities and metrics)
                evaluation_summary.csv              (Overall and per-grade summary)
                test_predictions/                   (all 731 individual slice prediction montages)
                r/                                  (31 core evaluation files: 25 diagnostic PNGs + 6 summary CSVs)
                conf/                               (10 Grade 1-5 correct and wrong confidence bar charts)

Usage:
    # Run default test inference (best model, batch size 6, full visuals and reports):
    python evaluation/model_4/inference_model_4.py

    # Run using the latest checkpoint instead:
    python evaluation/model_4/inference_model_4.py --checkpoint latest_checkpoint.pt

    # Custom batch size:
    python evaluation/model_4/inference_model_4.py --batch_size 6

    # Fast evaluation without re-generating all 731 image montages:
    python evaluation/model_4/inference_model_4.py --skip_visuals
"""

import sys
import shutil
import argparse
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# =============================================================================
# PROJECT ROOT RESOLUTION
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent

candidates = [
    _SCRIPT_DIR.parent.parent,         # if run from <root>/evaluation/model_4/
    _SCRIPT_DIR.parent.parent.parent,  # if run from <root>/context/evaluation/model_4/
    Path.cwd(),
]
PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for c in candidates:
    if (c / "data").exists():
        PROJECT_ROOT = c
        break

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.paths import MRI_CORE_CHECKPOINT_PATH
from training.models.pfirrmann_classifier_v2 import PfirrmannClassifierV2

# =============================================================================
# DEFAULT SETTINGS & CONSTANTS
# =============================================================================

DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "new_final" / "final_data" / "dataset_OG.csv"
DATASET_CSV = DEFAULT_DATASET_CSV

GRADES = [1, 2, 3, 4, 5]
PROBABILITY_COLUMNS = [f"Prob_Grade_{g}" for g in GRADES]

REQUIRED_COLUMNS = [
    "Patient_ID",
    "Slice_Number",
    "Offset",
    "Medial_Slice_Number",
    "IVD_Label",
    "Pfirrmann_Grade",
    "Predicted_Grade",
    "Correct",
    "Error_Distance",
    "Prediction_Error_Type",
    "Prediction_Confidence",
    "GT_Grade_Probability",
    *PROBABILITY_COLUMNS,
]

GRADE_TO_LABEL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
LABEL_TO_GRADE = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

IMAGE_SIZE = (1024, 1024)
HEADER_HEIGHT = 270
OUTPUT_HEIGHT = IMAGE_SIZE[0] + HEADER_HEIGHT

# Global OUTPUT_DIR for core report figure saving
OUTPUT_DIR = _SCRIPT_DIR / "results" / "r"

class PfirrmannDatasetModel4(Dataset):
    """
    Exact Model 4 test dataset logic.

    CSV:
        data/new_final/final_data/dataset_OG.csv

    Image:
        1024 x 1024 grayscale PNG

    Output:
        image: [3, 1024, 1024] float32 tensor in [0, 1]
        label: integer class index 0-4
    """

    def __init__(self, csv_path=DATASET_CSV, split=None):
        super().__init__()

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found:\n{self.csv_path}"
            )

        self.df = pd.read_csv(self.csv_path)

        required_columns = [
            "MRI_Output_Path",
            "Pfirrmann_Grade",
            "Split",
        ]

        for column in required_columns:
            if column not in self.df.columns:
                raise ValueError(
                    f"Required column '{column}' not found in dataset CSV."
                )

        if split is not None:
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

        grades = pd.to_numeric(
            self.df["Pfirrmann_Grade"],
            errors="coerce",
        )

        if grades.isna().any():
            raise ValueError(
                "Dataset contains invalid Pfirrmann grades."
            )

        invalid_grades = sorted(
            set(grades.astype(int)) - set(GRADE_TO_LABEL.keys())
        )

        if invalid_grades:
            raise ValueError(
                f"Invalid Pfirrmann grades found: {invalid_grades}"
            )

        self.df["Pfirrmann_Grade"] = grades.astype(int)

        print(
            f"PfirrmannDatasetModel4 | "
            f"Split: {split or 'all':5s} | "
            f"Samples: {len(self.df)}"
        )

    def __len__(self):
        return len(self.df)

    def _load_image(self, relative_path):
        image_path = PROJECT_ROOT / str(relative_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(image_path).convert("L")

        if image.size != IMAGE_SIZE:
            raise ValueError(
                f"Expected image size {IMAGE_SIZE}, "
                f"got {image.size} for:\n{image_path}"
            )

        image = np.asarray(image, dtype=np.uint8)
        image = image.astype(np.float32) / 255.0

        image = np.stack(
            [image, image, image],
            axis=0,
        )

        return torch.from_numpy(image)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = self._load_image(
            row["MRI_Output_Path"]
        )

        grade = int(row["Pfirrmann_Grade"])
        label = GRADE_TO_LABEL[grade]

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return image, label


# =============================================================================
# DEVICE
# =============================================================================


# =============================================================================
# CHECKPOINT RESOLVER
# =============================================================================

def resolve_checkpoint_path(checkpoint_arg: str, project_root: Path) -> Path:
    """Find checkpoint file whether given as full path, relative path, or filename."""
    p = Path(checkpoint_arg)
    if p.is_file():
        return p.resolve()
    if (project_root / checkpoint_arg).is_file():
        return (project_root / checkpoint_arg).resolve()

    filename = p.name
    candidate_dirs = [
        project_root / "outputs" / "model_4" / "checkpoints",
        project_root / "outputs" / "model_4_28th_epoch_model",
        project_root / "outputs" / "model_4",
        _SCRIPT_DIR / "checkpoints",
    ]
    for d in candidate_dirs:
        cand = d / filename
        if cand.is_file():
            return cand.resolve()

    return (candidate_dirs[0] / filename).resolve()


# =============================================================================
# DEVICE & MODEL LOADING
# =============================================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return device
    print("WARNING: CUDA unavailable. Inference will use CPU.")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device):
    print()
    print("=" * 80)
    print("LOADING MODEL 4")
    print("=" * 80)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model 4 checkpoint not found:\n{checkpoint_path}")

    if not MRI_CORE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"MRI-CORE foundation checkpoint not found:\n{MRI_CORE_CHECKPOINT_PATH}"
        )

    print(f"Checkpoint:\n  {checkpoint_path}")

    model = PfirrmannClassifierV2()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print(f"Checkpoint epoch        : {checkpoint.get('epoch', 'N/A')}")
            print(f"Best validation Macro F1: {checkpoint.get('best_val_macro_f1', 'N/A')}")
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print("Model 4 loaded successfully.")
    print("=" * 80)
    return model

# =============================================================================
# TEST DATALOADER
# =============================================================================

def create_test_loader(csv_path: Path, batch_size: int = 6, num_workers: int = 4):
    test_dataset = PfirrmannDatasetModel4(csv_path=csv_path, split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_dataset, test_loader
def get_font(size):

    possible_fonts = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
    ]

    for font_name in possible_fonts:
        try:
            return ImageFont.truetype(
                font_name,
                size,
            )
        except OSError:
            continue

    return ImageFont.load_default()


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================

def get_error_type(gt_grade, predicted_grade):

    error_distance = abs(
        gt_grade - predicted_grade
    )

    if error_distance == 0:
        return "Correct"

    if error_distance == 1:
        return "Adjacent Error"

    return "Non-Adjacent Error"


# =============================================================================
# VISUAL FILENAMES
# =============================================================================

def create_visual_filenames(results):

    base_names = []

    for row in results:
        base_names.append(
            f"Patient{int(row['Patient_ID'])}"
            f"_Slice{int(row['Slice_Number'])}"
            f"_GT{int(row['Pfirrmann_Grade'])}"
            f"_Pred{int(row['Predicted_Grade'])}.png"
        )

    total_counts = Counter(base_names)
    occurrence_counts = Counter()

    filenames = []

    for base_name in base_names:
        occurrence_counts[base_name] += 1

        if total_counts[base_name] > 1:
            stem = Path(base_name).stem

            filename = (
                f"{stem}_Idx"
                f"{occurrence_counts[base_name]}"
                f".png"
            )
        else:
            filename = base_name

        filenames.append(filename)

    return filenames


# =============================================================================
# CREATE VISUAL
# =============================================================================

def create_prediction_visual(
    image_path,
    patient_id,
    slice_number,
    offset,
    medial_slice,
    ivd_label,
    gt_grade,
    predicted_grade,
    confidence,
    gt_probability,
    error_distance,
    error_type,
    probabilities,
    output_path,
):

    image = Image.open(
        image_path
    ).convert("L")

    if image.size != IMAGE_SIZE:
        raise ValueError(
            f"Expected 1024 x 1024 image, "
            f"got {image.size}:\n"
            f"{image_path}"
        )

    image = image.convert("RGB")

    width, height = image.size

    canvas = Image.new(
        "RGB",
        (
            width,
            OUTPUT_HEIGHT,
        ),
        "white",
    )

    canvas.paste(
        image,
        (
            0,
            HEADER_HEIGHT,
        ),
    )

    draw = ImageDraw.Draw(canvas)

    font_title = get_font(22)
    font_main = get_font(19)
    font_small = get_font(18)

    draw.text(
        (15, 12),
        "Model 4 Test Prediction",
        fill="black",
        font=font_title,
    )

    left_x = 15

    draw.text(
        (left_x, 52),
        f"Patient ID: {patient_id}",
        fill="black",
        font=font_main,
    )

    draw.text(
        (left_x, 84),
        f"IVD Label: {ivd_label}",
        fill="black",
        font=font_main,
    )

    draw.text(
        (left_x, 116),
        f"Slice Number: {slice_number}",
        fill="black",
        font=font_main,
    )

    draw.text(
        (left_x, 148),
        f"Offset: {offset:+d}",
        fill="black",
        font=font_main,
    )

    draw.text(
        (left_x, 180),
        f"Medial Slice: {medial_slice}",
        fill="black",
        font=font_main,
    )

    middle_x = 365

    draw.text(
        (middle_x, 52),
        f"GT: {gt_grade}",
        fill="black",
        font=font_main,
    )

    draw.text(
        (middle_x + 125, 52),
        f"Predicted: {predicted_grade}",
        fill="black",
        font=font_main,
    )

    draw.text(
        (middle_x, 88),
        f"Confidence: {confidence:.4f}",
        fill="black",
        font=font_main,
    )

    draw.text(
        (middle_x, 124),
        f"GT Probability: {gt_probability:.4f}",
        fill="black",
        font=font_main,
    )

    draw.text(
        (middle_x, 160),
        f"Error Distance: {error_distance}",
        fill="black",
        font=font_main,
    )

    draw.text(
        (middle_x, 196),
        f"Type: {error_type}",
        fill="black",
        font=font_small,
    )

    probability_x = 690

    draw.text(
        (probability_x, 42),
        "Grade Probabilities",
        fill="black",
        font=font_main,
    )

    probability_y = [
        76,
        108,
        140,
        172,
        204,
    ]

    for grade, y in zip(
        GRADES,
        probability_y,
    ):
        probability = float(
            probabilities[grade - 1]
        )

        draw.text(
            (probability_x, y),
            f"P(G{grade}) = {probability:.4f}",
            fill="black",
            font=font_small,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        output_path
    )


# =============================================================================
# CLEAR OLD OUTPUTS
# =============================================================================

@torch.inference_mode()
def run_inference(
    model,
    test_dataset,
    test_loader,
    device,
):

    print()
    print("=" * 80)
    print("RUNNING MODEL 4 TEST INFERENCE")
    print("=" * 80)

    results = []

    processed = 0
    total = len(test_dataset)

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predicted_labels = torch.argmax(
            probabilities,
            dim=1,
        )

        prediction_confidences = torch.max(
            probabilities,
            dim=1,
        ).values

        probabilities = (
            probabilities
            .cpu()
            .numpy()
        )

        predicted_labels = (
            predicted_labels
            .cpu()
            .numpy()
        )

        prediction_confidences = (
            prediction_confidences
            .cpu()
            .numpy()
        )

        batch_size = len(labels)

        batch_start = processed
        batch_end = processed + batch_size

        # shuffle=False means this dataframe slice corresponds
        # exactly to the current inference batch.
        batch_df = test_dataset.df.iloc[
            batch_start:batch_end
        ]

        for local_idx, (_, row) in enumerate(
            batch_df.iterrows()
        ):

            # GT comes directly from the original CSV.
            gt_grade = int(
                row["Pfirrmann_Grade"]
            )

            # Model output label 0-4 -> displayed grade 1-5.
            predicted_grade = (
                int(
                    predicted_labels[local_idx]
                )
                + 1
            )

            confidence = float(
                prediction_confidences[local_idx]
            )

            sample_probabilities = (
                probabilities[local_idx]
            )

            correct = (
                gt_grade == predicted_grade
            )

            error_distance = abs(
                gt_grade - predicted_grade
            )

            error_type = get_error_type(
                gt_grade,
                predicted_grade,
            )

            gt_grade_probability = float(
                sample_probabilities[
                    gt_grade - 1
                ]
            )

            # Probability/prediction consistency check.
            probability_argmax = (
                int(
                    np.argmax(
                        sample_probabilities
                    )
                )
                + 1
            )

            if probability_argmax != predicted_grade:
                raise RuntimeError(
                    "Prediction/probability mismatch for "
                    f"Patient {row['Patient_ID']}, "
                    f"Slice {row['Slice_Number']}, "
                    f"IVD {row['IVD_Label']}."
                )

            relative_image_path = str(
                row["MRI_Output_Path"]
            )

            absolute_image_path = (
                PROJECT_ROOT
                / relative_image_path
            )

            if not absolute_image_path.exists():
                raise FileNotFoundError(
                    f"MRI image not found:\n"
                    f"{absolute_image_path}"
                )

            result = {
                "Patient_ID": int(row["Patient_ID"]),
                "Slice_Number": int(row["Slice_Number"]),
                "Offset": int(row["Offset"]),
                "Medial_Slice_Number": int(
                    row["Medial_Slice_Number"]
                ),
                "IVD_Label": int(row["IVD_Label"]),
                "Pfirrmann_Grade": gt_grade,
                "Predicted_Grade": predicted_grade,
                "Correct": correct,
                "Error_Distance": error_distance,
                "Prediction_Error_Type": error_type,
                "Prediction_Confidence": confidence,
                "GT_Grade_Probability": gt_grade_probability,
                "Prob_Grade_1": float(sample_probabilities[0]),
                "Prob_Grade_2": float(sample_probabilities[1]),
                "Prob_Grade_3": float(sample_probabilities[2]),
                "Prob_Grade_4": float(sample_probabilities[3]),
                "Prob_Grade_5": float(sample_probabilities[4]),
                "MRI_Output_Path": relative_image_path,
            }

            results.append(result)

        processed = batch_end

        print(f"Processed {processed}/{total} ({processed / total * 100:.1f}%)", flush=True)

    filenames = create_visual_filenames(
        results
    )

    for result, filename in zip(results, filenames):
        result["Prediction_Image"] = filename

    return pd.DataFrame(results)


# =============================================================================
# SUMMARY
# =============================================================================

def create_summary(results_df):

    y_true = (
        results_df["Pfirrmann_Grade"]
        .astype(int)
        .to_numpy()
    )

    y_pred = (
        results_df["Predicted_Grade"]
        .astype(int)
        .to_numpy()
    )

    total = len(results_df)

    accuracy = float(
        np.mean(y_true == y_pred)
    )

    per_grade_rows = []
    f1_values = []

    for grade in GRADES:

        tp = np.sum(
            (y_true == grade)
            & (y_pred == grade)
        )

        fp = np.sum(
            (y_true != grade)
            & (y_pred == grade)
        )

        fn = np.sum(
            (y_true == grade)
            & (y_pred != grade)
        )

        support = np.sum(
            y_true == grade
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        f1_values.append(f1)

        grade_mask = y_true == grade

        per_grade_rows.append(
            {
                "Grade": grade,
                "GT_Samples": int(support),
                "Predicted_Samples": int(
                    np.sum(y_pred == grade)
                ),
                "Precision": float(precision),
                "Recall": float(recall),
                "F1": float(f1),
                "Mean_Confidence": float(
                    results_df.loc[
                        grade_mask,
                        "Prediction_Confidence",
                    ].mean()
                ) if support > 0 else np.nan,
                "Mean_GT_Probability": float(
                    results_df.loc[
                        grade_mask,
                        "GT_Grade_Probability",
                    ].mean()
                ) if support > 0 else np.nan,
            }
        )

    macro_f1 = float(
        np.mean(f1_values)
    )

    correct_count = int(
        np.sum(y_true == y_pred)
    )

    incorrect_count = (
        total - correct_count
    )

    adjacent_count = int(
        np.sum(
            results_df["Error_Distance"].to_numpy()
            == 1
        )
    )

    non_adjacent_count = int(
        np.sum(
            results_df["Error_Distance"].to_numpy()
            > 1
        )
    )

    overall = pd.DataFrame(
        [
            {
                "Section": "Overall",
                "Grade": "All",
                "Test_Samples": total,
                "Test_Patients": int(
                    results_df["Patient_ID"].nunique()
                ),
                "GT_Samples": total,
                "Predicted_Samples": total,
                "Correct_Samples": correct_count,
                "Incorrect_Samples": incorrect_count,
                "Accuracy": accuracy,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": macro_f1,
                "Macro_F1": macro_f1,
                "Mean_Confidence": float(
                    results_df[
                        "Prediction_Confidence"
                    ].mean()
                ),
                "Mean_GT_Probability": float(
                    results_df[
                        "GT_Grade_Probability"
                    ].mean()
                ),
                "Mean_Error_Distance": float(
                    results_df[
                        "Error_Distance"
                    ].mean()
                ),
                "Correct_Count": correct_count,
                "Adjacent_Error_Count": adjacent_count,
                "Non_Adjacent_Error_Count": (
                    non_adjacent_count
                ),
            }
        ]
    )

    per_grade_df = pd.DataFrame(
        [
            {
                "Section": "Per Grade",
                "Grade": row["Grade"],
                "Test_Samples": total,
                "Test_Patients": np.nan,
                "GT_Samples": row["GT_Samples"],
                "Predicted_Samples": row["Predicted_Samples"],
                "Correct_Samples": np.nan,
                "Incorrect_Samples": np.nan,
                "Accuracy": np.nan,
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1": row["F1"],
                "Macro_F1": macro_f1,
                "Mean_Confidence": row[
                    "Mean_Confidence"
                ],
                "Mean_GT_Probability": row[
                    "Mean_GT_Probability"
                ],
                "Mean_Error_Distance": np.nan,
                "Correct_Count": np.nan,
                "Adjacent_Error_Count": np.nan,
                "Non_Adjacent_Error_Count": np.nan,
            }
            for row in per_grade_rows
        ]
    )

    return pd.concat(
        [
            overall,
            per_grade_df,
        ],
        ignore_index=True,
    )


# =============================================================================
# VALIDATION
# =============================================================================

def validate_results(results_df):

    print()
    print("=" * 80)
    print("VALIDATING MODEL 4 INFERENCE RESULTS")
    print("=" * 80)

    if len(results_df) == 0:
        raise ValueError(
            "Inference produced zero results."
        )

    unique_gt = sorted(
        results_df[
            "Pfirrmann_Grade"
        ].unique()
    )

    print(
        f"GT grades found: {unique_gt}"
    )

    if unique_gt != GRADES:
        raise ValueError(
            "Unexpected GT grades found. "
            f"Expected {GRADES}, got {unique_gt}"
        )

    unique_pred = sorted(
        results_df[
            "Predicted_Grade"
        ].unique()
    )

    print(
        f"Predicted grades found: {unique_pred}"
    )

    probability_sums = results_df[
        PROBABILITY_COLUMNS
    ].sum(axis=1)

    max_probability_error = float(
        np.max(
            np.abs(
                probability_sums - 1.0
            )
        )
    )

    print(
        f"Maximum probability sum error: "
        f"{max_probability_error:.8f}"
    )

    if max_probability_error > 1e-4:
        raise ValueError(
            "Probability columns do not sum to 1."
        )

    probability_argmax = (
        results_df[
            PROBABILITY_COLUMNS
        ]
        .to_numpy()
        .argmax(axis=1)
        + 1
    )

    mismatch_count = int(
        np.sum(
            probability_argmax
            != results_df[
                "Predicted_Grade"
            ].to_numpy()
        )
    )

    print(
        f"Prediction/probability mismatches: "
        f"{mismatch_count}"
    )

    if mismatch_count:
        raise ValueError(
            "Predicted_Grade does not match "
            "the probability argmax."
        )

    print("Validation passed.")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# CORE EVALUATION REPORT BATTERY (25 FIGURES + 6 CSV TABLES)
# =============================================================================
def save_figure(fig, filename):
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def validate_input(df):
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "prediction_results.csv is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    if len(df) == 0:
        raise ValueError(
            "prediction_results.csv contains no rows."
        )


# =============================================================================
# 1. CONFUSION MATRIX
# =============================================================================

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=GRADES,
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted Pfirrmann Grade",
        ylabel="Ground Truth Pfirrmann Grade",
        xticks=np.arange(5),
        yticks=np.arange(5),
        xticklabels=GRADES,
        yticklabels=GRADES,
    )

    threshold = cm.max() / 2.0

    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
            )

    save_figure(
        fig,
        "01_confusion_matrix.png",
    )


# =============================================================================
# 2. NORMALIZED CONFUSION MATRIX
# =============================================================================

def plot_normalized_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=GRADES,
    )

    row_sums = cm.sum(
        axis=1,
        keepdims=True,
    )

    normalized = np.divide(
        cm,
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=float,
        ),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        normalized,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        title="Normalized Confusion Matrix",
        xlabel="Predicted Pfirrmann Grade",
        ylabel="Ground Truth Pfirrmann Grade",
        xticks=np.arange(5),
        yticks=np.arange(5),
        xticklabels=GRADES,
        yticklabels=GRADES,
    )

    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{normalized[i, j] * 100:.1f}%",
                ha="center",
                va="center",
            )

    save_figure(
        fig,
        "02_normalized_confusion_matrix.png",
    )


# =============================================================================
# 3. GT VS PREDICTED DISTRIBUTION
# =============================================================================

def plot_grade_distribution(df):

    gt_counts = (
        df["Pfirrmann_Grade"]
        .value_counts()
        .reindex(GRADES, fill_value=0)
    )

    pred_counts = (
        df["Predicted_Grade"]
        .value_counts()
        .reindex(GRADES, fill_value=0)
    )

    x = np.arange(5)
    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        gt_counts.values,
        width,
        label="Ground Truth",
    )

    ax.bar(
        x + width / 2,
        pred_counts.values,
        width,
        label="Predicted",
    )

    ax.set(
        title="Ground Truth vs Predicted Grade Distribution",
        xlabel="Pfirrmann Grade",
        ylabel="Number of Samples",
        xticks=x,
        xticklabels=GRADES,
    )

    ax.legend()

    save_figure(
        fig,
        "03_gt_vs_predicted_distribution.png",
    )


# =============================================================================
# 4. PER-GRADE PRECISION / RECALL / F1
# =============================================================================

def calculate_per_grade_metrics(y_true, y_pred):

    precision = precision_score(
        y_true,
        y_pred,
        labels=GRADES,
        average=None,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=GRADES,
        average=None,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=GRADES,
        average=None,
        zero_division=0,
    )

    support = (
        pd.Series(y_true)
        .value_counts()
        .reindex(GRADES, fill_value=0)
        .values
    )

    return pd.DataFrame(
        {
            "Grade": GRADES,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support,
        }
    )


def plot_per_grade_metrics(per_grade):

    x = np.arange(5)
    width = 0.25

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width,
        per_grade["Precision"],
        width,
        label="Precision",
    )

    ax.bar(
        x,
        per_grade["Recall"],
        width,
        label="Recall",
    )

    ax.bar(
        x + width,
        per_grade["F1"],
        width,
        label="F1",
    )

    ax.set(
        title="Per-Grade Precision, Recall and F1",
        xlabel="Pfirrmann Grade",
        ylabel="Score",
        xticks=x,
        xticklabels=GRADES,
        ylim=(0, 1),
    )

    ax.legend()

    save_figure(
        fig,
        "04_per_grade_precision_recall_f1.png",
    )


# =============================================================================
# 5. CORRECT VS INCORRECT
# =============================================================================

def plot_correct_incorrect(df):

    correct = int(
        df["Correct"].sum()
    )

    incorrect = (
        len(df) - correct
    )

    labels = [
        "Correct",
        "Incorrect",
    ]

    values = [
        correct,
        incorrect,
    ]

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set(
        title="Correct vs Incorrect Predictions",
        ylabel="Number of Samples",
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n"
            f"({value / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
        )

    save_figure(
        fig,
        "05_correct_vs_incorrect.png",
    )


# =============================================================================
# 6. ERROR DISTANCE
# =============================================================================

def plot_error_distance(df):

    counts = (
        df["Error_Distance"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    bars = ax.bar(
        counts.index.astype(str),
        counts.values,
    )

    ax.set(
        title="Error Distance Distribution",
        xlabel="|Ground Truth - Prediction|",
        ylabel="Number of Samples",
    )

    for bar, value in zip(
        bars,
        counts.values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            str(value),
            ha="center",
            va="bottom",
        )

    save_figure(
        fig,
        "06_error_distance_distribution.png",
    )


# =============================================================================
# 7. CORRECT / ADJACENT / NON-ADJACENT
# =============================================================================

def plot_error_types(df):

    order = [
        "Correct",
        "Adjacent Error",
        "Non-Adjacent Error",
    ]

    counts = (
        df["Prediction_Error_Type"]
        .value_counts()
        .reindex(
            order,
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        counts.index,
        counts.values,
    )

    ax.set(
        title="Prediction Error Type",
        ylabel="Number of Samples",
    )

    for bar, value in zip(
        bars,
        counts.values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n"
            f"({value / len(df) * 100:.1f}%)",
            ha="center",
            va="bottom",
        )

    save_figure(
        fig,
        "07_error_type_distribution.png",
    )


# =============================================================================
# 8. CONFIDENCE DISTRIBUTION
# =============================================================================

def plot_confidence_distribution(df):

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.hist(
        df["Prediction_Confidence"],
        bins=20,
    )

    ax.set(
        title="Prediction Confidence Distribution",
        xlabel="Prediction Confidence",
        ylabel="Number of Samples",
        xlim=(0, 1),
    )

    save_figure(
        fig,
        "08_confidence_distribution.png",
    )


# =============================================================================
# 9. CONFIDENCE: CORRECT VS INCORRECT
# =============================================================================

def plot_confidence_correct_vs_incorrect(df):

    correct = df.loc[
        df["Correct"],
        "Prediction_Confidence",
    ]

    incorrect = df.loc[
        ~df["Correct"],
        "Prediction_Confidence",
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.boxplot(
        [correct, incorrect],
        tick_labels=[
            "Correct",
            "Incorrect",
        ],
        showmeans=True,
    )

    ax.set(
        title="Prediction Confidence: Correct vs Incorrect",
        ylabel="Prediction Confidence",
        ylim=(0, 1),
    )

    save_figure(
        fig,
        "09_confidence_correct_vs_incorrect.png",
    )


# =============================================================================
# 10. GRADE-WISE CONFIDENCE
# =============================================================================

def plot_grade_wise_confidence(df):

    data = [
        df.loc[
            df["Pfirrmann_Grade"] == grade,
            "Prediction_Confidence",
        ].values
        for grade in GRADES
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.boxplot(
        data,
        tick_labels=[
            str(grade)
            for grade in GRADES
        ],
        showmeans=True,
    )

    ax.set(
        title="Prediction Confidence by Ground Truth Grade",
        xlabel="Ground Truth Pfirrmann Grade",
        ylabel="Prediction Confidence",
        ylim=(0, 1),
    )

    save_figure(
        fig,
        "10_grade_wise_confidence.png",
    )


# =============================================================================
# 11. MEAN CLASS PROBABILITIES BY GT GRADE
# =============================================================================

def plot_mean_class_probabilities(df):

    grouped = (
        df.groupby(
            "Pfirrmann_Grade"
        )[PROBABILITY_COLUMNS]
        .mean()
        .reindex(GRADES)
    )

    matrix = grouped.values

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(
        image,
        ax=ax,
    )

    ax.set(
        title="Mean Predicted Class Probabilities by Ground Truth Grade",
        xlabel="Predicted Grade",
        ylabel="Ground Truth Grade",
        xticks=np.arange(5),
        yticks=np.arange(5),
        xticklabels=GRADES,
        yticklabels=GRADES,
    )

    for i in range(5):
        for j in range(5):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.3f}",
                ha="center",
                va="center",
            )

    save_figure(
        fig,
        "11_mean_class_probabilities_by_gt.png",
    )


# =============================================================================
# 12. CALIBRATION / RELIABILITY
# =============================================================================

def plot_calibration_reliability(df):

    confidence = (
        df["Prediction_Confidence"]
        .to_numpy()
    )

    correct = (
        df["Correct"]
        .astype(int)
        .to_numpy()
    )

    bins = np.linspace(
        0,
        1,
        11,
    )

    mean_confidence = []
    observed_accuracy = []
    sample_count = []

    for lower, upper in zip(
        bins[:-1],
        bins[1:],
    ):

        if upper == 1:
            mask = (
                (confidence >= lower)
                &
                (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                &
                (confidence < upper)
            )

        if not mask.any():
            continue

        mean_confidence.append(
            confidence[mask].mean()
        )

        observed_accuracy.append(
            correct[mask].mean()
        )

        sample_count.append(
            int(mask.sum())
        )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    ax.plot(
        mean_confidence,
        observed_accuracy,
        marker="o",
        label="Model",
    )

    ax.set(
        title="Calibration / Reliability Plot",
        xlabel="Mean Predicted Confidence",
        ylabel="Observed Accuracy",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax.legend()

    save_figure(
        fig,
        "12_calibration_reliability.png",
    )

    pd.DataFrame(
        {
            "Mean_Confidence": mean_confidence,
            "Observed_Accuracy": observed_accuracy,
            "Sample_Count": sample_count,
        }
    ).to_csv(
        OUTPUT_DIR / "calibration_bins.csv",
        index=False,
    )


# =============================================================================
# 13. PATIENT-LEVEL ACCURACY
# =============================================================================

def calculate_patient_metrics(df):

    rows = []

    for patient_id, group in df.groupby(
        "Patient_ID",
        sort=True,
    ):

        y_true = group[
            "Pfirrmann_Grade"
        ]

        y_pred = group[
            "Predicted_Grade"
        ]

        rows.append(
            {
                "Patient_ID": patient_id,
                "Test_Samples": len(group),
                "Correct": int(
                    group["Correct"].sum()
                ),
                "Accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),
                "Macro_F1": f1_score(
                    y_true,
                    y_pred,
                    labels=GRADES,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_patient_accuracy(patient_metrics):

    sorted_df = patient_metrics.sort_values(
        "Accuracy"
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.bar(
        sorted_df["Patient_ID"].astype(str),
        sorted_df["Accuracy"],
    )

    ax.set(
        title="Patient-Level Test Accuracy",
        xlabel="Patient ID",
        ylabel="Accuracy",
        ylim=(0, 1),
    )

    ax.tick_params(
        axis="x",
        rotation=90,
    )

    save_figure(
        fig,
        "13_patient_level_accuracy.png",
    )


# =============================================================================
# 14. OFFSET-WISE PERFORMANCE
# =============================================================================

def calculate_offset_metrics(df):

    rows = []

    for offset, group in df.groupby(
        "Offset",
        sort=True,
    ):

        y_true = group[
            "Pfirrmann_Grade"
        ]

        y_pred = group[
            "Predicted_Grade"
        ]

        rows.append(
            {
                "Offset": offset,
                "Test_Samples": len(group),
                "Accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),
                "Macro_F1": f1_score(
                    y_true,
                    y_pred,
                    labels=GRADES,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_offset_performance(offset_metrics):

    x = np.arange(
        len(offset_metrics)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        offset_metrics["Accuracy"],
        width,
        label="Accuracy",
    )

    ax.bar(
        x + width / 2,
        offset_metrics["Macro_F1"],
        width,
        label="Macro F1",
    )

    ax.set(
        title="Performance by Slice Offset",
        xlabel="Offset from Medial Slice",
        ylabel="Score",
        xticks=x,
        xticklabels=[
            str(offset)
            for offset in offset_metrics["Offset"]
        ],
        ylim=(0, 1),
    )

    ax.legend()

    save_figure(
        fig,
        "14_offset_wise_performance.png",
    )


# =============================================================================
# 15. ERROR-PAIR DISTRIBUTION
# =============================================================================

def calculate_error_pairs(df):

    errors = df.loc[
        ~df["Correct"]
    ]

    if len(errors) == 0:
        return pd.DataFrame(
            columns=[
                "GT_Grade",
                "Predicted_Grade",
                "Error_Count",
            ]
        )

    pairs = (
        errors.groupby(
            [
                "Pfirrmann_Grade",
                "Predicted_Grade",
            ]
        )
        .size()
        .reset_index(
            name="Error_Count"
        )
        .sort_values(
            "Error_Count",
            ascending=False,
        )
    )

    return pairs.rename(
        columns={
            "Pfirrmann_Grade": "GT_Grade",
        }
    )


def plot_error_pairs(error_pairs):

    if len(error_pairs) == 0:

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.text(
            0.5,
            0.5,
            "No incorrect predictions",
            ha="center",
            va="center",
        )

        ax.set_axis_off()

        save_figure(
            fig,
            "15_error_pair_distribution.png",
        )

        return

    labels = [
        f"G{gt} → G{pred}"
        for gt, pred in zip(
            error_pairs["GT_Grade"],
            error_pairs["Predicted_Grade"],
        )
    ]

    counts = (
        error_pairs["Error_Count"]
        .to_numpy()
    )

    labels = labels[::-1]
    counts = counts[::-1]

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    bars = ax.barh(
        labels,
        counts,
    )

    ax.set(
        title="Ground Truth → Predicted Error Pairs",
        xlabel="Number of Errors",
        ylabel="Error Pair",
    )

    for bar, value in zip(
        bars,
        counts,
    ):
        ax.text(
            bar.get_width(),
            bar.get_y()
            + bar.get_height() / 2,
            f" {value}",
            va="center",
        )

    save_figure(
        fig,
        "15_error_pair_distribution.png",
    )



# =============================================================================
# 16-25. GRADE-WISE PATIENT CONFIDENCE: CORRECT / INCORRECT
# =============================================================================
#
# For every ground-truth grade, two separate figures are generated:
#   - Correct predictions: confidence of every correct test case
#   - Incorrect predictions: confidence of every incorrect test case
#
# Each point represents one test sample. Points are grouped by Patient_ID.
# Mean and median confidence are written in the figure and shown as horizontal
# reference lines.
#
# Output files:
#   16_grade_1_correct_confidence_per_patient.png
#   17_grade_1_incorrect_confidence_per_patient.png
#   ...
#   24_grade_5_correct_confidence_per_patient.png
#   25_grade_5_incorrect_confidence_per_patient.png
#
# =============================================================================

def plot_grade_confidence_per_patient(
    df,
    grade,
    correct=True,
    filename=None,
):
    """
    Plot prediction confidence for one GT grade, split by correctness.

    X-axis:
        Patient ID

    Y-axis:
        Prediction confidence

    Each point:
        One test sample belonging to the selected GT grade.

    The figure explicitly reports:
        Mean confidence
        Median confidence
        Number of samples
        Number of patients
    """

    subset = df[
        (df["Pfirrmann_Grade"] == grade)
        & (df["Correct"] == correct)
    ].copy()

    status = "Correct" if correct else "Incorrect"

    fig, ax = plt.subplots(figsize=(12, 7))

    if len(subset) == 0:
        ax.text(
            0.5,
            0.5,
            f"No {status.lower()} predictions for Grade {grade}",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )

        ax.set(
            title=(
                f"Grade {grade} - {status} Prediction Confidence "
                f"per Patient"
            ),
            xlabel="Patient ID",
            ylabel="Prediction Confidence",
            ylim=(0, 1),
        )

        save_figure(fig, filename)
        return

    # Keep Patient IDs in sorted order where possible.
    patient_ids = sorted(
        subset["Patient_ID"].unique(),
        key=lambda value: str(value),
    )

    patient_to_x = {
        patient_id: index
        for index, patient_id in enumerate(patient_ids)
    }

    # Small deterministic horizontal jitter prevents overlapping points from
    # the same patient from hiding each other.
    rng = np.random.default_rng(42 + grade + (0 if correct else 100))

    x_values = np.array(
        [
            patient_to_x[patient_id]
            for patient_id in subset["Patient_ID"]
        ],
        dtype=float,
    )

    if len(subset) > 1:
        x_values += rng.uniform(
            -0.18,
            0.18,
            size=len(subset),
        )

    confidence = subset[
        "Prediction_Confidence"
    ].to_numpy(dtype=float)

    mean_confidence = float(np.mean(confidence))
    median_confidence = float(np.median(confidence))

    ax.scatter(
        x_values,
        confidence,
        alpha=0.75,
        s=42,
    )

    ax.axhline(
        mean_confidence,
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_confidence:.3f}",
    )

    ax.axhline(
        median_confidence,
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {median_confidence:.3f}",
    )

    ax.set(
        title=(
            f"Grade {grade} - {status} Prediction Confidence "
            f"per Patient"
        ),
        xlabel="Patient ID",
        ylabel="Prediction Confidence",
        ylim=(0, 1),
        xticks=np.arange(len(patient_ids)),
        xticklabels=[str(pid) for pid in patient_ids],
    )

    ax.tick_params(
        axis="x",
        rotation=90,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        loc="best",
    )

    ax.text(
        0.02,
        0.97,
        (
            f"Samples: {len(subset)}\n"
            f"Patients: {subset['Patient_ID'].nunique()}\n"
            f"Mean confidence: {mean_confidence:.3f}\n"
            f"Median confidence: {median_confidence:.3f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.4",
            alpha=0.85,
        ),
    )

    save_figure(fig, filename)


def generate_grade_confidence_per_patient_reports(df):
    """
    Generate 10 figures:
        5 grades x 2 correctness categories.
    """

    file_index = 16

    for grade in GRADES:
        correct_filename = (
            f"{file_index:02d}_grade_{grade}_"
            "correct_confidence_per_patient.png"
        )

        print(
            f"[{file_index}/25] "
            f"Grade {grade} correct confidence per patient"
        )

        plot_grade_confidence_per_patient(
            df=df,
            grade=grade,
            correct=True,
            filename=correct_filename,
        )

        file_index += 1

        incorrect_filename = (
            f"{file_index:02d}_grade_{grade}_"
            "incorrect_confidence_per_patient.png"
        )

        print(
            f"[{file_index}/25] "
            f"Grade {grade} incorrect confidence per patient"
        )

        plot_grade_confidence_per_patient(
            df=df,
            grade=grade,
            correct=False,
            filename=incorrect_filename,
        )

        file_index += 1


# =============================================================================
# SUMMARY TABLES
# =============================================================================

def create_overall_metrics(df):

    y_true = df[
        "Pfirrmann_Grade"
    ]

    y_pred = df[
        "Predicted_Grade"
    ]

    return pd.DataFrame(
        [
            {
                "Test_Samples": len(df),
                "Test_Patients": df[
                    "Patient_ID"
                ].nunique(),
                "Accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),
                "Macro_Precision": precision_score(
                    y_true,
                    y_pred,
                    labels=GRADES,
                    average="macro",
                    zero_division=0,
                ),
                "Weighted_Precision": precision_score(
                    y_true,
                    y_pred,
                    labels=GRADES,
                    average="weighted",
                    zero_division=0,
                ),
                "Macro_Recall": recall_score(
                    y_true,
                    y_pred,
                    labels=GRADES,
                    average="macro",
                    zero_division=0,
                ),
                "Weighted_Recall": recall_score(
                    y_true,
                    y_pred,
                    labels=GRADES,
                    average="weighted",
                    zero_division=0,
                ),
                "Macro_F1": f1_score(
                    y_true,
                    y_pred,
                    labels=GRADES,
                    average="macro",
                    zero_division=0,
                ),
                "Weighted_F1": f1_score(
                    y_true,
                    y_pred,
                    labels=GRADES,
                    average="weighted",
                    zero_division=0,
                ),
                "Correct_Samples": int(
                    df["Correct"].sum()
                ),
                "Incorrect_Samples": int(
                    (~df["Correct"]).sum()
                ),
                "Mean_Confidence": df[
                    "Prediction_Confidence"
                ].mean(),
                "Mean_GT_Grade_Probability": df[
                    "GT_Grade_Probability"
                ].mean(),
                "Mean_Error_Distance": df[
                    "Error_Distance"
                ].mean(),
            }
        ]
    )


# =============================================================================
# MAIN
# =============================================================================


def run_core_evaluation_report(df: pd.DataFrame, r_dir: Path):
    global OUTPUT_DIR
    OUTPUT_DIR = r_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print("GENERATING 25 CORE EVALUATION VISUALIZATIONS & 6 CSV REPORTS")
    print(f"Target folder: {OUTPUT_DIR}")
    print("=" * 80)

    validate_input(df)

    y_true = df["Pfirrmann_Grade"].to_numpy(dtype=int)
    y_pred = df["Predicted_Grade"].to_numpy(dtype=int)

    print("\n[1/15] Confusion matrix")
    plot_confusion_matrix(y_true, y_pred)

    print("[2/15] Normalized confusion matrix")
    plot_normalized_confusion_matrix(y_true, y_pred)

    print("[3/15] GT vs predicted distribution")
    plot_grade_distribution(df)

    print("[4/15] Per-grade precision / recall / F1")
    per_grade = calculate_per_grade_metrics(y_true, y_pred)
    plot_per_grade_metrics(per_grade)

    print("[5/15] Correct vs incorrect")
    plot_correct_incorrect(df)

    print("[6/15] Error distance distribution")
    plot_error_distance(df)

    print("[7/15] Error type distribution")
    plot_error_types(df)

    print("[8/15] Confidence distribution")
    plot_confidence_distribution(df)

    print("[9/15] Confidence: correct vs incorrect")
    plot_confidence_correct_vs_incorrect(df)

    print("[10/15] Grade-wise confidence")
    plot_grade_wise_confidence(df)

    print("[11/15] Mean class probabilities by GT grade")
    plot_mean_class_probabilities(df)

    print("[12/15] Calibration / reliability")
    plot_calibration_reliability(df)

    print("[13/15] Patient-level accuracy")
    patient_metrics = calculate_patient_metrics(df)
    plot_patient_accuracy(patient_metrics)

    print("[14/15] Offset-wise performance")
    offset_metrics = calculate_offset_metrics(df)
    plot_offset_performance(offset_metrics)

    print("[15/25] Error-pair distribution")
    error_pairs = calculate_error_pairs(df)
    plot_error_pairs(error_pairs)

    # Plots 16 to 25
    generate_grade_confidence_per_patient_reports(df)

    # Save numerical summary CSVs
    create_overall_metrics(df).to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    per_grade.to_csv(OUTPUT_DIR / "per_grade_metrics.csv", index=False)
    patient_metrics.to_csv(OUTPUT_DIR / "patient_level_metrics.csv", index=False)
    offset_metrics.to_csv(OUTPUT_DIR / "offset_level_metrics.csv", index=False)
    error_pairs.to_csv(OUTPUT_DIR / "error_pair_counts.csv", index=False)

    print("Core evaluation report complete (31 files generated).")

# =============================================================================
# CONFIDENCE DISTRIBUTION REPORT (10 BAR CHARTS)
# =============================================================================

def generate_confidence_plot(grade, subset, correct, output_path):
    confidence_values = subset["_confidence_percent"].to_numpy(dtype=float)
    patient_values = subset["_patient_numeric"].astype(int).tolist()
    case_count = len(subset)

    if case_count > 0:
        mean_confidence = float(np.mean(confidence_values))
        median_confidence = float(np.median(confidence_values))
    else:
        mean_confidence = np.nan
        median_confidence = np.nan

    figure_width = max(12, case_count * 0.18)
    figure_width = min(figure_width, 40)

    fig, ax = plt.subplots(figsize=(figure_width, 7))
    x = np.arange(case_count)

    if case_count > 0:
        bars = ax.bar(x, confidence_values, width=0.82)
        for bar, confidence in zip(bars, confidence_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(confidence + 1.0, 98.5),
                f"{confidence:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=0,
            )
        ax.axhline(mean_confidence, linestyle="--", linewidth=1.2, label=f"Mean confidence: {mean_confidence:.2f}%")
        ax.axhline(median_confidence, linestyle=":", linewidth=1.5, label=f"Median confidence: {median_confidence:.2f}%")

    prediction_type = "Correct Predictions" if correct else "Wrong Predictions"
    ax.set_title(
        f"Pfirrmann Grade {grade} | {prediction_type}\nGT Grade {grade} | {case_count} cases | Confidence = P(Grade {grade})",
        fontsize=14,
        pad=10,
    )
    ax.set_ylabel(f"Confidence for Grade {grade} (%)", fontsize=11)
    ax.set_xlabel("Patient ID / Individual Case", fontsize=11)
    ax.set_ylim(0, 100)

    if case_count > 0:
        ax.set_xticks(x)
        ax.set_xticklabels(patient_values, rotation=90, fontsize=7)
    else:
        ax.set_xticks([])

    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)

    if case_count > 0:
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_confidence_distributions(results_df: pd.DataFrame, conf_dir: Path):
    conf_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("=" * 80)
    print("GENERATING 10 GRADE CONFIDENCE DISTRIBUTION BAR CHARTS")
    print(f"Target folder: {conf_dir}")
    print("=" * 80)

    df = results_df.copy()
    df["_patient_numeric"] = pd.to_numeric(df["Patient_ID"], errors="coerce")
    df["_gt_numeric"] = pd.to_numeric(df["Pfirrmann_Grade"], errors="coerce")
    df["_pred_numeric"] = pd.to_numeric(df["Predicted_Grade"], errors="coerce")
    df["_confidence_numeric"] = pd.to_numeric(df["Prediction_Confidence"], errors="coerce")

    max_conf = df["_confidence_numeric"].max()
    if max_conf <= 1.0:
        df["_confidence_percent"] = df["_confidence_numeric"] * 100.0
    else:
        df["_confidence_percent"] = df["_confidence_numeric"]

    df["_confidence_percent"] = df["_confidence_percent"].clip(lower=0, upper=100)

    for grade in GRADES:
        grade_df = df[df["_gt_numeric"] == grade].sort_values(by=["_patient_numeric"], kind="stable")
        
        # Correct predictions
        correct_df = grade_df[grade_df["_pred_numeric"] == grade]
        c_path = conf_dir / f"grade_{grade}_correct_confidence.png"
        generate_confidence_plot(grade, correct_df, True, c_path)
        print(f"  [SAVED] grade_{grade}_correct_confidence.png ({len(correct_df)} cases)")

        # Wrong predictions
        wrong_df = grade_df[grade_df["_pred_numeric"] != grade]
        w_path = conf_dir / f"grade_{grade}_wrong_confidence.png"
        generate_confidence_plot(grade, wrong_df, False, w_path)
        print(f"  [SAVED] grade_{grade}_wrong_confidence.png ({len(wrong_df)} cases)")

    print("Confidence distribution graphs complete (10 files generated).")

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Model 4 Comprehensive Test Set Inference and Report Generator"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename ('best_model.pt', 'latest_checkpoint.pt') or full path. Default: best_model.pt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=6,
        help="Inference batch size. Default: 6",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers. Default: 4",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store results. Default: <script_dir>/results",
    )
    parser.add_argument(
        "--skip_visuals",
        action="store_true",
        help="Skip generating individual 731 image montages in test_predictions/",
    )
    args = parser.parse_args()

    results_dir = Path(args.output_dir) if args.output_dir else (_SCRIPT_DIR / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    prediction_csv = results_dir / "prediction_results.csv"
    summary_csv = results_dir / "evaluation_summary.csv"
    visual_dir = results_dir / "test_predictions"
    r_dir = results_dir / "r"
    conf_dir = results_dir / "conf"

    checkpoint_path = resolve_checkpoint_path(args.checkpoint, PROJECT_ROOT)
    device = get_device()
    model = load_model(checkpoint_path, device)

    print()
    print("=" * 80)
    print("RUNNING MODEL 4 TEST INFERENCE")
    print("=" * 80)
    print(f"  Dataset CSV : {DEFAULT_DATASET_CSV}")
    print(f"  Batch Size  : {args.batch_size}")
    print(f"  Results Dir : {results_dir}")
    print("=" * 80)

    test_dataset, test_loader = create_test_loader(
        csv_path=DEFAULT_DATASET_CSV,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results_df = run_inference(model, test_dataset, test_loader, device)
    validate_results(results_df)

    # Generate visual montages (731 images)
    if not args.skip_visuals:
        visual_dir.mkdir(parents=True, exist_ok=True)
        print()
        print("=" * 80)
        print("GENERATING 731 TEST PREDICTION VISUAL MONTAGES")
        print(f"Target folder: {visual_dir}")
        print("=" * 80)

        for i in tqdm(range(len(results_df)), desc="Saving visual montages"):
            row = results_df.iloc[i]
            filename = row["Prediction_Image"]
            output_image_path = visual_dir / filename
            image_path = PROJECT_ROOT / row["MRI_Output_Path"]
            probs = [float(row[col]) for col in PROBABILITY_COLUMNS]

            create_prediction_visual(
                image_path=image_path,
                patient_id=int(row["Patient_ID"]),
                slice_number=int(row["Slice_Number"]),
                offset=int(row["Offset"]),
                medial_slice=int(row["Medial_Slice_Number"]),
                ivd_label=str(row["IVD_Label"]),
                gt_grade=int(row["Pfirrmann_Grade"]),
                predicted_grade=int(row["Predicted_Grade"]),
                confidence=float(row["Prediction_Confidence"]),
                gt_probability=float(row["GT_Grade_Probability"]),
                error_distance=int(row["Error_Distance"]),
                error_type=str(row["Prediction_Error_Type"]),
                probabilities=probs,
                output_path=output_image_path,
            )
        print(f"Saved {len(results_df)} visual montages in {visual_dir}")

        # Update Prediction_Image column with path relative to project root
        results_df["Prediction_Image"] = [
            (visual_dir / fn).relative_to(PROJECT_ROOT).as_posix()
            for fn in results_df["Prediction_Image"]
        ]
    else:
        print("\nSkipping individual visual montages as requested (--skip_visuals).")

    # Save CSVs
    results_df.to_csv(prediction_csv, index=False)
    print(f"\nSaved prediction results:\n  {prediction_csv}")

    summary_df = create_summary(results_df)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved evaluation summary:\n  {summary_csv}")

    # Generate 31 core evaluation figures & CSVs in results/r
    run_core_evaluation_report(results_df, r_dir)

    # Generate 10 confidence distribution bar charts in results/conf
    run_confidence_distributions(results_df, conf_dir)

    correct = int(results_df["Correct"].sum())
    total = len(results_df)
    accuracy = correct / total

    print()
    print("=" * 80)
    print("MODEL 4 COMPLETE INFERENCE & EVALUATION SUITE FINISHED!")
    print("=" * 80)
    print(f"  Total Test Slices  : {total}")
    print(f"  Patients           : {results_df['Patient_ID'].nunique()}")
    print(f"  Exact Accuracy     : {accuracy * 100:.2f}%")
    print(f"  All Results Stored In:")
    print(f"    - CSVs          : {prediction_csv.name}, {summary_csv.name}")
    print(f"    - Montages (731): {visual_dir}")
    print(f"    - Core Suite (31): {r_dir}")
    print(f"    - Conf Suite (10): {conf_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()