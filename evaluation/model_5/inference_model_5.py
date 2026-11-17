"""
Model 5 Multi-View Comprehensive Test Set Inference and Reporting Pipeline
────────────────────────────────────────────────────────────────────────────
Evaluates PfirrmannClassifierV5 on the multi-view test dataset (149 3D disc volumes,
each consisting of 5 contiguous sagittal slices: offsets [-2, -1, 0, +1, +2]).

Features:
    - Runs full test-set inference on PfirrmannClassifierV5 (frozen ViT-B backbone,
      SAM neck, spatial attention pooling, and view attention aggregator).
    - Checkpoint Flexibility:
        * Defaults to 'best_model.pt' in outputs/model_5/phase1_2/checkpoints/ (or candidate dirs).
        * Can be switched to 'latest_checkpoint.pt' or any custom path via --checkpoint.
    - Default Batch Size: 1 (--batch_size 1) for strict memory stability.
    - Headless Matplotlib Backend: 'Agg' enabled for zero-display server execution.
    - Consolidated Output Structure:
        All results, figures, and CSVs are automatically written into a dedicated
        'results/' subfolder:
            results/
                prediction_results.csv        (149 disc volume predictions with all probabilities & metrics)
                evaluation_summary.csv        (Overall and per-grade summary table)
                test_predictions/             (149 5-slice composite prediction visual montages)
                r/                            (31 core evaluation files: 25 diagnostic PNGs + 6 summary CSVs)
                conf/                         (10 Grade 1-5 correct and wrong confidence bar charts)

Usage:
    # Run default test inference (best model, batch size 1, all visuals and reports):
    python evaluation/model_5/inference_model_5.py

    # Run using the latest checkpoint instead:
    python evaluation/model_5/inference_model_5.py --checkpoint latest_checkpoint.pt

    # Custom batch size (e.g., 2):
    python evaluation/model_5/inference_model_5.py --batch_size 2

    # Fast evaluation without generating individual 5-slice montages:
    python evaluation/model_5/inference_model_5.py --skip_visuals
"""

import argparse
from collections import Counter
import csv
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# =============================================================================
# PROJECT ROOT & IMPORT RESOLUTION
# =============================================================================

_SCRIPT_DIR = Path(__file__).resolve().parent

candidates = [
    _SCRIPT_DIR.parent.parent,         # if run from <root>/evaluation/model_5/ -> <root>
    _SCRIPT_DIR.parent.parent.parent,  # if run from <root>/context/evaluation/model_5/ -> <root>
    _SCRIPT_DIR.parent,                # if run from <root>/evaluation/ -> <root>
    Path.cwd(),
]

PROJECT_ROOT = _SCRIPT_DIR.parent.parent
for c in candidates:
    if (c / "data").exists() or (c / "training").exists():
        PROJECT_ROOT = c.resolve()
        break

for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "context"), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Model import with multi-tier fallbacks
try:
    from training.models.pfirrmann_classifier_v5 import PfirrmannClassifierV5
except ImportError:
    try:
        from model_5.training.models.pfirrmann_classifier_v5 import PfirrmannClassifierV5
    except ImportError:
        from models.pfirrmann_classifier_v5 import PfirrmannClassifierV5


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

DEFAULT_DATASET_CSV = PROJECT_ROOT / "data" / "new_final" / "final_data" / "dataset_multi_view_v2.csv"
DATASET_CSV = DEFAULT_DATASET_CSV

OFFSETS = [-2, -1, 0, 1, 2]
OFFSET_COLS = [f"Image_{off}" for off in OFFSETS]
NUM_VIEWS = len(OFFSETS)
NUM_CLASSES = 5
GRADES = [1, 2, 3, 4, 5]

GRADE_TO_LABEL = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
LABEL_TO_GRADE = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}

PROBABILITY_COLUMNS = [f"Prob_Grade_{g}" for g in GRADES]
IMAGE_SIZE = (1024, 1024)
DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_WORKERS = 2


def resolve_checkpoint_path(checkpoint_arg: str = None) -> Path:
    if checkpoint_arg:
        p = Path(checkpoint_arg)
        if p.exists():
            return p
        candidates = [
            PROJECT_ROOT / "outputs" / "model_5" / "phase1_2" / "checkpoints" / checkpoint_arg,
            PROJECT_ROOT / "outputs" / "model_5" / "checkpoints" / checkpoint_arg,
            PROJECT_ROOT / checkpoint_arg,
            Path("/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/checkpoints") / checkpoint_arg,
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_arg}")

    default_paths = [
        PROJECT_ROOT / "outputs" / "model_5" / "phase1_2" / "checkpoints" / "best_model.pt",
        Path("/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/checkpoints/best_model.pt"),
        PROJECT_ROOT / "outputs" / "model_5" / "checkpoints" / "best_model.pt",
        PROJECT_ROOT / "context" / "outputs" / "model_5" / "phase1_2" / "checkpoints" / "best_model.pt",
        PROJECT_ROOT / "context" / "outputs" / "model_5" / "checkpoints" / "best_model.pt",
        _SCRIPT_DIR.parent.parent / "outputs" / "model_5" / "phase1_2" / "checkpoints" / "best_model.pt",
        _SCRIPT_DIR.parent.parent / "outputs" / "model_5" / "checkpoints" / "best_model.pt",
        Path("D:/BAAA project/spider-pfirrmann-mri-core/outputs/model_5/checkpoints/best_model.pt"),
    ]
    for p in default_paths:
        if p.exists():
            return p

    return PROJECT_ROOT / "outputs" / "model_5" / "phase1_2" / "checkpoints" / "best_model.pt"


# =============================================================================
# DATASET
# =============================================================================

class PfirrmannMultiViewTestDataset(Dataset):
    """
    Multi-view Pfirrmann test dataset.
    Loads 5 contiguous sagittal slice crops for each disc volume (offsets: -2, -1, 0, +1, +2).
    """

    def __init__(self, csv_path: Path, split: str = "test"):
        super().__init__()
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            alt_candidates = [
                PROJECT_ROOT / "data" / "new_final" / "final_data" / "dataset_multi_view_v2.csv",
                PROJECT_ROOT / "context" / "data" / "new_final" / "final_data" / "dataset_multi_view_v2.csv",
                Path("D:/BAAA project/spider-pfirrmann-mri-core/data/new_final/final_data/dataset_multi_view_v2.csv"),
            ]
            found = False
            for c in alt_candidates:
                if c.exists():
                    self.csv_path = c
                    found = True
                    break
            if not found:
                raise FileNotFoundError(f"Dataset CSV not found:\n{csv_path}")

        self.df = pd.read_csv(self.csv_path)

        if "Split" in self.df.columns and split is not None:
            self.df = self.df[
                self.df["Split"].astype(str).str.lower() == split.lower()
            ].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split: {split}")

        # Validate grades
        grades = pd.to_numeric(self.df["Pfirrmann_Grade"], errors="coerce")
        if grades.isna().any():
            raise ValueError("Dataset contains invalid Pfirrmann grades.")
        self.df["Pfirrmann_Grade"] = grades.astype(int)

        print(f"PfirrmannMultiViewTestDataset | Split: {split} | Samples: {len(self.df)} disc volumes")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, relative_path: str) -> Path:
        rel = str(relative_path).strip().replace("\\", "/")
        candidates = [
            PROJECT_ROOT / rel,
            PROJECT_ROOT / "context" / rel,
            Path("D:/BAAA project/spider-pfirrmann-mri-core") / rel,
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(f"Image not found at any candidate location for:\n{rel}")

    def _load_single_slice(self, relative_path: str) -> torch.Tensor:
        path = self._resolve_image_path(relative_path)
        image = Image.open(path).convert("L")
        if image.size != IMAGE_SIZE:
            image = image.resize(IMAGE_SIZE, resample=Image.BILINEAR)

        arr = np.asarray(image, dtype=np.float32) / 255.0
        # 3 channels [3, 1024, 1024]
        tensor = np.stack([arr, arr, arr], axis=0)
        return torch.from_numpy(tensor)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        row = self.df.iloc[index]
        slices = []
        for col in OFFSET_COLS:
            slice_tensor = self._load_single_slice(row[col])
            slices.append(slice_tensor)

        # [5, 3, 1024, 1024]
        images = torch.stack(slices, dim=0)

        gt_grade = int(row["Pfirrmann_Grade"])
        label = torch.tensor(GRADE_TO_LABEL[gt_grade], dtype=torch.long)

        metadata = {
            "Patient_ID": int(row["Patient_ID"]),
            "IVD_Label": int(row["IVD_Label"]),
            "Pfirrmann_Grade": gt_grade,
            "Image_-2": str(row["Image_-2"]),
            "Image_-1": str(row["Image_-1"]),
            "Image_0": str(row["Image_0"]),
            "Image_1": str(row["Image_1"]),
            "Image_2": str(row["Image_2"]),
        }

        return images, label, metadata


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_5(checkpoint_path: Path, device: torch.device) -> tuple[PfirrmannClassifierV5, dict]:
    print("\n" + "=" * 80)
    print("LOADING MODEL 5 MULTI-VIEW CLASSIFIER")
    print("=" * 80)
    print(f"  Checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model 5 checkpoint not found at:\n{checkpoint_path}")

    model = PfirrmannClassifierV5(num_views=NUM_VIEWS, num_classes=NUM_CLASSES)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        print(f"  Loaded state_dict from epoch : {ckpt.get('epoch', 'N/A')}")
        if "best_metrics" in ckpt:
            print(f"  Best metrics in checkpoint   : {ckpt['best_metrics']}")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print("  Model 5 loaded successfully (all 174 state_dict keys matched strictly).")
    print("=" * 80 + "\n")
    return model, ckpt if isinstance(ckpt, dict) else {}


# =============================================================================
# INFERENCE LOOP
# =============================================================================

@torch.inference_mode()
def run_test_inference(
    model: PfirrmannClassifierV5,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    records = []

    print("\nRunning Multi-View Test Set Forward Pass...")
    for images, labels, metadatas in tqdm(loader, desc="Model 5 Inference", dynamic_ncols=True):
        images = images.to(device, non_blocking=True)  # [B, 5, 3, 1024, 1024]
        primary_logits, center_logits = model(images)  # [B, 5], [B, 5]

        probs = F.softmax(primary_logits, dim=-1).cpu().numpy()
        center_probs = F.softmax(center_logits, dim=-1).cpu().numpy()

        preds = np.argmax(probs, axis=1)
        center_preds = np.argmax(center_probs, axis=1)

        batch_size = labels.size(0)
        for i in range(batch_size):
            gt_grade = int(metadatas["Pfirrmann_Grade"][i])
            pred_grade = int(LABEL_TO_GRADE[preds[i]])
            center_pred_grade = int(LABEL_TO_GRADE[center_preds[i]])
            conf = float(probs[i, preds[i]])
            gt_prob = float(probs[i, GRADE_TO_LABEL[gt_grade]])
            err_dist = abs(gt_grade - pred_grade)

            if err_dist == 0:
                err_type = "Correct"
            elif err_dist == 1:
                err_type = "Adjacent Error"
            else:
                err_type = "Non-Adjacent Error"

            row = {
                "Patient_ID": int(metadatas["Patient_ID"][i]),
                "IVD_Label": int(metadatas["IVD_Label"][i]),
                "Pfirrmann_Grade": gt_grade,
                "Predicted_Grade": pred_grade,
                "Correct": bool(gt_grade == pred_grade),
                "Error_Distance": err_dist,
                "Prediction_Error_Type": err_type,
                "Prediction_Confidence": conf,
                "GT_Grade_Probability": gt_prob,
                "Center_Predicted_Grade": center_pred_grade,
                "Center_Agreement": bool(pred_grade == center_pred_grade),
                "Image_-2": metadatas["Image_-2"][i],
                "Image_-1": metadatas["Image_-1"][i],
                "Image_0": metadatas["Image_0"][i],
                "Image_1": metadatas["Image_1"][i],
                "Image_2": metadatas["Image_2"][i],
            }

            for g in GRADES:
                row[f"Prob_Grade_{g}"] = float(probs[i, GRADE_TO_LABEL[g]])

            records.append(row)

    df_results = pd.DataFrame(records)
    return df_results


# =============================================================================
# EVALUATION SUMMARY BUILDER
# =============================================================================

def compute_evaluation_summary(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["Pfirrmann_Grade"].to_numpy(dtype=int)
    y_pred = df["Predicted_Grade"].to_numpy(dtype=int)

    total_samples = len(df)
    unique_patients = df["Patient_ID"].nunique()
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mean_conf = df["Prediction_Confidence"].mean()
    mean_gt_prob = df["GT_Grade_Probability"].mean()
    mean_err_dist = df["Error_Distance"].mean()

    err_counts = df["Prediction_Error_Type"].value_counts().to_dict()
    correct_count = err_counts.get("Correct", 0)
    adjacent_count = err_counts.get("Adjacent Error", 0)
    non_adjacent_count = err_counts.get("Non-Adjacent Error", 0)

    rows = []
    # Overall row
    rows.append({
        "Section": "Overall",
        "Grade": "All",
        "Test_Samples": total_samples,
        "Test_Patients": unique_patients,
        "GT_Samples": total_samples,
        "Predicted_Samples": total_samples,
        "Correct_Samples": correct_count,
        "Incorrect_Samples": total_samples - correct_count,
        "Accuracy": acc,
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": macro_f1,
        "Macro_F1": macro_f1,
        "Mean_Confidence": mean_conf,
        "Mean_GT_Probability": mean_gt_prob,
        "Mean_Error_Distance": mean_err_dist,
        "Correct_Count": correct_count,
        "Adjacent_Error_Count": adjacent_count,
        "Non_Adjacent_Error_Count": non_adjacent_count,
    })

    # Per-grade rows
    for g in GRADES:
        mask_gt = (y_true == g)
        gt_count = int(np.sum(mask_gt))
        pred_count = int(np.sum(y_pred == g))

        p = precision_score(y_true == g, y_pred == g, zero_division=0)
        r = recall_score(y_true == g, y_pred == g, zero_division=0)
        f = f1_score(y_true == g, y_pred == g, zero_division=0)

        sub = df[df["Pfirrmann_Grade"] == g]
        grade_conf = sub["Prediction_Confidence"].mean() if len(sub) > 0 else 0.0
        grade_gt_prob = sub["GT_Grade_Probability"].mean() if len(sub) > 0 else 0.0

        rows.append({
            "Section": "Per Grade",
            "Grade": g,
            "Test_Samples": total_samples,
            "Test_Patients": "",
            "GT_Samples": gt_count,
            "Predicted_Samples": pred_count,
            "Correct_Samples": "",
            "Incorrect_Samples": "",
            "Accuracy": "",
            "Precision": p,
            "Recall": r,
            "F1": f,
            "Macro_F1": macro_f1,
            "Mean_Confidence": grade_conf,
            "Mean_GT_Probability": grade_gt_prob,
            "Mean_Error_Distance": "",
            "Correct_Count": "",
            "Adjacent_Error_Count": "",
            "Non_Adjacent_Error_Count": "",
        })

    summary_df = pd.DataFrame(rows)
    return summary_df


# =============================================================================
# VISUAL REPORT GENERATOR (5-View Composite Strip)
# =============================================================================

def get_font(size: int):
    candidates = ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "calibri.ttf"]
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            continue
    return ImageFont.load_default()


def generate_visual_reports(df: pd.DataFrame, visual_dir: Path) -> None:
    visual_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating 5-View Visual Prediction Montages in {visual_dir}...")

    font_title = get_font(22)
    font_main = get_font(18)
    font_small = get_font(15)

    base_names = []
    for _, row in df.iterrows():
        base_names.append(
            f"Patient{int(row['Patient_ID'])}_IVD{int(row['IVD_Label'])}_GT{int(row['Pfirrmann_Grade'])}_Pred{int(row['Predicted_Grade'])}.png"
        )

    counts = Counter(base_names)
    occurrences = Counter()
    image_paths = []

    for name in base_names:
        occurrences[name] += 1
        if counts[name] > 1:
            stem = Path(name).stem
            filename = f"{stem}_Idx{occurrences[name]}.png"
        else:
            filename = name
        image_paths.append(str(visual_dir / filename))

    df["Prediction_Image"] = image_paths

    thumb_w, thumb_h = 320, 320
    header_h = 230
    footer_h = 40
    total_w = thumb_w * 5
    total_h = header_h + thumb_h + footer_h

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Saving 5-Slice Montages", dynamic_ncols=True):
        canvas = Image.new("RGB", (total_w, total_h), "white")
        draw = ImageDraw.Draw(canvas)

        gt = int(row["Pfirrmann_Grade"])
        pred = int(row["Predicted_Grade"])
        conf = float(row["Prediction_Confidence"])
        err_dist = int(row["Error_Distance"])
        err_type = str(row["Prediction_Error_Type"])
        center_pred = int(row["Center_Predicted_Grade"])

        # Status Banner background color
        if err_dist == 0:
            badge_color = "#2e7d32"  # Green
        elif err_dist == 1:
            badge_color = "#e65100"  # Orange
        else:
            badge_color = "#c62828"  # Red

        # Header Box
        draw.rectangle([(0, 0), (total_w, header_h)], fill="#f8f9fa")
        draw.line([(0, header_h), (total_w, header_h)], fill="#dee2e6", width=2)

        # Title
        title_text = "SPIDER Pfirrmann Grading — Model 5 Multi-View Evaluation"
        draw.text((20, 15), title_text, fill="#212529", font=font_title)

        meta_text = f"Patient ID: {int(row['Patient_ID'])}  |  IVD Level: L{int(row['IVD_Label'])}  |  5 Contiguous Sagittal Slices"
        draw.text((20, 50), meta_text, fill="#495057", font=font_main)

        # Status Badge Pill
        status_text = f" {err_type.upper()}  —  Ground Truth: Grade {gt}  vs.  Multi-View Predicted: Grade {pred} "
        draw.rectangle([(20, 85), (820, 125)], fill=badge_color)
        draw.text((30, 93), status_text, fill="white", font=font_main)

        # Confidence & Probabilities
        prob_str = "  |  ".join([f"P(G{g}): {row[f'Prob_Grade_{g}'] * 100:.1f}%" for g in GRADES])
        conf_text = f"Multi-View Fused Confidence: {conf * 100:.2f}%  |  {prob_str}"
        draw.text((20, 140), conf_text, fill="#212529", font=font_small)

        center_agree = "MATCH" if pred == center_pred else "DISAGREE"
        center_text = f"Auxiliary Center Slice (Offset 0): Grade {center_pred}  [{center_agree}]  |  Error Distance: {err_dist} grade(s)"
        draw.text((20, 175), center_text, fill="#6c757d", font=font_small)

        # Paste 5 Slice Thumbnails
        for i, off in enumerate(OFFSETS):
            col_name = f"Image_{off}"
            rel_path = str(row[col_name]).strip().replace("\\", "/")
            try:
                candidates_img = [
                    PROJECT_ROOT / rel_path,
                    PROJECT_ROOT / "context" / rel_path,
                    Path("D:/BAAA project/spider-pfirrmann-mri-core") / rel_path,
                ]
                img_path = None
                for ci in candidates_img:
                    if ci.exists():
                        img_path = ci
                        break
                if img_path is not None:
                    slice_img = Image.open(img_path).convert("RGB")
                else:
                    slice_img = Image.new("RGB", (thumb_w, thumb_h), "#212529")
            except Exception:
                slice_img = Image.new("RGB", (thumb_w, thumb_h), "#212529")

            slice_thumb = slice_img.resize((thumb_w, thumb_h), Image.BILINEAR)
            x_pos = i * thumb_w
            canvas.paste(slice_thumb, (x_pos, header_h))

            # Highlight Center Slice (Offset 0 = index 2)
            if off == 0:
                draw.rectangle(
                    [(x_pos, header_h), (x_pos + thumb_w, header_h + thumb_h)],
                    outline="#007bff",
                    width=4,
                )

            # Footer label below each slice
            label_tag = f"Offset {off:+d}  {'[Center]' if off == 0 else ''}"
            draw.rectangle([(x_pos, header_h + thumb_h), (x_pos + thumb_w, total_h)], fill="#e9ecef")
            draw.text((x_pos + 100, header_h + thumb_h + 10), label_tag, fill="#343a40", font=font_small)

        canvas.save(df.at[idx, "Prediction_Image"], quality=95)

    # Format Prediction_Image as portable relative POSIX paths
    try:
        df["Prediction_Image"] = [
            Path(p).relative_to(PROJECT_ROOT).as_posix()
            for p in df["Prediction_Image"]
        ]
    except Exception:
        df["Prediction_Image"] = [Path(p).as_posix() for p in df["Prediction_Image"]]


# =============================================================================
# CONFIDENCE DISTRIBUTION GENERATOR (Right & Wrong for All 5 Grades -> 10 Charts)
# =============================================================================

def run_confidence_distributions(df: pd.DataFrame, conf_dir: Path) -> None:
    conf_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating 10 Confidence Distribution Bar Charts in {conf_dir}...")

    df_copy = df.copy()
    df_copy["_confidence_percent"] = df_copy["Prediction_Confidence"] * 100.0

    min_w = 12
    max_w = 36
    w_per_case = 0.28
    base_h = 7

    for g in GRADES:
        for correct in [True, False]:
            status_str = "correct" if correct else "wrong"
            subset = df_copy[(df_copy["Pfirrmann_Grade"] == g) & (df_copy["Correct"] == correct)].reset_index(drop=True)

            case_count = len(subset)
            out_file = conf_dir / f"grade_{g}_{status_str}_confidence.png"

            fig_w = min(max_w, max(min_w, case_count * w_per_case))
            fig, ax = plt.subplots(figsize=(fig_w, base_h), dpi=200)

            if case_count > 0:
                confs = subset["_confidence_percent"].to_numpy()
                case_labels = [f"P{int(row['Patient_ID'])}_IVD{int(row['IVD_Label'])}" for _, row in subset.iterrows()]
                x = np.arange(case_count)

                bar_color = "#2e7d32" if correct else "#c62828"
                bars = ax.bar(x, confs, color=bar_color, width=0.75, alpha=0.88, edgecolor="black", linewidth=0.6)

                mean_val = float(np.mean(confs))
                median_val = float(np.median(confs))

                for bar, c_val in zip(bars, confs):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        min(c_val + 1.2, 98.0),
                        f"{c_val:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=0,
                        fontweight="bold",
                    )

                ax.axhline(mean_val, color="#004d40" if correct else "#b71c1c", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}%")
                ax.axhline(median_val, color="#1b5e20" if correct else "#880e4f", linestyle=":", linewidth=1.8, label=f"Median: {median_val:.2f}%")

                ax.set_xticks(x)
                ax.set_xticklabels(case_labels, rotation=90, fontsize=7)
                ax.legend(loc="upper right", framealpha=0.9)
            else:
                ax.text(0.5, 0.5, f"No {status_str} cases for Grade {g}", ha="center", va="center", fontsize=14, color="gray")
                ax.set_xticks([])

            pred_title = "Correct Predictions" if correct else "Wrong Predictions"
            ax.set_title(
                f"Model 5 — Pfirrmann Grade {g} | {pred_title}\n"
                f"Ground Truth Grade {g} | Total Cases: {case_count} | Multi-View Fused Confidence",
                fontsize=13,
                pad=12,
                fontweight="bold",
            )
            ax.set_ylabel(f"Confidence for Grade {g} (%)", fontsize=11)
            ax.set_xlabel("Patient & Disc Case (Patient_IVD)", fontsize=11)
            ax.set_ylim(0, 105)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()
            plt.savefig(out_file, dpi=200)
            plt.close(fig)

    print(f"  Saved 10 confidence distribution bar charts to {conf_dir}")


# =============================================================================
# CORE EVALUATION REPORT BATTERY (25 Figures + 6 CSV Reports -> 31 Files)
# =============================================================================

def _save_r_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_core_evaluation_report(df: pd.DataFrame, r_dir: Path) -> None:
    r_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("GENERATING COMPLETE 25-FIGURE DIAGNOSTIC SUITE & 6 CSV REPORTS IN r/")
    print(f"  Target directory: {r_dir}")
    print("=" * 80)

    y_true = df["Pfirrmann_Grade"].to_numpy(dtype=int)
    y_pred = df["Predicted_Grade"].to_numpy(dtype=int)

    # 1. 01_confusion_matrix.png
    cm = confusion_matrix(y_true, y_pred, labels=GRADES)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        title="Model 5 Multi-View — Confusion Matrix",
        xlabel="Predicted Pfirrmann Grade",
        ylabel="Ground Truth Pfirrmann Grade",
        xticks=np.arange(5),
        yticks=np.arange(5),
        xticklabels=GRADES,
        yticklabels=GRADES,
    )
    thresh = cm.max() / 2.0
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontweight="bold")
    _save_r_fig(fig, r_dir / "01_confusion_matrix.png")

    # 2. 02_normalized_confusion_matrix.png
    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    image = ax.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set(
        title="Model 5 Multi-View — Normalized Confusion Matrix (Recall)",
        xlabel="Predicted Pfirrmann Grade",
        ylabel="Ground Truth Pfirrmann Grade",
        xticks=np.arange(5),
        yticks=np.arange(5),
        xticklabels=GRADES,
        yticklabels=GRADES,
    )
    for i in range(5):
        for j in range(5):
            val = normalized[i, j] * 100.0
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color="white" if val > 50 else "black", fontweight="bold")
    _save_r_fig(fig, r_dir / "02_normalized_confusion_matrix.png")

    # 3. 03_gt_vs_predicted_distribution.png
    gt_counts = df["Pfirrmann_Grade"].value_counts().reindex(GRADES, fill_value=0)
    pred_counts = df["Predicted_Grade"].value_counts().reindex(GRADES, fill_value=0)
    x = np.arange(5)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.bar(x - width / 2, gt_counts.values, width, label="Ground Truth", color="#1976d2", alpha=0.9)
    ax.bar(x + width / 2, pred_counts.values, width, label="Predicted", color="#ff9800", alpha=0.9)
    ax.set(
        title="Model 5 — Ground Truth vs Predicted Grade Distribution",
        xlabel="Pfirrmann Grade",
        ylabel="Number of Discs",
        xticks=x,
        xticklabels=GRADES,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    _save_r_fig(fig, r_dir / "03_gt_vs_predicted_distribution.png")

    # 4. 04_per_grade_precision_recall_f1.png + per_grade_metrics.csv
    precision = precision_score(y_true, y_pred, labels=GRADES, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=GRADES, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=GRADES, average=None, zero_division=0)
    support = pd.Series(y_true).value_counts().reindex(GRADES, fill_value=0).values
    per_grade_df = pd.DataFrame({
        "Grade": GRADES,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Support": support,
    })
    per_grade_df.to_csv(r_dir / "per_grade_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    width = 0.25
    ax.bar(x - width, per_grade_df["Precision"], width, label="Precision", color="#1976d2", alpha=0.9)
    ax.bar(x, per_grade_df["Recall"], width, label="Recall", color="#388e3c", alpha=0.9)
    ax.bar(x + width, per_grade_df["F1"], width, label="F1", color="#f57c00", alpha=0.9)
    ax.set(
        title="Model 5 — Per-Grade Precision, Recall and F1",
        xlabel="Pfirrmann Grade",
        ylabel="Score",
        xticks=x,
        xticklabels=GRADES,
        ylim=(0, 1.05),
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    _save_r_fig(fig, r_dir / "04_per_grade_precision_recall_f1.png")

    # 5. 05_correct_vs_incorrect.png
    correct = int(df["Correct"].sum())
    incorrect = len(df) - correct
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    bars = ax.bar(["Correct", "Incorrect"], [correct, incorrect], color=["#2e7d32", "#c62828"], width=0.55, edgecolor="black")
    ax.set(title="Model 5 — Correct vs Incorrect Disc Predictions", ylabel="Number of Discs", ylim=(0, len(df) * 1.15))
    for bar, val in zip(bars, [correct, incorrect]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{val}\n({val / len(df) * 100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save_r_fig(fig, r_dir / "05_correct_vs_incorrect.png")

    # 6. 06_error_distance_distribution.png
    dist_counts = df["Error_Distance"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.bar(dist_counts.index.astype(str), dist_counts.values, color=["#2e7d32", "#e65100", "#c62828", "#b71c1c"][:len(dist_counts)], width=0.6, edgecolor="black")
    ax.set(
        title="Model 5 — Error Distance Distribution",
        xlabel="|Ground Truth - Prediction| (Pfirrmann Grades)",
        ylabel="Number of Discs",
        ylim=(0, max(dist_counts.values) * 1.15),
    )
    for bar, val in zip(bars, dist_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val} ({val / len(df) * 100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save_r_fig(fig, r_dir / "06_error_distance_distribution.png")

    # 7. 07_error_type_distribution.png
    order = ["Correct", "Adjacent Error", "Non-Adjacent Error"]
    type_counts = df["Prediction_Error_Type"].value_counts().reindex(order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    bars = ax.bar(type_counts.index, type_counts.values, color=["#2e7d32", "#e65100", "#c62828"], width=0.6, edgecolor="black")
    ax.set(title="Model 5 — Prediction Error Type Distribution", ylabel="Number of Discs", ylim=(0, max(type_counts.values) * 1.15))
    for bar, val in zip(bars, type_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val}\n({val / len(df) * 100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save_r_fig(fig, r_dir / "07_error_type_distribution.png")

    # 8. 08_confidence_distribution.png
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.hist(df["Prediction_Confidence"] * 100.0, bins=20, color="#1976d2", edgecolor="black", alpha=0.85)
    mean_c = df["Prediction_Confidence"].mean() * 100.0
    ax.axvline(mean_c, color="#d32f2f", linestyle="--", linewidth=1.5, label=f"Mean: {mean_c:.1f}%")
    ax.set(title="Model 5 — Multi-View Prediction Confidence Distribution", xlabel="Prediction Confidence (%)", ylabel="Number of Discs", xlim=(0, 100))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    _save_r_fig(fig, r_dir / "08_confidence_distribution.png")

    # 9. 09_confidence_correct_vs_incorrect.png
    c_conf = df.loc[df["Correct"], "Prediction_Confidence"] * 100.0
    inc_conf = df.loc[~df["Correct"], "Prediction_Confidence"] * 100.0
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.boxplot([c_conf, inc_conf], tick_labels=["Correct", "Incorrect"], showmeans=True)
    ax.set(title="Model 5 — Prediction Confidence: Correct vs Incorrect", ylabel="Prediction Confidence (%)", ylim=(0, 105))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save_r_fig(fig, r_dir / "09_confidence_correct_vs_incorrect.png")

    # 10. 10_grade_wise_confidence.png
    data_gw = [df.loc[df["Pfirrmann_Grade"] == g, "Prediction_Confidence"].values * 100.0 for g in GRADES]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    ax.boxplot(data_gw, tick_labels=[f"Grade {g}" for g in GRADES], showmeans=True)
    ax.set(title="Model 5 — Prediction Confidence by Ground Truth Grade", xlabel="Ground Truth Pfirrmann Grade", ylabel="Prediction Confidence (%)", ylim=(0, 105))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save_r_fig(fig, r_dir / "10_grade_wise_confidence.png")

    # 11. 11_mean_class_probabilities_by_gt.png
    grouped = df.groupby("Pfirrmann_Grade")[PROBABILITY_COLUMNS].mean().reindex(GRADES, fill_value=0)
    matrix = grouped.values
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax)
    ax.set(
        title="Model 5 — Mean Predicted Class Probabilities by GT Grade",
        xlabel="Predicted Grade",
        ylabel="Ground Truth Grade",
        xticks=np.arange(5),
        yticks=np.arange(5),
        xticklabels=GRADES,
        yticklabels=GRADES,
    )
    for i in range(5):
        for j in range(5):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="white" if val > 0.5 else "black", fontweight="bold")
    _save_r_fig(fig, r_dir / "11_mean_class_probabilities_by_gt.png")

    # 12. 12_calibration_reliability.png + calibration_bins.csv
    conf_vals = df["Prediction_Confidence"].to_numpy()
    corr_vals = df["Correct"].astype(int).to_numpy()
    bins = np.linspace(0, 1, 11)
    mean_conf_list, obs_acc_list, count_list = [], [], []
    for lower, upper in zip(bins[:-1], bins[1:]):
        if upper == 1:
            mask = (conf_vals >= lower) & (conf_vals <= upper)
        else:
            mask = (conf_vals >= lower) & (conf_vals < upper)
        if not mask.any():
            continue
        mean_conf_list.append(conf_vals[mask].mean())
        obs_acc_list.append(corr_vals[mask].mean())
        count_list.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.plot(mean_conf_list, obs_acc_list, marker="o", color="#1976d2", linewidth=2, label="Model 5 Multi-View")
    ax.set(title="Model 5 — Calibration / Reliability Plot", xlabel="Mean Predicted Confidence", ylabel="Observed Accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    _save_r_fig(fig, r_dir / "12_calibration_reliability.png")

    pd.DataFrame({
        "Mean_Confidence": mean_conf_list,
        "Observed_Accuracy": obs_acc_list,
        "Sample_Count": count_list,
    }).to_csv(r_dir / "calibration_bins.csv", index=False)

    # 13. 13_patient_level_accuracy.png + patient_level_metrics.csv
    p_rows = []
    for pid, grp in df.groupby("Patient_ID", sort=True):
        yt = grp["Pfirrmann_Grade"]
        yp = grp["Predicted_Grade"]
        p_rows.append({
            "Patient_ID": pid,
            "Test_Discs": len(grp),
            "Correct": int(grp["Correct"].sum()),
            "Accuracy": accuracy_score(yt, yp),
            "Macro_F1": f1_score(yt, yp, labels=GRADES, average="macro", zero_division=0),
        })
    patient_df = pd.DataFrame(p_rows)
    patient_df.to_csv(r_dir / "patient_level_metrics.csv", index=False)

    sorted_p = patient_df.sort_values("Accuracy")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    ax.bar(sorted_p["Patient_ID"].astype(str), sorted_p["Accuracy"] * 100.0, color="#1976d2", edgecolor="black")
    ax.set(title="Model 5 — Patient-Level Test Accuracy", xlabel="Patient ID", ylabel="Accuracy (%)", ylim=(0, 105))
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save_r_fig(fig, r_dir / "13_patient_level_accuracy.png")

    # 14. 14_multi_view_vs_center_agreement.png + multi_view_agreement_metrics.csv
    total_samples = len(df)
    agree_cnt = int(df["Center_Agreement"].sum())
    disagree_cnt = total_samples - agree_cnt
    mv_agree_df = pd.DataFrame([
        {"Category": "Agreed with Center", "Count": agree_cnt, "Percentage": agree_cnt / total_samples * 100.0},
        {"Category": "Corrected / Changed Center", "Count": disagree_cnt, "Percentage": disagree_cnt / total_samples * 100.0},
    ])
    mv_agree_df.to_csv(r_dir / "multi_view_agreement_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    bars = ax.bar(mv_agree_df["Category"], mv_agree_df["Count"], color=["#1976d2", "#ff9800"], width=0.55, edgecolor="black")
    ax.set(title="Model 5 — Multi-View vs Center-Slice Agreement", ylabel="Number of Discs", ylim=(0, max(mv_agree_df["Count"]) * 1.15))
    for bar, val in zip(bars, mv_agree_df["Count"]):
        pct = val / total_samples * 100.0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val} ({pct:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    _save_r_fig(fig, r_dir / "14_multi_view_vs_center_agreement.png")

    # 15. 15_error_pair_distribution.png + error_pair_counts.csv
    errors = df.loc[~df["Correct"]]
    if len(errors) > 0:
        error_pairs = (
            errors.groupby(["Pfirrmann_Grade", "Predicted_Grade"])
            .size()
            .reset_index(name="Error_Count")
            .sort_values("Error_Count", ascending=False)
        ).rename(columns={"Pfirrmann_Grade": "GT_Grade"})
    else:
        error_pairs = pd.DataFrame(columns=["GT_Grade", "Predicted_Grade", "Error_Count"])
    error_pairs.to_csv(r_dir / "error_pair_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    if len(error_pairs) > 0:
        labels_ep = [f"G{gt} → G{pred}" for gt, pred in zip(error_pairs["GT_Grade"], error_pairs["Predicted_Grade"])][::-1]
        counts_ep = error_pairs["Error_Count"].to_numpy()[::-1]
        bars = ax.barh(labels_ep, counts_ep, color="#d32f2f", alpha=0.85, edgecolor="black")
        ax.set(title="Model 5 — Ground Truth → Predicted Error Pairs", xlabel="Number of Errors", ylabel="Error Pair")
        for bar, val in zip(bars, counts_ep):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val}", va="center", fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
    else:
        ax.text(0.5, 0.5, "No incorrect predictions", ha="center", va="center")
        ax.set_axis_off()
    _save_r_fig(fig, r_dir / "15_error_pair_distribution.png")

    # 16-25. Grade-wise Patient Confidence Reports (10 figures)
    file_index = 16
    for g in GRADES:
        for status_corr in [True, False]:
            status_name = "Correct" if status_corr else "Incorrect"
            prefix = "correct" if status_corr else "incorrect"
            fn = f"{file_index:02d}_grade_{g}_{prefix}_confidence_per_patient.png"
            file_index += 1

            subset = df[(df["Pfirrmann_Grade"] == g) & (df["Correct"] == status_corr)].copy()
            fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

            if len(subset) == 0:
                ax.text(0.5, 0.5, f"No {status_name.lower()} predictions for Grade {g}", ha="center", va="center", fontsize=14, transform=ax.transAxes)
                ax.set(title=f"Grade {g} - {status_name} Prediction Confidence per Patient", xlabel="Patient ID", ylabel="Prediction Confidence (%)", ylim=(0, 105))
            else:
                pids = sorted(subset["Patient_ID"].unique(), key=lambda v: str(v))
                pid_to_x = {pid: idx for idx, pid in enumerate(pids)}
                rng = np.random.default_rng(42 + g + (0 if status_corr else 100))
                x_vals = np.array([pid_to_x[pid] for pid in subset["Patient_ID"]], dtype=float)
                if len(subset) > 1:
                    x_vals += rng.uniform(-0.18, 0.18, size=len(subset))

                c_vals = subset["Prediction_Confidence"].to_numpy(dtype=float) * 100.0
                mean_c_val = float(np.mean(c_vals))
                med_c_val = float(np.median(c_vals))

                dot_col = "#2e7d32" if status_corr else "#c62828"
                ax.scatter(x_vals, c_vals, alpha=0.85, s=55, color=dot_col, edgecolors="black", linewidth=0.6)
                ax.axhline(mean_c_val, linestyle="--", linewidth=1.5, color="#004d40" if status_corr else "#b71c1c", label=f"Mean = {mean_c_val:.1f}%")
                ax.axhline(med_c_val, linestyle=":", linewidth=1.8, color="#1b5e20" if status_corr else "#880e4f", label=f"Median = {med_c_val:.1f}%")

                ax.set(
                    title=f"Model 5 — Grade {g} {status_name} Prediction Confidence per Patient",
                    xlabel="Patient ID",
                    ylabel="Prediction Confidence (%)",
                    ylim=(0, 105),
                    xticks=np.arange(len(pids)),
                    xticklabels=[str(pid) for pid in pids],
                )
                ax.tick_params(axis="x", rotation=90)
                ax.grid(axis="y", alpha=0.3)
                ax.legend(loc="upper right")
                ax.text(
                    0.02, 0.97,
                    f"Discs: {len(subset)}\nPatients: {subset['Patient_ID'].nunique()}\nMean conf: {mean_c_val:.1f}%\nMedian conf: {med_c_val:.1f}%",
                    transform=ax.transAxes, ha="left", va="top", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.88),
                )
            _save_r_fig(fig, r_dir / fn)

    # Summary metrics CSV
    overall_metrics = pd.DataFrame([{
        "Test_Discs": len(df),
        "Test_Patients": df["Patient_ID"].nunique(),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro_Precision": precision_score(y_true, y_pred, labels=GRADES, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, labels=GRADES, average="macro", zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, labels=GRADES, average="macro", zero_division=0),
        "Mean_Confidence": df["Prediction_Confidence"].mean(),
        "Mean_Error_Distance": df["Error_Distance"].mean(),
        "Exact_Correct_Count": int(df["Correct"].sum()),
        "Adjacent_Error_Count": int((df["Error_Distance"] == 1).sum()),
        "Non_Adjacent_Error_Count": int((df["Error_Distance"] > 1).sum()),
    }])
    overall_metrics.to_csv(r_dir / "metrics_summary.csv", index=False)

    print(f"  Successfully saved all 25 figures and 6 CSV reports to {r_dir}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Model 5 Comprehensive Test Set Inference and Reporting Pipeline")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path or filename of checkpoint (defaults to best_model.pt)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS, help=f"DataLoader workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--dataset_csv", type=str, default=str(DATASET_CSV), help=f"Path to dataset CSV (default: {DATASET_CSV})")
    parser.add_argument("--results_dir", type=str, default=None, help="Root folder for output results (default: evaluation/model_5/results)")
    parser.add_argument("--skip_visuals", action="store_true", help="Skip generating the 149 individual 5-view composite montages")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Available Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # Resolve paths
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    print(f"Resolved checkpoint: {checkpoint_path}")

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = _SCRIPT_DIR / "results"

    results_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = results_dir / "test_predictions"
    conf_dir = results_dir / "conf"
    r_dir = results_dir / "r"

    prediction_csv = results_dir / "prediction_results.csv"
    summary_csv = results_dir / "evaluation_summary.csv"

    # Load dataset
    print("\nLoading test dataset...")
    test_dataset = PfirrmannMultiViewTestDataset(csv_path=Path(args.dataset_csv), split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Load model
    model, ckpt_meta = load_model_5(checkpoint_path=checkpoint_path, device=device)

    # Run inference
    start_time = time.time()
    results_df = run_test_inference(model=model, loader=test_loader, device=device)
    elapsed = time.time() - start_time
    print(f"Inference complete on {len(results_df)} test volumes in {elapsed:.2f} seconds.")

    # 1. Visual Montages (149 images in results/test_predictions/)
    if not args.skip_visuals:
        generate_visual_reports(results_df, visual_dir)
    else:
        print("\nSkipping individual 5-slice montages as requested (--skip_visuals).")

    # 2. Save Prediction Results CSV
    results_df.to_csv(prediction_csv, index=False)
    print(f"\nSaved prediction results:\n  {prediction_csv}")

    # 3. Save Evaluation Summary CSV
    summary_df = compute_evaluation_summary(results_df)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved evaluation summary:\n  {summary_csv}")

    # 4. Generate Core Evaluation Battery (31 files: 25 PNGs + 6 CSVs in results/r/)
    run_core_evaluation_report(results_df, r_dir)

    # 5. Generate Confidence Distributions (10 PNGs in results/conf/)
    run_confidence_distributions(results_df, conf_dir)

    correct_count = int(results_df["Correct"].sum())
    total_count = len(results_df)
    exact_acc = correct_count / total_count * 100.0

    print()
    print("=" * 80)
    print("MODEL 5 MULTI-VIEW COMPLETE INFERENCE & EVALUATION SUITE FINISHED!")
    print("=" * 80)
    print(f"  Total Test Discs   : {total_count}")
    print(f"  Unique Patients    : {results_df['Patient_ID'].nunique()}")
    print(f"  Exact Accuracy     : {exact_acc:.2f}% ({correct_count}/{total_count})")
    print(f"  All Results Stored In:")
    print(f"    - CSVs          : {prediction_csv.name}, {summary_csv.name}")
    print(f"    - Montages (149): {visual_dir}")
    print(f"    - Core Suite (31): {r_dir}")
    print(f"    - Conf Suite (10): {conf_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()