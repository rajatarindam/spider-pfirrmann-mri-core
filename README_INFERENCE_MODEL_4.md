# Model 4 — Single-View 2D Pfirrmann Classifier Inference Guide

> [!NOTE]
> **Model Selection Context**:
> Models 1, 2, and 3 were **not included** because those models were trained with the SAM neck being **frozen**.
> **Model 4** unfreezes and trains the SAM neck alongside spatial attention pooling and a 2-layer classification head, achieving substantial improvements in feature localization and Pfirrmann grading accuracy.

---

## 1. Server Environment & File Paths

All paths are standardized for the remote server environment:

| Component | Server Path |
| :--- | :--- |
| **Main Project Root** | `/home/arindam/spider-pfirrmann-mri-core` |
| **Model 4 Training Script** | `/home/arindam/spider-pfirrmann-mri-core/training/train_classifier_v2.py` |
| **Classifier Architecture** | `/home/arindam/spider-pfirrmann-mri-core/training/models/pfirrmann_classifier_v2.py` |
| **Dataset CSV** | `/home/arindam/spider-pfirrmann-mri-core/dataset_OG.csv` *(or `data/new_final/final_data/dataset_OG.csv`)* |
| **MRI Images Directory** | `/home/arindam/spider-pfirrmann-mri-core/data/new_final/final_data/images` |
| **Best Model Checkpoint** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_4/checkpoints/best_model.pt` |
| **Latest Model Checkpoint** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_4/checkpoints/latest_checkpoint.pt` |
| **Training Log CSV** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_4/logs/training_log.csv` |
| **Inference Script** | `/home/arindam/spider-pfirrmann-mri-core/evaluation/model_4/inference_model_4.py` |

---

## 2. Output Directory & Generated Results

When the inference script completes, all generated artifacts are stored in a dedicated results directory:
```
/home/arindam/spider-pfirrmann-mri-core/evaluation/model_4/results/
```

Inside `results/`, exactly **5 primary items** are produced:

```
evaluation/model_4/results/
├── prediction_results.csv        # Detailed per-slice predictions and probability distributions
├── evaluation_summary.csv        # Macro performance metrics and per-grade summary table
├── test_predictions/             # 731 diagnostic visual montage images (one per test slice)
├── r/                            # 31 core evaluation report files (25 figures + 6 CSV tables)
└── conf/                         # 10 Grade 1–5 confidence distribution bar charts
```

### Breakdown of the 5 Result Items:

1. **`prediction_results.csv`**:
   - Contains all 731 test slice predictions.
   - Columns include: `Patient_ID`, `Slice_Number`, `Offset`, `Medial_Slice_Number`, `IVD_Label`, `Pfirrmann_Grade` (Ground Truth), `Predicted_Grade`, `Correct` (Boolean), `Error_Distance`, `Prediction_Error_Type` (*Correct*, *Adjacent Error*, *Non-Adjacent Error*), `Prediction_Confidence`, `GT_Grade_Probability`, all 5 individual class probabilities (`Prob_Grade_1` to `Prob_Grade_5`), and `Prediction_Image` path.

2. **`evaluation_summary.csv`**:
   - High-level summary report for rapid verification.
   - Contains overall metrics: Exact Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Mean Confidence, Adjacent Error Rate, Non-Adjacent Error Rate.
   - Contains per-grade breakdown: Precision, Recall, F1, Ground Truth sample count, and Predicted count for each grade.

3. **`test_predictions/` (731 Montage Images)**:
   - Contains individual image montages for every test slice (`Patient<ID>_Slice<NUM>_IVD<L>_GT<G>_Pred<P>.png`).
   - Each montage features the cropped MRI image side-by-side with an information panel containing: Patient ID, IVD Level, Slice Offset, color-coded outcome badge (Green = Correct, Orange = Adjacent Error, Red = Non-Adjacent Error), prediction confidence, ground truth probability, and full class probability bar chart.

4. **`r/` (31 Core Diagnostic Files)**:
   - **25 Diagnostic Figures**:
     - `01_confusion_matrix.png` & `02_normalized_confusion_matrix.png`
     - `03_gt_vs_predicted_distribution.png`
     - `04_per_grade_precision_recall_f1.png`
     - `05_correct_vs_incorrect.png`
     - `06_error_distance_distribution.png` & `07_error_type_distribution.png`
     - `08_confidence_distribution.png` & `09_confidence_correct_vs_incorrect.png`
     - `10_grade_wise_confidence.png` & `11_mean_class_probabilities_by_gt.png`
     - `12_calibration_reliability.png`
     - `13_patient_level_accuracy.png` & `14_offset_performance.png`
     - `15_error_pair_distribution.png`
     - `16_grade_1_correct_confidence_per_patient.png` through `25_grade_5_incorrect_confidence_per_patient.png` (10 per-patient confidence scatter plots)
   - **6 Summary CSV Tables**:
     - `per_grade_metrics.csv`, `calibration_bins.csv`, `patient_level_metrics.csv`, `offset_metrics.csv`, `error_pair_counts.csv`, `metrics_summary.csv`

5. **`conf/` (10 Confidence Distribution Charts)**:
   - Grade-by-grade confidence distribution bar charts for both correct and wrong predictions:
     - `grade_1_correct_confidence.png` & `grade_1_wrong_confidence.png`
     - `grade_2_correct_confidence.png` & `grade_2_wrong_confidence.png`
     - `grade_3_correct_confidence.png` & `grade_3_wrong_confidence.png`
     - `grade_4_correct_confidence.png` & `grade_4_wrong_confidence.png`
     - `grade_5_correct_confidence.png` & `grade_5_wrong_confidence.png`
   - Annotated with mean and median thresholds and individual sample confidence percentages.

---

## 3. How to Run Model 4 Inference

Activate your virtual environment and run the script from the repository root:

```bash
cd /home/arindam/spider-pfirrmann-mri-core
source .venv/bin/activate
```

### Option A: Standard Full Inference (Recommended)
Evaluates the best checkpoint, saves all CSVs, generates all 731 image montages, and builds all 41 diagnostic figures:
```bash
python evaluation/model_4/inference_model_4.py
```

### Option B: Evaluate Latest Checkpoint Instead
To evaluate the latest training checkpoint instead of the best model:
```bash
python evaluation/model_4/inference_model_4.py --checkpoint latest_checkpoint.pt
```

### Option C: Fast Evaluation (Skip Individual Visual Montages)
If you only need the metrics CSVs, 31 core report files, and 10 confidence charts without re-rendering all 731 individual slice montages:
```bash
python evaluation/model_4/inference_model_4.py --skip_visuals
```

### Option D: Custom Batch Size
Batch size defaults to 6 (optimized for GPU memory stability). To change:
```bash
python evaluation/model_4/inference_model_4.py --batch_size 6
```

---

## 4. Execution Performance & Estimated Time to Completion

Benchmarks measured on remote server (NVIDIA T4 / A10G GPU, 4 CPU workers):

| Phase | Description | Estimated Duration |
| :--- | :--- | :--- |
| **Model Forward Pass** | 731 single-slice test inferences (Batch Size = 6) | ~15 – 25 seconds |
| **Visual Montages** | Rendering all 731 diagnostic PNG images | ~1.5 – 2.5 minutes |
| **Core Suite (`r/`)** | Generating 25 figures + 6 CSV tables | ~25 – 35 seconds |
| **Conf Suite (`conf/`)** | Generating 10 confidence distribution charts | ~10 – 15 seconds |
| **Total Pipeline Time** | **Complete end-to-end execution** | **~2.5 – 3.5 minutes** |
| **Fast Pipeline (`--skip_visuals`)** | **CSVs + 31 `r/` + 10 `conf/`** | **~45 – 60 seconds** |

---

## 5. What to Expect from the Script

1. **Terminal Logging**:
   - Automatically detects and displays GPU specifications and available VRAM.
   - Shows progress bars (`tqdm`) for inference and visual generation.
   - Prints a formatted completion summary with:
     - Exact Test Accuracy
     - Total test slices evaluated (731)
     - Number of unique patients (149)
     - Absolute paths to all generated output folders.

2. **Verifying File Counts via Terminal**:
   Run these commands to verify that all output files are present:
   ```bash
   ls -1 evaluation/model_4/results/test_predictions | wc -l   # Expected: 731
   ls -1 evaluation/model_4/results/r | wc -l                  # Expected: 31
   ls -1 evaluation/model_4/results/conf | wc -l               # Expected: 10
   ```
   *(Note: Using `ls -1` counts files strictly. If using `ls -la | wc -l`, add +3 for directory header lines).*
