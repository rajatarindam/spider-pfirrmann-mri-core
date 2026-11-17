# Model 5 — Multi-View 3D Pfirrmann Classifier Inference Guide

> [!NOTE]
> **Model Selection Context**:
> Models 1, 2, and 3 were **not included** because those models were trained with the SAM neck being **frozen**.
> **Model 5** operates on **5 contiguous sagittal slices** per intervertebral disc volume (offsets: `[-2, -1, 0, +1, +2]`). It combines the frozen MRI-CORE ViT-B backbone with an unfrozen SAM neck, learned 1D view positional embeddings, spatial attention pooling, dynamic view attention aggregation, and dual output heads (primary multi-view classifier and auxiliary center-slice supervisor).

---

## 1. Server Environment & File Paths

All paths are standardized for the remote server environment:

| Component | Server Path |
| :--- | :--- |
| **Main Project Root** | `/home/arindam/spider-pfirrmann-mri-core` |
| **Model 5 Training Script** | `/home/arindam/spider-pfirrmann-mri-core/training/train_classifier_v7.py` |
| **Classifier Architecture** | `/home/arindam/spider-pfirrmann-mri-core/training/models/pfirrmann_classifier_v5.py` |
| **Dataset CSV** | `/home/arindam/spider-pfirrmann-mri-core/data/new_final/dataset_multi_view_v2.csv` *(or `data/new_final/final_data/dataset_multi_view_v2.csv`)* |
| **MRI Images Directory** | `/home/arindam/spider-pfirrmann-mri-core/data/new_final/final_data/images` |
| **Best Model Checkpoint** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/checkpoints/best_model.pt` |
| **Latest Model Checkpoint** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/checkpoints/latest_checkpoint.pt` |
| **Training Log CSV** | `/home/arindam/spider-pfirrmann-mri-core/outputs/model_5/phase1_2/logs/training_log.csv` |
| **Inference Script** | `/home/arindam/spider-pfirrmann-mri-core/evaluation/model_5/inference_model_5.py` |

---

## 2. Output Directory & Generated Results

When the inference script completes, all generated artifacts are stored in a dedicated results directory:
```
/home/arindam/spider-pfirrmann-mri-core/evaluation/model_5/results/
```

Inside `results/`, exactly **5 primary items** are produced:

```
evaluation/model_5/results/
├── prediction_results.csv        # Detailed per-disc predictions, fused confidence, auxiliary center grade
├── evaluation_summary.csv        # Macro performance metrics and per-grade summary table
├── test_predictions/             # 149 panoramic 5-slice composite visual montages (one per test disc)
├── r/                            # 31 core evaluation report files (25 figures + 6 CSV tables)
└── conf/                         # 10 Grade 1–5 confidence distribution bar charts
```

### Breakdown of the 5 Result Items:

1. **`prediction_results.csv`**:
   - Contains evaluations for all 149 test 3D disc volumes.
   - Columns include: `Patient_ID`, `IVD_Label`, `Pfirrmann_Grade` (Ground Truth), `Predicted_Grade` (Multi-View Fused), `Correct` (Boolean), `Error_Distance`, `Prediction_Error_Type` (*Correct*, *Adjacent Error*, *Non-Adjacent Error*), `Prediction_Confidence`, `GT_Grade_Probability`, `Center_Predicted_Grade` (Auxiliary Head), `Center_Agreement` (Boolean), all 5 individual class probabilities (`Prob_Grade_1` to `Prob_Grade_5`), image paths for all 5 slices (`Image_-2` through `Image_2`), and `Prediction_Image` path.

2. **`evaluation_summary.csv`**:
   - High-level summary report for rapid verification.
   - Contains overall metrics: Exact Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, Mean Confidence, Mean GT Probability, Mean Error Distance, Adjacent Error Count, Non-Adjacent Error Count.
   - Contains per-grade breakdown: Precision, Recall, F1, Ground Truth sample count, and Predicted count for each grade.

3. **`test_predictions/` (149 5-Slice Composite Montages)**:
   - Contains panoramic composite montages for every test disc (`Patient<ID>_IVD<L>_GT<G>_Pred<P>.png`).
   - Each montage displays **all 5 contiguous sagittal slices** side-by-side (`-2`, `-1`, `0`, `+1`, `+2`), with:
     - Prominent blue outline highlighting the anatomical center slice (`Offset 0`).
     - Header banner with Patient ID, IVD Level, and color-coded outcome badge (Green = Correct, Orange = Adjacent Error, Red = Non-Adjacent Error).
     - Multi-view fused prediction confidence and full 5-grade probability distribution.
     - Auxiliary center-slice head prediction and agreement status (*MATCH* / *DISAGREE*).

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
     - `13_patient_level_accuracy.png`
     - `14_multi_view_vs_center_agreement.png` (Measures multi-view fusion vs. single center slice)
     - `15_error_pair_distribution.png`
     - `16_grade_1_correct_confidence_per_patient.png` through `25_grade_5_incorrect_confidence_per_patient.png` (10 per-patient confidence scatter plots)
   - **6 Summary CSV Tables**:
     - `per_grade_metrics.csv`, `calibration_bins.csv`, `patient_level_metrics.csv`, `multi_view_agreement_metrics.csv`, `error_pair_counts.csv`, `metrics_summary.csv`

5. **`conf/` (10 Confidence Distribution Charts)**:
   - Grade-by-grade confidence distribution bar charts for both correct and wrong predictions:
     - `grade_1_correct_confidence.png` & `grade_1_wrong_confidence.png`
     - `grade_2_correct_confidence.png` & `grade_2_wrong_confidence.png`
     - `grade_3_correct_confidence.png` & `grade_3_wrong_confidence.png`
     - `grade_4_correct_confidence.png` & `grade_4_wrong_confidence.png`
     - `grade_5_correct_confidence.png` & `grade_5_wrong_confidence.png`
   - Annotated with mean and median thresholds and individual sample confidence percentages.

---

## 3. How to Run Model 5 Inference

Activate your virtual environment and run the script from the repository root:

```bash
cd /home/arindam/spider-pfirrmann-mri-core
source .venv/bin/activate
```

### Option A: Standard Full Inference (Recommended)
Evaluates the best checkpoint, saves all CSVs, generates all 149 5-slice composite montages, and builds all 41 diagnostic figures:
```bash
python evaluation/model_5/inference_model_5.py
```

### Option B: Evaluate Latest Checkpoint Instead
To evaluate the latest training checkpoint instead of the best model:
```bash
python evaluation/model_5/inference_model_5.py --checkpoint latest_checkpoint.pt
```

### Option C: Fast Evaluation (Skip Individual Visual Montages)
If you only need the metrics CSVs, 31 core report files, and 10 confidence charts without rendering the 149 5-slice montages:
```bash
python evaluation/model_5/inference_model_5.py --skip_visuals
```

### Option D: Batch Size Control
Batch size defaults to 1 (dictated by 5 slices × 1024×1024 ViT attention memory):
```bash
python evaluation/model_5/inference_model_5.py --batch_size 1
```

---

## 4. Execution Performance & Estimated Time to Completion

Benchmarks measured on remote server (NVIDIA T4 / A10G GPU, 2 CPU workers):

| Phase | Description | Estimated Duration |
| :--- | :--- | :--- |
| **Model Forward Pass** | 149 multi-view disc evaluations (745 slices, Batch Size = 1) | ~20 – 30 seconds |
| **5-View Visual Montages** | Assembling and rendering 149 5-slice composite PNGs | ~40 – 60 seconds |
| **Core Suite (`r/`)** | Generating 25 figures + 6 CSV tables | ~20 – 30 seconds |
| **Conf Suite (`conf/`)** | Generating 10 confidence distribution charts | ~10 – 15 seconds |
| **Total Pipeline Time** | **Complete end-to-end execution** | **~1.5 – 2.5 minutes** |
| **Fast Pipeline (`--skip_visuals`)** | **CSVs + 31 `r/` + 10 `conf/`** | **~35 – 50 seconds** |

---

## 5. What to Expect from the Script

1. **Terminal Logging**:
   - Automatically detects GPU hardware and VRAM.
   - Confirms strict matching of all 174 state_dict parameter keys.
   - Displays real-time progress bars (`tqdm`) for forward pass and image rendering.
   - Prints a formatted completion summary with:
     - Exact Test Accuracy
     - Total test disc volumes (149)
     - Unique test patients
     - Absolute paths to all generated output folders.

2. **Verifying File Counts via Terminal**:
   Run these commands to verify that all output files are present:
   ```bash
   ls -1 evaluation/model_5/results/test_predictions | wc -l   # Expected: 149
   ls -1 evaluation/model_5/results/r | wc -l                  # Expected: 31
   ls -1 evaluation/model_5/results/conf | wc -l               # Expected: 10
   ```
   *(Note: Using `ls -1` counts files strictly. If using `ls -la | wc -l`, add +3 for directory header lines).*
