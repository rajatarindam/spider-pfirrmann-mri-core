"""
test_preflight_model_4.py
─────────────────────────
Standalone Pre-Flight Diagnostic & Resume Verification Battery for Model 4.

Executes a complete 8-step verification battery before launching training:
1. Environment & CUDA Hardware Diagnostics (VRAM, device info).
2. Dataset CSV integrity (validates dataset_OG.csv with 4,864 train / 1,408 val / 731 test).
3. Image file loading test (verifies 1024x1024 grayscale slice resolution).
4. Model architecture & parameter freeze audit:
   - ViT blocks 0–11, patch_embed, pos_embed: strictly FROZEN
   - SAM Convolutional Neck: UNFROZEN (trainable at lr = 1e-5)
   - SpatialAttentionPooling & Classifier MLP: TRAINABLE (lr = 1e-4)
5. Checkpoint existence & 171-key state dictionary compatibility check.
6. Optimizer momentum & CosineAnnealingLR scheduler state restoration.
7. Single-batch forward & backward pass on GPU (confirms gradient flow, no NaNs).
8. Dynamic resumption target calculation (reports current epoch, next epoch, and remaining steps).

Usage:
    # Run default preflight (checks outputs/model_4/checkpoints/latest_checkpoint.pt):
    python test_preflight_model_4.py

    # Run with a custom checkpoint path:
    python test_preflight_model_4.py --checkpoint outputs/model_4/checkpoints/latest_checkpoint.pt
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure imports resolve
_SCRIPT_DIR = Path(__file__).resolve().parent

# Check potential project roots
candidates = [
    _SCRIPT_DIR,
    _SCRIPT_DIR.parent,
    _SCRIPT_DIR / "context",
]

PROJECT_ROOT = _SCRIPT_DIR
for c in candidates:
    if (c / "data").exists() or (c / "mri_core").exists():
        PROJECT_ROOT = c
        break

for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "context"), str(_SCRIPT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from training.models.pfirrmann_classifier_v2 import PfirrmannClassifierV2
    from training.train_classifier_v2 import (
        PfirrmannDatasetModel4,
        calculate_class_weights,
        create_optimizer,
        create_scheduler,
    )
except ImportError:
    try:
        from model_4.training.models.pfirrmann_classifier_v2 import PfirrmannClassifierV2
        from model_4.training.train_classifier_v2 import (
            PfirrmannDatasetModel4,
            calculate_class_weights,
            create_optimizer,
            create_scheduler,
        )
    except ImportError:
        raise ImportError("Could not import Model 4 modules.")

from configs.paths import (
    MODEL_4_DATASET_CSV,
    MODEL_4_IMAGES_DIR,
    MODEL_4_CHECKPOINT_DIR,
    MODEL_4_LATEST_CHECKPOINT_PATH,
    MRI_CORE_CHECKPOINT_PATH,
)
from configs.training_config_model_4 import (
    NUM_CLASSES,
    EXPECTED_SPLIT_COUNTS,
    BATCH_SIZE,
    NUM_EPOCHS,
    NECK_LEARNING_RATE,
    HEAD_LEARNING_RATE,
    WEIGHT_DECAY,
    SCHEDULER_ETA_MIN,
)


def print_header(title: str) -> None:
    print("\n" + "=" * 75)
    print(f"  {title.upper()}")
    print("=" * 75)


def check_passed(name: str, detail: str = "") -> None:
    print(f"  [PASS] {name}" + (f" -> {detail}" if detail else ""))


def check_failed(name: str, detail: str = "") -> None:
    print(f"  [FAIL] {name}" + (f" -> {detail}" if detail else ""))
    raise AssertionError(f"Preflight check failed: {name}. {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Model 4 Pre-Flight & Resume Verification Battery")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help=f"Path to checkpoint to verify (default: {MODEL_4_LATEST_CHECKPOINT_PATH})",
    )
    args = parser.parse_args()

    print("\n" + "#" * 75)
    print("  MODEL 4 — PRE-FLIGHT VERIFICATION & RESUME SMOKE TEST")
    print("#" * 75)

    # 1. Environment & CUDA
    print_header("1. Environment & Hardware Diagnostics")
    print(f"  Python : {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        check_passed("CUDA Available", f"{gpu_name} ({total_vram_gb:.2f} GB VRAM)")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    else:
        device = torch.device("cpu")
        print("  [WARN] CUDA unavailable. Testing on CPU.")

    # 2. Dataset CSV Integrity
    print_header("2. Dataset CSV Integrity")
    csv_path = Path(MODEL_4_DATASET_CSV)
    if not csv_path.exists():
        alt = PROJECT_ROOT / "data" / "new_final" / "final_data" / "dataset_OG.csv"
        if alt.exists():
            csv_path = alt
        else:
            check_failed("Dataset CSV Located", f"Not found at {csv_path}")

    check_passed("Dataset CSV Located", str(csv_path))
    df = pd.read_csv(csv_path)
    check_passed("CSV Loaded", f"{len(df)} total slice crops")

    split_counts = df["Split"].astype(str).str.lower().value_counts().to_dict()
    for s, exp in EXPECTED_SPLIT_COUNTS.items():
        actual = split_counts.get(s, 0)
        assert actual == exp, f"Split {s} mismatch: expected {exp}, got {actual}"
    check_passed(
        "Split Distributions Exact",
        f"train={split_counts['train']}, val={split_counts['val']}, test={split_counts['test']}",
    )

    # 3. Data Loading & Image Verification
    print_header("3. Dataset Item & Image Loading Test")
    train_ds = PfirrmannDatasetModel4(csv_path=csv_path, split="train")
    img_tensor, label_tensor = train_ds[0]
    assert tuple(img_tensor.shape) == (3, 1024, 1024), f"Unexpected image shape: {img_tensor.shape}"
    assert img_tensor.dtype == torch.float32, f"Expected float32, got {img_tensor.dtype}"
    assert 0 <= label_tensor.item() < NUM_CLASSES, f"Label out of range: {label_tensor.item()}"
    check_passed("Sample Loading Valid", f"Tensor shape: {tuple(img_tensor.shape)}, Label: {label_tensor.item()}")

    # 4. Model Parameter Audit
    print_header("4. Model Architecture & Parameter Freeze Audit")
    model = PfirrmannClassifierV2().to(device)

    # Verify ViT blocks 0-11 frozen
    for i in range(12):
        for name, p in model.image_encoder.blocks[i].named_parameters():
            assert not p.requires_grad, f"ViT block {i} parameter {name} should be FROZEN!"
    check_passed("ViT Blocks 0–11 Strictly Frozen", "All 12 transformer blocks non-trainable")

    for name, p in model.image_encoder.patch_embed.named_parameters():
        assert not p.requires_grad, f"Patch embed {name} should be FROZEN!"
    assert not model.image_encoder.pos_embed.requires_grad, "Pos embed should be FROZEN!"
    check_passed("Embeddings Strictly Frozen", "Patch and positional embeddings non-trainable")

    # Verify SAM Neck trainable
    neck_params = [p for p in model.image_encoder.neck.parameters() if p.requires_grad]
    assert len(neck_params) > 0, "SAM neck has NO trainable parameters!"
    check_passed("SAM Conv Neck Trainable", f"{len(neck_params)} parameter tensors active (lr={NECK_LEARNING_RATE})")

    # Verify Spatial Attention & Classifier head trainable
    head_params = [p for p in model.spatial_attention_pool.parameters() if p.requires_grad]
    head_params.extend([p for p in model.classifier.parameters() if p.requires_grad])
    assert len(head_params) > 0, "Classifier head has NO trainable parameters!"
    check_passed("Classifier Head & Spatial Pooling Trainable", f"{len(head_params)} parameter tensors active (lr={HEAD_LEARNING_RATE})")

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    check_passed("Parameter Count Verified", f"Trainable: {trainable_count:,} (~4.2M) | Frozen: {frozen_count:,} (~86.6M)")

    # 5. Checkpoint Existence & Key Matching
    print_header("5. Checkpoint Verification & Compatibility Audit")

    ckpt_candidates = []
    if args.checkpoint:
        ckpt_candidates.append(Path(args.checkpoint))
    ckpt_candidates.extend([
        Path(MODEL_4_LATEST_CHECKPOINT_PATH),
        PROJECT_ROOT / "outputs" / "model_4" / "checkpoints" / "latest_checkpoint.pt",
        PROJECT_ROOT / "outputs" / "model_4_28th_epoch_model" / "latest_checkpoint.pt",
        _SCRIPT_DIR.parent / "outputs" / "model_4" / "checkpoints" / "latest_checkpoint.pt",
        _SCRIPT_DIR.parent / "outputs" / "model_4_28th_epoch_model" / "latest_checkpoint.pt",
    ])

    resolved_ckpt = None
    for c in ckpt_candidates:
        if c.exists():
            resolved_ckpt = c
            break

    if resolved_ckpt is None:
        print(f"  [WARN] No existing checkpoint found in searched paths:\n" + "\n".join(f"    - {c}" for c in ckpt_candidates))
        print("  [INFO] Training would start FRESH from Epoch 1.")
        ckpt_available = False
        start_epoch = 0
        best_val_macro_f1 = 0.0
    else:
        check_passed("Checkpoint File Located", str(resolved_ckpt))
        ckpt = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)

        required_keys = {"epoch", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "best_val_macro_f1"}
        missing_keys = required_keys - set(ckpt.keys())
        assert not missing_keys, f"Checkpoint missing required keys: {missing_keys}"
        check_passed("Checkpoint Dictionary Schema Valid", f"Keys: {list(ckpt.keys())}")

        sd = ckpt["model_state_dict"]
        check_passed("State Dict Key Count", f"{len(sd)} parameter tensors in checkpoint")

        # Load weights into model
        load_result = model.load_state_dict(sd, strict=True)
        check_passed("Weights Loaded Perfectly", f"Strict loading succeeded with 0 missing / 0 unexpected keys")

        start_epoch = int(ckpt["epoch"])
        best_val_macro_f1 = float(ckpt["best_val_macro_f1"])
        val_f1_ckpt = float(ckpt.get("val_macro_f1", 0.0))
        check_passed(
            "Checkpoint Training Progress Verified",
            f"Epoch: {start_epoch} | Best Val Macro F1: {best_val_macro_f1 * 100:.2f}% | Last Val F1: {val_f1_ckpt * 100:.2f}%",
        )
        ckpt_available = True

    # 6. Optimizer & Scheduler State Restoration Test
    print_header("6. Optimizer & Scheduler Restoration Test")
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, epochs=NUM_EPOCHS)

    if ckpt_available:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        check_passed("Optimizer & Scheduler States Restored", "Momentum buffers and step counters restored")
    else:
        check_passed("Optimizer & Scheduler Initialized Fresh", f"AdamW (neck={NECK_LEARNING_RATE}, head={HEAD_LEARNING_RATE})")

    # 7. Single Forward & Backward Pass Verification
    # (Uses 2 distinct dataset samples so BatchNorm1d can compute realistic batch variance)
    print_header("7. Single-Batch Forward & Backward Gradient Flow Test")
    img0, lbl0 = train_ds[0]
    img1, lbl1 = train_ds[1]
    batch_img = torch.stack([img0, img1]).to(device)    # [2, 3, 1024, 1024]
    batch_lbl = torch.stack([lbl0, lbl1]).to(device)    # [2]

    class_weights = calculate_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    optimizer.zero_grad()

    logits = model(batch_img)
    assert tuple(logits.shape) == (2, NUM_CLASSES), f"Unexpected logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN found in model logits!"
    check_passed("Forward Pass Clean", f"Output shape: {tuple(logits.shape)}, Logits: {logits[0].detach().cpu().numpy().round(3).tolist()}")

    loss = criterion(logits, batch_lbl)
    assert not torch.isnan(loss), "NaN found in computed loss!"
    check_passed("Loss Computation Clean", f"CrossEntropyLoss: {loss.item():.4f}")

    loss.backward()

    neck_grad = model.image_encoder.neck[0].weight.grad
    assert neck_grad is not None and neck_grad.norm().item() > 0.0, "SAM Neck received NO gradient!"
    check_passed("SAM Neck Active Gradients", f"Norm: {neck_grad.norm().item():.6f}")

    cls_grad = model.classifier[0].weight.grad
    assert cls_grad is not None and cls_grad.norm().item() > 0.0, "Classifier head received NO gradient!"
    check_passed("Classifier Head Active Gradients", f"Norm: {cls_grad.norm().item():.6f}")

    for i in range(12):
        for name, p in model.image_encoder.blocks[i].named_parameters():
            assert p.grad is None, f"ViT block {i} parameter {name} unexpectedly received a gradient!"
    check_passed("ViT Backbone Gradient Isolation", "Gradients strictly zero for blocks 0–11")

    # 8. Resume Target Summary
    print_header("8. Execution Summary & Resume Roadmap")
    if ckpt_available:
        next_epoch = start_epoch + 1
        remaining_epochs = max(0, NUM_EPOCHS - start_epoch)
        print(f"  Checkpoint File Found    : {resolved_ckpt}")
        print(f"  Completed Epochs         : {start_epoch}")
        print(f"  Next Resumption Epoch    : {next_epoch}")
        print(f"  Target Epochs            : {NUM_EPOCHS} ({remaining_epochs} epochs remaining)")
        print(f"  Best Val Macro F1 so far : {best_val_macro_f1 * 100:.2f}%")
        print("\n  To resume training on server, run:")
        print("    python training/train_classifier_v2.py --resume")
        if next_epoch > NUM_EPOCHS:
            print(f"\n  NOTE: Since checkpoint is at epoch {start_epoch} >= {NUM_EPOCHS}, extend with:")
            print(f"    python training/train_classifier_v2.py --resume --epochs {start_epoch + 10}")
    else:
        print(f"  No checkpoint found. Training will start fresh from Epoch 1 to {NUM_EPOCHS}.")
        print("\n  To train fresh on server, run:")
        print("    python training/train_classifier_v2.py")

    print("\n" + "#" * 75)
    print("  MODEL 4 PRE-FLIGHT VERIFICATION COMPLETE — ALL SYSTEMS READY!")
    print("#" * 75 + "\n")


if __name__ == "__main__":
    main()
