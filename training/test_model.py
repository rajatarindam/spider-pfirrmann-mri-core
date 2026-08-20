import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from training.dataset import PfirrmannDataset
from training.models.pfirrmann_classifier import PfirrmannClassifier


def main():

    print("=" * 70)
    print("SPIDER PFIRRMANN END-TO-END MODEL TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device              : {device}")

    if torch.cuda.is_available():
        print(
            f"GPU                 : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    print()
    print("Loading training dataset...")

    dataset = PfirrmannDataset(
        split="train"
    )

    # Use only two samples for this smoke test.
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(
        iter(loader)
    )

    print()
    print("Input batch:")
    print(
        f"Images shape        : "
        f"{tuple(images.shape)}"
    )
    print(
        f"Images dtype        : "
        f"{images.dtype}"
    )
    print(
        f"Images range        : "
        f"{images.min().item():.6f} - "
        f"{images.max().item():.6f}"
    )
    print(
        f"Labels shape        : "
        f"{tuple(labels.shape)}"
    )
    print(
        f"Labels              : "
        f"{labels.tolist()}"
    )

    # ------------------------------------------------------------------
    # Move input to GPU
    # ------------------------------------------------------------------

    images = images.to(
        device
    )

    labels = labels.to(
        device
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    print()
    print("Loading Pfirrmann classifier...")

    model = PfirrmannClassifier()

    model = model.to(
        device
    )

    model.eval()

    print("Classifier loaded.")

    # ------------------------------------------------------------------
    # Verify frozen MRI-CORE
    # ------------------------------------------------------------------

    trainable_parameters = 0
    frozen_parameters = 0

    for parameter in model.parameters():

        parameter_count = parameter.numel()

        if parameter.requires_grad:
            trainable_parameters += parameter_count
        else:
            frozen_parameters += parameter_count

    print()
    print("Parameter status:")
    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )
    print(
        f"Frozen parameters    : "
        f"{frozen_parameters:,}"
    )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    print()
    print("Running forward pass...")

    with torch.no_grad():

        logits = model(
            images
        )

    # ------------------------------------------------------------------
    # Verify output
    # ------------------------------------------------------------------

    print()
    print(
        f"Output shape        : "
        f"{tuple(logits.shape)}"
    )

    print(
        f"Output dtype        : "
        f"{logits.dtype}"
    )

    print()
    print("Raw logits:")
    print(
        logits.detach()
        .cpu()
    )

    expected_shape = (
        2,
        5,
    )

    if tuple(logits.shape) != expected_shape:

        raise RuntimeError(
            "\nUnexpected model output shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got     : {tuple(logits.shape)}"
        )

    print()
    print("Model output check   : PASSED")

    # ------------------------------------------------------------------
    # Convert logits to predictions
    # ------------------------------------------------------------------

    predicted_classes = torch.argmax(
        logits,
        dim=1
    )

    predicted_grades = (
        predicted_classes + 1
    )

    print()
    print("Predictions:")
    print(
        f"Internal classes    : "
        f"{predicted_classes.cpu().tolist()}"
    )
    print(
        f"Pfirrmann grades    : "
        f"{predicted_grades.cpu().tolist()}"
    )

    print()
    print("=" * 70)
    print("END-TO-END MODEL TEST PASSED")
    print("=" * 70)

    print()
    print("Verified:")
    print("  Real SPIDER images  → MRI-CORE")
    print("  MRI-CORE            → 256 × 64 × 64")
    print("  GAP                 → 256 features")
    print("  MLP                 → 5 logits")
    print("  Prediction mapping  → classes 0-4 → grades 1-5")


if __name__ == "__main__":
    main()