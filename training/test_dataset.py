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


def test_split(split):

    print()
    print("=" * 70)
    print(f"TESTING {split.upper()} DATASET")
    print("=" * 70)

    dataset = PfirrmannDataset(
        split=split
    )

    print(f"Dataset length      : {len(dataset)}")

    # ------------------------------------------------------------------
    # Load first sample
    # ------------------------------------------------------------------

    image, label = dataset[0]

    print()
    print("First sample:")
    print(f"Image shape         : {tuple(image.shape)}")
    print(f"Image dtype         : {image.dtype}")
    print(
        f"Image range         : "
        f"{image.min().item():.6f} - "
        f"{image.max().item():.6f}"
    )
    print(f"Label               : {label.item()}")

    grade = dataset.LABEL_TO_GRADE[
        label.item()
    ]

    print(f"Pfirrmann Grade     : {grade}")

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    assert image.shape == (
        3,
        1024,
        1024,
    ), (
        f"Unexpected image shape: "
        f"{tuple(image.shape)}"
    )

    assert image.dtype == torch.float32

    assert (
        image.min().item() >= 0.0
    )

    assert (
        image.max().item() <= 1.0
    )

    assert label.dtype == torch.long

    assert (
        label.item() in range(5)
    )

    assert (
        grade in range(1, 6)
    )

    print()
    print("Single-sample checks : PASSED")

    # ------------------------------------------------------------------
    # DataLoader test
    # ------------------------------------------------------------------

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
    print("First batch:")
    print(f"Images shape        : {tuple(images.shape)}")
    print(f"Images dtype        : {images.dtype}")
    print(f"Labels shape        : {tuple(labels.shape)}")
    print(f"Labels dtype        : {labels.dtype}")
    print(f"Labels              : {labels.tolist()}")

    # ------------------------------------------------------------------
    # Batch assertions
    # ------------------------------------------------------------------

    assert images.shape == (
        2,
        3,
        1024,
        1024,
    )

    assert images.dtype == torch.float32

    assert labels.shape == (
        2,
    )

    assert labels.dtype == torch.long

    print()
    print("DataLoader checks    : PASSED")


def main():

    print("=" * 70)
    print("SPIDER PFIRRMANN DATASET SMOKE TEST")
    print("=" * 70)

    for split in (
        "train",
        "val",
        "test",
    ):

        test_split(split)

    print()
    print("=" * 70)
    print("DATASET SMOKE TEST PASSED")
    print("=" * 70)

    print()
    print("Verified:")
    print("  Train dataset      : OK")
    print("  Validation dataset : OK")
    print("  Test dataset       : OK")
    print("  Image shape        : [3, 1024, 1024]")
    print("  Image dtype        : float32")
    print("  Image range        : [0, 1]")
    print("  Label dtype        : long")
    print("  Internal labels    : 0-4")
    print("  Pfirrmann grades   : 1-5")


if __name__ == "__main__":
    main()