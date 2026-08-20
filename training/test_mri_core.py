import sys
from pathlib import Path

import torch

# ----------------------------------------------------------------------
# Make sure the project root is available to Python
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from mri_core.cfg import parse_args
from mri_core.models.sam import sam_model_registry

from configs.paths import MRI_CORE_CHECKPOINT_PATH


def main():

    print("=" * 70)
    print("MRI-CORE LOADING TEST")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device              : {device}")

    if torch.cuda.is_available():
        print(f"GPU                 : {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # Check checkpoint
    # ------------------------------------------------------------------

    checkpoint_path = Path(MRI_CORE_CHECKPOINT_PATH)

    print(f"Checkpoint          : {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"MRI-CORE checkpoint not found:\n{checkpoint_path}"
        )

    print(
        f"Checkpoint Size     : "
        f"{checkpoint_path.stat().st_size / (1024 ** 2):.2f} MB"
    )

    # ------------------------------------------------------------------
    # MRI-CORE configuration
    # ------------------------------------------------------------------

    args = parse_args()

    args.arch = "vit_b"
    args.image_size = 1024

    # Feature extraction mode.
    args.if_encoder_adapter = False
    args.if_mask_decoder_adapter = False
    args.if_update_encoder = False

    # --------------------------MRICORE_CHECKPOINT_PATH----------------------------------------
    # Load MRI-CORE
    # ------------------------------------------------------------------

    print()
    print("Loading MRI-CORE ViT-B...")

    model = sam_model_registry["vit_b"](
        args=args,
        checkpoint=str(checkpoint_path),
        num_classes=1,
        image_size=1024,
        pretrained_sam=False,
    )

    print("MRI-CORE loaded successfully.")

    # ------------------------------------------------------------------
    # Extract image encoder
    # ------------------------------------------------------------------

    image_encoder = model.image_encoder

    image_encoder.eval()
    image_encoder.to(device)

    # ------------------------------------------------------------------
    # Create dummy input
    # ------------------------------------------------------------------

    # MRI-CORE expects:
    # [B, C, H, W]
    #
    # We use:
    # B = 1
    # C = 3
    # H = 1024
    # W = 1024
    #
    # Values are already in [0, 1].

    dummy_image = torch.rand(
        1,
        3,
        1024,
        1024,
        dtype=torch.float32,
        device=device,
    )

    print()
    print(f"Input Shape         : {tuple(dummy_image.shape)}")
    print(f"Input dtype         : {dummy_image.dtype}")
    print(
        f"Input range         : "
        f"{dummy_image.min().item():.4f} - "
        f"{dummy_image.max().item():.4f}"
    )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    print()
    print("Running MRI-CORE image encoder...")

    with torch.no_grad():

        feature_map = image_encoder(
            dummy_image
        )

    # ------------------------------------------------------------------
    # Verify output
    # ------------------------------------------------------------------

    print()
    print(f"Feature Map Shape    : {tuple(feature_map.shape)}")
    print(f"Feature Map dtype    : {feature_map.dtype}")

    expected_shape = (
        1,
        256,
        64,
        64,
    )

    if tuple(feature_map.shape) != expected_shape:

        raise RuntimeError(
            "\nUnexpected MRI-CORE feature shape.\n"
            f"Expected: {expected_shape}\n"
            f"Got     : {tuple(feature_map.shape)}"
        )

    print()
    print("Feature shape check  : PASSED")

    # ------------------------------------------------------------------
    # Global Average Pooling
    # ------------------------------------------------------------------

    feature_vector = feature_map.mean(
        dim=(2, 3)
    )

    print(
        f"Pooled Feature Shape : "
        f"{tuple(feature_vector.shape)}"
    )

    expected_vector_shape = (
        1,
        256,
    )

    if tuple(feature_vector.shape) != expected_vector_shape:

        raise RuntimeError(
            "\nUnexpected pooled feature shape.\n"
            f"Expected: {expected_vector_shape}\n"
            f"Got     : {tuple(feature_vector.shape)}"
        )

    print("Pooling check        : PASSED")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("MRI-CORE TEST PASSED")
    print("=" * 70)

    print()
    print("Pipeline verified:")
    print("  Input              : [1, 3, 1024, 1024]")
    print("  MRI-CORE ViT-B     : frozen")
    print("  Feature map        : [1, 256, 64, 64]")
    print("  Global Average Pool: [1, 256]")
    print()


if __name__ == "__main__":
    main()