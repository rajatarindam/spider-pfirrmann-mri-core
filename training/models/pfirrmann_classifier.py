import torch
import torch.nn as nn

from mri_core.cfg import parse_args
from mri_core.models.sam import sam_model_registry

from configs.paths import MRI_CORE_CHECKPOINT_PATH


class PfirrmannClassifier(nn.Module):
    """
    MRI-CORE ViT-B + MLP classifier for Pfirrmann grading.

    Input:
        [B, 3, 1024, 1024]

    MRI-CORE output:
        [B, 256, 64, 64]

    Global average pooling:
        [B, 256]

    Classifier output:
        [B, 5]

    Classes:
        0 -> Pfirrmann Grade 1
        1 -> Pfirrmann Grade 2
        2 -> Pfirrmann Grade 3
        3 -> Pfirrmann Grade 4
        4 -> Pfirrmann Grade 5
    """

    def __init__(self):

        super().__init__()

        # --------------------------------------------------------
        # MRI-CORE configuration
        # --------------------------------------------------------

        args = parse_args()

        # We are using MRI-CORE only as a frozen feature extractor.
        args.if_encoder_adapter = False
        args.if_mask_decoder_adapter = False
        args.if_update_encoder = False

        # --------------------------------------------------------
        # Load MRI-CORE ViT-B
        # --------------------------------------------------------

        model = sam_model_registry["vit_b"](
            args=args,
            checkpoint=MRI_CORE_CHECKPOINT_PATH,
            num_classes=1,
            image_size=1024,
            pretrained_sam=False,
        )

        # Keep only the image encoder.
        self.image_encoder = model.image_encoder

        # Freeze MRI-CORE.
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # --------------------------------------------------------
        # Classifier head
        # --------------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 5),
        )

    def forward(self, images):

        # --------------------------------------------------------
        # MRI-CORE feature extraction
        # --------------------------------------------------------

        with torch.no_grad():

            feature_map = self.image_encoder(images)

        # Expected:
        # [B, 256, 64, 64]

        assert feature_map.shape[1] == 256, (
            f"Expected MRI-CORE feature dimension 256, "
            f"got {feature_map.shape[1]}"
        )

        # --------------------------------------------------------
        # Global Average Pooling
        # --------------------------------------------------------

        feature_vector = feature_map.mean(
            dim=(2, 3)
        )

        # Expected:
        # [B, 256]

        assert feature_vector.shape[1] == 256, (
            f"Expected pooled feature dimension 256, "
            f"got {feature_vector.shape[1]}"
        )

        # --------------------------------------------------------
        # Pfirrmann classification
        # --------------------------------------------------------

        output = self.classifier(
            feature_vector
        )

        # Expected:
        # [B, 5]

        assert output.shape[1] == 5, (
            f"Expected 5 Pfirrmann classes, "
            f"got {output.shape[1]}"
        )

        return output