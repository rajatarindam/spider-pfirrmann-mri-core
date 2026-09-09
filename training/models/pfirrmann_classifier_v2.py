import torch
import torch.nn as nn
import torch.nn.functional as F

from mri_core.cfg import parse_args
from mri_core.models.sam import sam_model_registry
from configs.paths import MRI_CORE_CHECKPOINT_PATH


class SpatialAttentionPooling(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.attention = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, feature_map):
        scores = self.attention(feature_map)  # [B,1,64,64]
        b, _, h, w = scores.shape
        weights = F.softmax(scores.view(b, -1), dim=1).view(b, 1, h, w)
        return (feature_map * weights).sum(dim=(2, 3))  # [B,256]


class PfirrmannClassifierV2(nn.Module):
    """
    MRI-CORE ViT-B + trainable neck + spatial attention + MLP.

    Frozen: patch embedding, positional embedding, ViT blocks 0-11.
    Trainable: MRI-CORE neck, spatial attention, classification head.
    """

    def __init__(self):
        super().__init__()

        args = parse_args()
        args.arch = "vit_b"
        args.image_size = 1024
        args.if_encoder_adapter = False
        args.if_mask_decoder_adapter = False
        args.if_update_encoder = False

        model = sam_model_registry["vit_b"](
            args=args,
            checkpoint=MRI_CORE_CHECKPOINT_PATH,
            num_classes=1,
            image_size=1024,
            pretrained_sam=False,
        )

        self.image_encoder = model.image_encoder

        # Do not modify build_sam.py. Correct pos_embed directly here.
        self._load_corrected_positional_embedding(MRI_CORE_CHECKPOINT_PATH)

        # Freeze only the pretrained ViT feature extractor.
        for param in self.image_encoder.patch_embed.parameters():
            param.requires_grad = False

        self.image_encoder.pos_embed.requires_grad = False

        # Freeze the final ViT normalization layer when present.
        if hasattr(self.image_encoder, "norm") and self.image_encoder.norm is not None:
            for param in self.image_encoder.norm.parameters():
                param.requires_grad = False

        for block in self.image_encoder.blocks:
            for param in block.parameters():
                param.requires_grad = False

        # The MRI-CORE neck is deliberately trainable.
        for param in self.image_encoder.neck.parameters():
            param.requires_grad = True

        self.spatial_attention_pool = SpatialAttentionPooling(256)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 5),
        )

    @torch.no_grad()
    def _load_corrected_positional_embedding(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        if "teacher" in checkpoint:
            state_dict = checkpoint["teacher"]
        elif "student" in checkpoint:
            state_dict = checkpoint["student"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        key = "backbone.pos_embed"
        if key not in state_dict:
            raise KeyError(f"Could not find '{key}' in MRI-CORE checkpoint.")

        pos = state_dict[key].float()
        if tuple(pos.shape) != (1, 197, 768):
            raise ValueError(
                f"Expected checkpoint pos_embed [1,197,768], got {tuple(pos.shape)}"
            )

        # Remove CLS: [1,197,768] -> [1,196,768]
        pos = pos[:, 1:, :]
        # [1,196,768] -> [1,14,14,768]
        pos = pos.view(1, 14, 14, 768)
        # Interpolate spatial dimensions: 14x14 -> 64x64
        pos = pos.permute(0, 3, 1, 2)
        pos = F.interpolate(
            pos, size=(64, 64), mode="bilinear", align_corners=False
        )
        # [1,768,64,64] -> [1,64,64,768]
        pos = pos.permute(0, 2, 3, 1).contiguous()

        if tuple(pos.shape) != (1, 64, 64, 768):
            raise RuntimeError(f"Corrected pos_embed has shape {tuple(pos.shape)}")

        if self.image_encoder.pos_embed is None:
            raise RuntimeError("MRI-CORE image encoder has no positional embedding.")

        if tuple(self.image_encoder.pos_embed.shape) != (1, 64, 64, 768):
            raise RuntimeError(
                "Model pos_embed has unexpected shape: "
                f"{tuple(self.image_encoder.pos_embed.shape)}"
            )

        self.image_encoder.pos_embed.copy_(
            pos.to(
                device=self.image_encoder.pos_embed.device,
                dtype=self.image_encoder.pos_embed.dtype,
            )
        )

    def _frozen_vit_forward(self, images):
        # The patch embedding, positional embedding and all ViT blocks are frozen.
        # Keeping this portion under no_grad prevents autograd from storing the
        # large ViT computation graph.
        with torch.no_grad():
            x = self.image_encoder.patch_embed(images)
            x = x + self.image_encoder.pos_embed
            for block in self.image_encoder.blocks:
                x = block(x)

            # Some ImageEncoderViT variants include a final normalization
            # layer before the neck. Apply it when present.
            if hasattr(self.image_encoder, "norm") and self.image_encoder.norm is not None:
                x = self.image_encoder.norm(x)

        return x  # [B,64,64,768]

    def forward(self, images):
        x = self._frozen_vit_forward(images)

        if x.shape[1:] != (64, 64, 768):
            raise RuntimeError(
                f"Expected frozen ViT output [B,64,64,768], got {tuple(x.shape)}"
            )

        # IMPORTANT: neck is outside no_grad so its parameters receive gradients.
        x = x.permute(0, 3, 1, 2).contiguous()
        feature_map = self.image_encoder.neck(x)  # [B,256,64,64]

        if feature_map.shape[1:] != (256, 64, 64):
            raise RuntimeError(
                f"Expected neck output [B,256,64,64], got {tuple(feature_map.shape)}"
            )

        feature_vector = self.spatial_attention_pool(feature_map)  # [B,256]
        output = self.classifier(feature_vector)  # [B,5]

        if output.shape[1] != 5:
            raise RuntimeError(
                f"Expected 5 output classes, got {output.shape[1]}"
            )

        return output


def get_trainable_parameters(model):
    """
    Return only parameters explicitly marked as trainable.

    Training code MUST pass this function to AdamW rather than
    model.parameters(), so frozen MRI-CORE parameters are excluded.

    Example:
        optimizer = torch.optim.AdamW(
            get_trainable_parameters(model),
            lr=1e-4,
            weight_decay=1e-4,
        )
    """
    return [p for p in model.parameters() if p.requires_grad]


def print_trainable_parameter_summary(model):
    trainable = 0
    frozen = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if param.requires_grad:
            trainable += n
            status = "TRAINABLE"
        else:
            frozen += n
            status = "FROZEN"
        print(f"{status:10s} | {name:60s} | {n:,}")

    print("=" * 90)
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {frozen:,}")
    print(f"Total parameters     : {trainable + frozen:,}")
    print("=" * 90)
