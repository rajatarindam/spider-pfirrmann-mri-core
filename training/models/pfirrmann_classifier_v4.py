"""
pfirrmann_classifier_v4.py
──────────────────────────
Model 5 Multi-View Pfirrmann Classifier (revised architecture).

Key architectural change vs v3 (Model 5 first attempt):
    REMOVED : MRI-CORE SAM convolutional neck  (Conv2d stack, ~1.2M params,
               operated at 64×64 spatial resolution — VRAM-expensive,
               overkill since spatial tokens are immediately mean-pooled)
    REMOVED : SpatialAttentionPooling (1×1 Conv2d — redundant after the neck
               is replaced by a mean pool + linear projection)
    ADDED   : LinearProjection — mean pool ViT spatial tokens [B*N,64,64,768]
               → [B*N,768], then Linear(768,256) + LayerNorm + GELU + Dropout

All other components are unchanged from v3.

Trainable components (all start from random initialisation):
    LinearProjection       – replaces neck + SpatialAttentionPool
    ViewPositionalEmbedding – learnable 1D offset embedding (5 × 256)
    ViewAttentionAggregator – Q/K dot-product attention over the 5 views
    Classifier MLP          – Linear 256→256→128→5 with LayerNorm + GELU

Frozen components:
    image_encoder.patch_embed
    image_encoder.pos_embed
    image_encoder.blocks (0–11)
    image_encoder.norm
    image_encoder.neck  ← kept in memory (loads checkpoint cleanly),
                          frozen and never called in forward()

Single optimizer group at lr=1e-4 (no differential LR needed — no pretrained
conv weights remain in the trainable parameter set).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mri_core.cfg import parse_args
from mri_core.models.sam import sam_model_registry
from configs.paths import MRI_CORE_CHECKPOINT_PATH


# =============================================================================
# SUB-MODULES
# =============================================================================

class LinearProjection(nn.Module):
    """
    Lightweight projection from raw ViT token space to embedding dimension.

    Replaces the heavy SAM convolutional neck + SpatialAttentionPooling.

    Forward:
        1. Mean-pool spatial tokens:  [B*N, 64, 64, 768] → [B*N, 768]
        2. Linear(768, 256) + LayerNorm(256) + GELU + Dropout

    Input  shape: [B*N, H, W, C]  where H=W=64, C=768
    Output shape: [B*N, out_dim]  where out_dim=256
    """

    def __init__(self, in_dim: int = 768, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*N, H, W, C]
        x = x.mean(dim=(1, 2))  # [B*N, C]  – spatial mean pool
        return self.proj(x)     # [B*N, out_dim]


class ViewPositionalEmbedding(nn.Module):
    """
    Learnable 1D positional embedding for the 5 slice offsets (-2,-1,0,+1,+2).

    Input  shape: [B, 5, dim]
    Output shape: [B, 5, dim]
    """

    def __init__(self, dim: int = 256, num_views: int = 5):
        super().__init__()
        self.offset_embed = nn.Embedding(num_views, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        idx = torch.arange(n, device=x.device).unsqueeze(0).expand(b, -1)
        return x + self.offset_embed(idx)


class ViewAttentionAggregator(nn.Module):
    """
    Masked dot-product attention over the 5 slice views within a disc.

    Computes per-view scalar attention scores via Q·K dot product,
    masks out padded / missing slices (True = real, False = padded),
    and returns a single fused disc representation.

    Input  shapes: view_features [B, N, dim],  mask [B, N] bool
    Output shape : [B, dim]
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.query = nn.Linear(dim, dim, bias=False)
        self.key   = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5

    def forward(
        self,
        view_features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.query(view_features)              # [B, N, dim]
        k = self.key(view_features)                # [B, N, dim]
        attn = (q * k).sum(dim=-1) * self.scale   # [B, N]

        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
            # Safety: if an entire sample has no valid slice, unmask all equally
            all_masked = (~mask).all(dim=1, keepdim=True)
            if all_masked.any():
                attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        weights = F.softmax(attn, dim=1)                           # [B, N]
        fused   = (view_features * weights.unsqueeze(-1)).sum(dim=1)  # [B, dim]
        return fused


# =============================================================================
# MAIN MODEL
# =============================================================================

class PfirrmannClassifierV4(nn.Module):
    """
    Model 5 Multi-View Pfirrmann Classifier (revised architecture).

    Forward flow:
        images [B, 5, 3, 1024, 1024]
            ↓  flatten to [B*5, 3, 1024, 1024]
        frozen ViT backbone  →  [B*5, 64, 64, 768]
            ↓  LinearProjection (mean pool + linear)
        [B*5, 256]
            ↓  reshape to [B, 5, 256]
        ViewPositionalEmbedding  →  [B, 5, 256]
            ↓  ViewAttentionAggregator (masked attention)
        [B, 256]
            ↓  Classifier MLP
        logits  [B, 5]
    """

    def __init__(self, num_views: int = 5, num_classes: int = 5):
        super().__init__()

        # ── Load MRI-CORE backbone ──────────────────────────────────────────
        args = parse_args()
        args.arch                    = "vit_b"
        args.image_size              = 1024
        args.if_encoder_adapter      = False
        args.if_mask_decoder_adapter = False
        args.if_update_encoder       = False

        model = sam_model_registry["vit_b"](
            args=args,
            checkpoint=MRI_CORE_CHECKPOINT_PATH,
            num_classes=1,
            image_size=1024,
            pretrained_sam=False,
        )

        self.image_encoder = model.image_encoder

        # Correct the 14×14 → 64×64 positional embedding
        self._load_corrected_positional_embedding(MRI_CORE_CHECKPOINT_PATH)

        # ── Freeze the entire ViT backbone (including neck) ─────────────────
        # The neck is frozen AND unused in forward() – it remains in the model
        # so that checkpoint keys stay consistent with v3 if needed.
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # ── Trainable modules ───────────────────────────────────────────────
        self.projection   = LinearProjection(in_dim=768, out_dim=256, dropout=0.1)
        self.view_pos_embed = ViewPositionalEmbedding(dim=256, num_views=num_views)
        self.view_aggregator = ViewAttentionAggregator(dim=256)

        # Classification MLP – LayerNorm (not BatchNorm) supports B=1
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    # ── Positional embedding correction (identical to v3) ───────────────────

    @torch.no_grad()
    def _load_corrected_positional_embedding(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

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
            raise ValueError(f"Expected pos_embed [1,197,768], got {tuple(pos.shape)}")

        pos = pos[:, 1:, :]                                        # remove CLS  [1,196,768]
        pos = pos.view(1, 14, 14, 768).permute(0, 3, 1, 2)        # [1,768,14,14]
        pos = F.interpolate(pos, size=(64, 64), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).contiguous()                 # [1,64,64,768]

        if tuple(pos.shape) != (1, 64, 64, 768):
            raise RuntimeError(f"Corrected pos_embed has shape {tuple(pos.shape)}")

        self.image_encoder.pos_embed.copy_(
            pos.to(
                device=self.image_encoder.pos_embed.device,
                dtype=self.image_encoder.pos_embed.dtype,
            )
        )

    # ── Frozen ViT forward pass ──────────────────────────────────────────────

    def _frozen_vit_forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Pass images through the frozen ViT blocks under torch.no_grad()
        to avoid storing the large backbone computation graph.

        Input : [B*N, 3, 1024, 1024]
        Output: [B*N, 64, 64, 768]
        """
        with torch.no_grad():
            x = self.image_encoder.patch_embed(images)
            x = x + self.image_encoder.pos_embed
            for block in self.image_encoder.blocks:
                x = block(x)
            if hasattr(self.image_encoder, "norm") and self.image_encoder.norm is not None:
                x = self.image_encoder.norm(x)
        return x  # [B*N, 64, 64, 768]

    # ── Full forward pass ────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            images : [B, 5, 3, 1024, 1024]
            mask   : [B, 5] BoolTensor — True for real slice, False for padded
                     (with the v2 CSV all slices are real; mask is kept for
                      safety and backward-compatibility)

        Returns:
            logits : [B, 5]
        """
        b, n, c, h, w = images.shape

        flat_images = images.view(b * n, c, h, w)              # [B*N, 3, 1024, 1024]

        vit_feat   = self._frozen_vit_forward(flat_images)      # [B*N, 64, 64, 768]
        proj_feat  = self.projection(vit_feat)                  # [B*N, 256]

        view_feat  = proj_feat.view(b, n, 256)                  # [B, 5, 256]
        view_feat  = self.view_pos_embed(view_feat)             # [B, 5, 256]

        if mask is None:
            mask = torch.ones(b, n, dtype=torch.bool, device=images.device)

        fused  = self.view_aggregator(view_feat, mask=mask)     # [B, 256]
        logits = self.classifier(fused)                         # [B, 5]

        return logits

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print("=" * 80)
        print("PfirrmannClassifierV4 – Parameter Summary")
        print(f"  Trainable parameters : {trainable:,}")
        print(f"  Frozen parameters    : {frozen:,}")
        print(f"  Total parameters     : {trainable + frozen:,}")
        print("=" * 80)

        print("\nTrainable module breakdown:")
        for name, module in self.named_children():
            if any(p.requires_grad for p in module.parameters()):
                n = sum(p.numel() for p in module.parameters() if p.requires_grad)
                print(f"  {name:25s} : {n:>10,} params")
