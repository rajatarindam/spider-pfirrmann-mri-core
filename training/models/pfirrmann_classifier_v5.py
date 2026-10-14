"""
pfirrmann_classifier_v5.py
──────────────────────────
Model 5 Phase 1 + Phase 2 Architecture: Multi-View Pfirrmann Classifier.

Architecture Flow:
1. Frozen MRI-CORE ViT-B (blocks 0-11) processes 5 contiguous sagittal MRI slices.
   - Slices are processed in mini-batches under torch.no_grad() to keep peak VRAM < 3 GB.
2. Unfrozen Pretrained SAM Neck (lr=1e-5):
   - 1x1 Conv (768 -> 256) + 3x3 Conv (256 -> 256) creates local spatial receptive fields.
3. SpatialAttentionPooling (lr=1e-4):
   - 1x1 Conv learns dynamic token attention over the 64x64 feature map, replacing uniform mean pooling.
4. ViewPositionalEmbedding (lr=1e-4):
   - 1D offset embedding for [-2, -1, 0, +1, +2].
5. Dual Output Heads:
   - Auxiliary Center Classifier: Evaluates mid-sagittal slice (offset 0, index 2) directly.
   - ViewAttentionAggregator + Primary Classifier: Fuses all 5 slices dynamically.
"""

from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure mri_core can be imported from root or parent
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent  # .../spider-pfirrmann-mri-core/context or repo root
for _p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mri_core.cfg import parse_args
from mri_core.models.sam import sam_model_registry
from configs.paths import MRI_CORE_CHECKPOINT_PATH


def _resolve_checkpoint_path() -> Path:
    """Finds mri_foundation.pth whether run inside context/ or at repo root."""
    ckpt = Path(MRI_CORE_CHECKPOINT_PATH)
    if ckpt.exists():
        return ckpt
    # Fallback: check repo root parent
    alt = Path(__file__).resolve().parents[4] / "mri_core" / "pretrained_weights" / "mri_foundation.pth"
    if alt.exists():
        return alt
    alt2 = Path("D:/BAAA project/spider-pfirrmann-mri-core/mri_core/pretrained_weights/mri_foundation.pth")
    if alt2.exists():
        return alt2
    raise FileNotFoundError(f"MRI-CORE checkpoint not found at {ckpt} or {alt}")


# =============================================================================
# SUB-MODULES
# =============================================================================

class SpatialAttentionPooling(nn.Module):
    """
    Spatial Attention Pooling for 2D feature maps.
    Learns a 1x1 convolution over channels to weight informative patches
    (e.g., disc nucleus, annular border, endplates) and suppress black padding.

    Input shape:  [B*N, C, H, W]  (e.g., [5, 256, 64, 64])
    Output shape: [B*N, C]        (e.g., [5, 256])
    """

    def __init__(self, channels: int = 256):
        super().__init__()
        self.attention = nn.Conv2d(channels, 1, kernel_size=1, bias=True)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        scores = self.attention(feature_map)  # [B*N, 1, H, W]
        b_n, _, h, w = scores.shape
        weights = F.softmax(scores.view(b_n, -1), dim=1).view(b_n, 1, h, w)
        pooled = (feature_map * weights).sum(dim=(2, 3))  # [B*N, C]
        return pooled


class ViewPositionalEmbedding(nn.Module):
    """
    Learnable 1D positional embedding for the 5 slice offsets (-2, -1, 0, +1, +2).
    Input shape:  [B, 5, dim]
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
    Masked scaled dot-product attention over the 5 slice views within a disc.
    Computes per-view scalar attention scores via Q·K dot product,
    masks out padded/missing slices, and returns a single fused disc representation.

    Input shapes: view_features [B, N, dim], mask [B, N] bool
    Output shape: [B, dim]
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
        attn = (q * k).sum(dim=-1) * self.scale    # [B, N]

        if mask is not None:
            attn = attn.masked_fill(~mask, float("-inf"))
            all_masked = (~mask).all(dim=1, keepdim=True)
            if all_masked.any():
                attn = torch.where(all_masked, torch.zeros_like(attn), attn)

        weights = F.softmax(attn, dim=1)                              # [B, N]
        fused   = (view_features * weights.unsqueeze(-1)).sum(dim=1)  # [B, dim]
        return fused


# =============================================================================
# MAIN CLASSIFIER MODEL
# =============================================================================

class PfirrmannClassifierV5(nn.Module):
    """
    Combined Phase 1 + Phase 2 Pfirrmann Grading Architecture.

    Forward flow:
        images: [B, 5, 3, 1024, 1024]
            ↓ (slice-by-slice frozen ViT-B under no_grad)
        vit_feat: [B*5, 64, 64, 768]
            ↓ permute to [B*5, 768, 64, 64]
        unfrozen SAM Neck: [B*5, 256, 64, 64]
            ↓ SpatialAttentionPooling
        pooled_slices: [B*5, 256]
            ↓ reshape to [B, 5, 256] + ViewPositionalEmbedding
        view_features: [B, 5, 256]
            ├── Center Slice (index 2) → Aux Classifier Head → center_logits [B, 5]
            └── ViewAttentionAggregator → Fused [B, 256] → Primary Classifier → primary_logits [B, 5]
    """

    def __init__(
        self,
        num_views: int = 5,
        num_classes: int = 5,
        checkpoint_path: str | Path | None = None,
    ):
        super().__init__()
        self.num_views = num_views
        self.num_classes = num_classes

        # ── 1. Load Pretrained MRI-CORE SAM Backbone ─────────────────────────
        ckpt_file = str(checkpoint_path) if checkpoint_path else str(_resolve_checkpoint_path())

        args = parse_args()
        args.arch                    = "vit_b"
        args.image_size              = 1024
        args.if_encoder_adapter      = False
        args.if_mask_decoder_adapter = False
        args.if_update_encoder       = False

        sam_model = sam_model_registry["vit_b"](
            args=args,
            checkpoint=ckpt_file,
            num_classes=1,
            image_size=1024,
            pretrained_sam=False,
        )

        self.image_encoder = sam_model.image_encoder

        # Correct 14x14 -> 64x64 interpolated positional embedding
        self._load_corrected_positional_embedding(ckpt_file)

        # ── 2. Freeze ViT Transformer Blocks & Embeddings ────────────────────
        for param in self.image_encoder.patch_embed.parameters():
            param.requires_grad = False
        self.image_encoder.pos_embed.requires_grad = False

        if hasattr(self.image_encoder, "norm") and self.image_encoder.norm is not None:
            for param in self.image_encoder.norm.parameters():
                param.requires_grad = False

        for block in self.image_encoder.blocks:
            for param in block.parameters():
                param.requires_grad = False

        # ── 3. Unfreeze SAM Neck (Phase 1A) ──────────────────────────────────
        # Pretrained 1.2M conv parameters for local spatial patch interaction
        for param in self.image_encoder.neck.parameters():
            param.requires_grad = True

        # ── 4. Trainable Modules ─────────────────────────────────────────────
        # Spatial Attention Pooling (Phase 1B)
        self.spatial_attention_pool = SpatialAttentionPooling(channels=256)

        # Slice view positional embedding
        self.view_pos_embed = ViewPositionalEmbedding(dim=256, num_views=num_views)

        # Multi-view attention aggregator
        self.view_aggregator = ViewAttentionAggregator(dim=256)

        # Primary Classifier (Multi-view fused disc representation)
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

        # Auxiliary Center Slice Classifier (Phase 2C - Offset 0 mid-sagittal view)
        self.aux_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

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
            raise ValueError(f"Expected checkpoint pos_embed [1,197,768], got {tuple(pos.shape)}")

        # Remove CLS token: [1, 197, 768] -> [1, 196, 768]
        pos = pos[:, 1:, :]
        # Reshape to grid: [1, 14, 14, 768]
        pos = pos.view(1, 14, 14, 768).permute(0, 3, 1, 2)
        # Bilinear interpolation: 14x14 -> 64x64
        pos = F.interpolate(pos, size=(64, 64), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).contiguous()

        if tuple(pos.shape) != (1, 64, 64, 768):
            raise RuntimeError(f"Corrected pos_embed has shape {tuple(pos.shape)}")

        self.image_encoder.pos_embed.copy_(
            pos.to(
                device=self.image_encoder.pos_embed.device,
                dtype=self.image_encoder.pos_embed.dtype,
            )
        )

    def _frozen_vit_forward_single(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass for a single slice image [1, 3, 1024, 1024] under no_grad."""
        x = self.image_encoder.patch_embed(image)
        x = x + self.image_encoder.pos_embed
        for block in self.image_encoder.blocks:
            x = block(x)
        if hasattr(self.image_encoder, "norm") and self.image_encoder.norm is not None:
            x = self.image_encoder.norm(x)
        return x  # [1, 64, 64, 768]

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, 5, 3, 1024, 1024]
            mask:   [B, 5] bool tensor (True for valid slice, False for padded)

        Returns:
            primary_logits: [B, 5] (Fused multi-view classification logits)
            center_logits:  [B, 5] (Mid-sagittal offset 0 auxiliary logits)
        """
        b, n, c, h, w = images.shape
        flat_images = images.view(b * n, c, h, w)  # [B*5, 3, 1024, 1024]

        # ── Step 1: Frozen ViT forward (Slice-by-slice to keep peak VRAM < 3GB) ──
        slice_feats = []
        with torch.no_grad():
            for i in range(b * n):
                feat = self._frozen_vit_forward_single(flat_images[i : i + 1])
                slice_feats.append(feat)
            vit_feat = torch.cat(slice_feats, dim=0)  # [B*5, 64, 64, 768]

        # ── Step 2: Unfrozen SAM Conv Neck ───────────────────────────────────
        # vit_feat: [B*5, 64, 64, 768] -> permute to [B*5, 768, 64, 64]
        vit_feat_perm = vit_feat.permute(0, 3, 1, 2).contiguous()
        neck_feat = self.image_encoder.neck(vit_feat_perm)  # [B*5, 256, 64, 64]

        # ── Step 3: Spatial Attention Pooling ────────────────────────────────
        pooled_slices = self.spatial_attention_pool(neck_feat)  # [B*5, 256]

        # ── Step 4: Multi-View Structuring & Positional Embedding ───────────
        view_feat = pooled_slices.view(b, n, 256)          # [B, 5, 256]
        view_feat = self.view_pos_embed(view_feat)         # [B, 5, 256]

        # ── Step 5: Auxiliary Mid-Sagittal Center Branch (Offset 0 = index 2) ──
        center_feat = view_feat[:, 2, :]                   # [B, 256]
        center_logits = self.aux_classifier(center_feat)   # [B, 5]

        # ── Step 6: Multi-View Attention Aggregator & Primary Head ───────────
        if mask is None:
            mask = torch.ones(b, n, dtype=torch.bool, device=images.device)

        fused_disc = self.view_aggregator(view_feat, mask=mask)  # [B, 256]
        primary_logits = self.classifier(fused_disc)             # [B, 5]

        return primary_logits, center_logits

    def print_summary(self) -> None:
        neck_params = sum(p.numel() for p in self.image_encoder.neck.parameters() if p.requires_grad)
        head_params = (
            sum(p.numel() for p in self.spatial_attention_pool.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.view_pos_embed.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.view_aggregator.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
            + sum(p.numel() for p in self.aux_classifier.parameters() if p.requires_grad)
        )
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)

        print("=" * 75)
        print("PfirrmannClassifierV5 — Parameter Summary")
        print(f"  Trainable Group 0 (SAM Neck, lr=1e-5)     : {neck_params:>10,} params")
        print(f"  Trainable Group 1 (Heads/Attn, lr=1e-4)   : {head_params:>10,} params")
        print(f"  Total Trainable Parameters                 : {neck_params + head_params:>10,} params")
        print(f"  Frozen Parameters (ViT-B Backbone)         : {frozen_params:>10,} params")
        print(f"  Total Model Parameters                     : {neck_params + head_params + frozen_params:>10,} params")
        print("=" * 75)
