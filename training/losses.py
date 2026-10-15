"""
losses.py
─────────
Loss functions for Model 5 Pfirrmann Grading (Phase 2).

Includes:
- SoftOrdinalFocalLoss: Combines Gaussian soft ordinal target encoding,
  Focal loss for hard borderline cases, class weighting for imbalance,
  and auxiliary supervision for the mid-sagittal center slice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftOrdinalFocalLoss(nn.Module):
    """
    Combined Gaussian Soft Ordinal Target Encoding + Focal Loss + Auxiliary Loss.

    Pfirrmann grading is an ordinal classification task (Grade 1 to 5).
    Standard cross-entropy penalises near-misses (Grade 2 vs 3) as heavily
    as catastrophic errors (Grade 1 vs 5).

    This loss:
    1. Converts integer labels into a Gaussian-smoothed probability distribution
       centred on the true grade, with standard deviation sigma.
    2. Applies a Focal modulating factor (1 - p_c)^gamma to down-weight easy
       examples and concentrate gradient energy on borderline cases.
    3. Incorporates class weights to handle dataset grade imbalance.
    4. Computes an auxiliary loss on the mid-sagittal center slice (offset 0).
    """

    def __init__(
        self,
        num_classes: int = 5,
        sigma: float = 0.70,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        aux_weight: float = 0.30,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = float(sigma)
        self.gamma = float(gamma)
        self.aux_weight = float(aux_weight)
        self.eps = eps

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        # Precompute Gaussian soft ordinal distribution matrix [num_classes, num_classes]
        # target_matrix[k] = soft target distribution for ground-truth grade k
        coords = torch.arange(num_classes, dtype=torch.float32)
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # [C, C]
        target_matrix = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        target_matrix = target_matrix / target_matrix.sum(dim=-1, keepdim=True)
        self.register_buffer("target_matrix", target_matrix)

    def _focal_ordinal_loss(
        self,
        logits: torch.Tensor,
        targets_soft: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes focal-weighted cross-entropy against soft probability targets.

        logits:       [B, C]
        targets_soft: [B, C]
        labels:       [B] Ground-truth class indices for sample-level weighting.
        """
        probs = F.softmax(logits, dim=-1)  # [B, C]
        log_probs = torch.log(probs + self.eps)

        # Focal modulating factor: (1 - p_c)^gamma
        focal_weight = torch.pow(1.0 - probs, self.gamma)  # [B, C]

        # Elementwise loss across class distribution: (1 - p_c)^gamma * T_c * log(p_c)
        loss = -focal_weight * targets_soft * log_probs  # [B, C]

        # Sum over classes -> loss per sample [B]
        sample_loss = loss.sum(dim=-1)  # [B]

        # Sample-level class weighting: weight each sample by the rarity of its true class
        if self.class_weights is not None and labels is not None:
            sample_loss = sample_loss * self.class_weights[labels]

        # Mean over batch
        return sample_loss.mean()

    def forward(
        self,
        primary_logits: torch.Tensor,
        labels: torch.Tensor,
        center_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            primary_logits: [B, C] Logits from multi-view attention aggregator.
            labels:         [B] Integer class indices (0 to num_classes - 1).
            center_logits:  [B, C] Optional logits from auxiliary center slice.

        Returns:
            total_loss: Scalar loss tensor.
            metrics:    Dict containing loss breakdown for logging.
        """
        # Retrieve soft Gaussian ordinal distribution for batch
        soft_targets = self.target_matrix[labels]  # [B, C]

        primary_loss = self._focal_ordinal_loss(primary_logits, soft_targets, labels=labels)

        if center_logits is not None and self.aux_weight > 0.0:
            center_loss = self._focal_ordinal_loss(center_logits, soft_targets, labels=labels)
            total_loss = primary_loss + self.aux_weight * center_loss
        else:
            center_loss = torch.tensor(0.0, device=primary_logits.device)
            total_loss = primary_loss

        metrics = {
            "loss_total": total_loss.item(),
            "loss_primary": primary_loss.item(),
            "loss_center": center_loss.item(),
        }

        return total_loss, metrics