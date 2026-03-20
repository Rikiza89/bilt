# BILT (Because I Like Twice) - A PyTorch-based object detection library
# Copyright (C) 2024 Rikiza89
# Licensed under the GNU Affero General Public License v3.0

"""
BILT training losses.

BILTLoss combines:
  - Focal loss  (classification)  — focuses training on hard examples by
    down-weighting well-classified ones.  Introduced in RetinaNet
    (Lin et al. 2017, https://arxiv.org/abs/1708.02002).
  - Smooth-L1 loss  (bounding-box regression)  — less sensitive to outliers
    than plain L1 or L2 loss.

Both losses are normalised by the number of positive anchors in the batch so
that the scale remains roughly constant across different batch sizes and
image densities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Binary sigmoid focal loss (sum reduction).

    Parameters
    ----------
    inputs  : (N, C)  raw logits
    targets : (N, C)  binary targets {0, 1}
    alpha   : float   class-balance weighting factor
    gamma   : float   focusing parameter (0 = cross-entropy)

    Returns
    -------
    loss : scalar  (sum over all elements)
    """
    p = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    return loss.sum()


# ---------------------------------------------------------------------------
# Smooth-L1 loss
# ---------------------------------------------------------------------------

def smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Smooth-L1 (Huber) loss with customisable transition point.

        L(x) = 0.5 * x^2 / beta   if |x| < beta
               |x| - 0.5 * beta   otherwise

    Parameters
    ----------
    pred   : (N, 4)
    target : (N, 4)
    beta   : float  transition from quadratic to linear

    Returns
    -------
    loss : scalar  (sum over all elements)
    """
    diff = (pred - target).abs()
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta,
    )
    return loss.sum()


# ---------------------------------------------------------------------------
# Combined loss module
# ---------------------------------------------------------------------------

class BILTLoss(nn.Module):
    """
    Joint focal + smooth-L1 detection loss.

    Parameters
    ----------
    num_classes : int    Number of object classes (excluding background).
    alpha       : float  Focal loss alpha (class-balance weight).
    gamma       : float  Focal loss gamma (focusing strength).
    box_weight  : float  Relative weight of regression loss vs. cls loss.
    """

    def __init__(
        self,
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        box_weight: float = 1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.box_weight = box_weight

    def forward(
        self,
        cls_preds: torch.Tensor,   # (B, A, num_classes)
        box_preds: torch.Tensor,   # (B, A, 4)
        cls_targets: torch.Tensor, # (B, A)  -1=ignore, 0=bg, >0=class
        box_targets: torch.Tensor, # (B, A, 4) encoded deltas
        pos_mask: torch.Tensor,    # (B, A)  bool
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys 'total', 'cls', 'box'
        """
        num_pos = pos_mask.sum().clamp(min=1).float()

        # ------------------------------------------------------------------ #
        # Classification loss (focal, applied to all non-ignored anchors)    #
        # ------------------------------------------------------------------ #
        valid_mask = cls_targets >= 0                       # exclude ignore

        # Build one-hot targets (background = all-zeros)
        cls_targets_oh = torch.zeros_like(cls_preds)        # (B, A, C)
        if pos_mask.any():
            pos_classes = (cls_targets[pos_mask] - 1).clamp(
                0, self.num_classes - 1
            )
            flat_pos = pos_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
            cls_targets_oh.reshape(-1, self.num_classes)[flat_pos, pos_classes] = 1.0

        cls_loss = sigmoid_focal_loss(
            cls_preds[valid_mask],
            cls_targets_oh[valid_mask],
            alpha=self.alpha,
            gamma=self.gamma,
        ) / num_pos

        # ------------------------------------------------------------------ #
        # Regression loss (smooth-L1, positive anchors only)                 #
        # ------------------------------------------------------------------ #
        if pos_mask.any():
            box_loss = smooth_l1_loss(
                box_preds[pos_mask],
                box_targets[pos_mask],
            ) / num_pos
        else:
            box_loss = cls_preds.new_tensor(0.0)

        total = cls_loss + self.box_weight * box_loss

        return {
            "total": total,
            "cls":   cls_loss.detach(),
            "box":   box_loss.detach(),
        }
