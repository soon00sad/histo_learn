"""Combined weighted cross-entropy + Dice loss for BCSS's imbalanced classes.

Weighted CE alone tends to still under-predict very rare classes on
histopathology segmentation; Dice loss directly optimizes region overlap and
is largely insensitive to class-frequency imbalance. Combining both is a
common, well-tested choice for exactly this kind of long-tailed
segmentation problem — see docs/ALGORITHMS.md.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from src.training.dataset import IGNORE_INDEX


class WeightedCeDiceLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        ignore_index: int = IGNORE_INDEX,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index, from_logits=True)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (1 - self.dice_weight) * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)


def compute_class_weights(class_pixel_counts: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so the mean weight is 1.

    `class_pixel_counts` is a `[num_classes]` tensor of total pixel counts
    per class across the training set (see `metrics.count_class_pixels`).
    Classes absent from the sampled set get clamped to a count of 1 so they
    don't produce an infinite weight.
    """
    counts = class_pixel_counts.float().clamp(min=1)
    weights = 1.0 / counts
    weights = weights * (weights.numel() / weights.sum())
    return weights
