"""Per-class IoU / mIoU tracking via an accumulated confusion matrix.

Standard semantic-segmentation metric: IoU_c = intersection_c / union_c,
accumulated over every batch in an epoch (not averaged per-batch, which
would bias small batches) rather than computed fresh each time.
"""
from __future__ import annotations

import torch

from src.training.dataset import IGNORE_INDEX


class ConfusionMatrix:
    def __init__(self, num_classes: int, ignore_index: int = IGNORE_INDEX) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """`preds`/`targets` are integer class-index tensors of the same shape."""
        valid = targets != self.ignore_index
        p = preds[valid].reshape(-1).to(torch.int64)
        t = targets[valid].reshape(-1).to(torch.int64)
        indices = t * self.num_classes + p
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes).cpu()

    def per_class_iou(self) -> torch.Tensor:
        """`[num_classes]` float tensor; NaN for classes absent from both preds and targets."""
        intersection = torch.diag(self.matrix).float()
        union = self.matrix.sum(dim=0).float() + self.matrix.sum(dim=1).float() - intersection
        iou = torch.full_like(union, float("nan"))
        present = union > 0
        iou[present] = intersection[present] / union[present]
        return iou

    def mean_iou(self) -> float:
        """Mean over classes that appeared at least once (NaN classes excluded)."""
        iou = self.per_class_iou()
        valid = iou[~torch.isnan(iou)]
        return valid.mean().item() if valid.numel() else 0.0

    def reset(self) -> None:
        self.matrix.zero_()


def count_class_pixels(dataset, num_classes: int, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Total per-class pixel counts across a dataset, for `losses.compute_class_weights`."""
    counts = torch.zeros(num_classes, dtype=torch.int64)
    for _, mask in dataset:
        valid = mask[mask != ignore_index]
        counts += torch.bincount(valid.reshape(-1), minlength=num_classes)[:num_classes]
    return counts
