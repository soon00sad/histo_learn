"""Patch-level semantic segmentation: H&E image tile -> per-pixel tissue-class mask.

Mirrors src/inference/classifier.py's shape (load/cache model, build
transform, single + batched inference) but for segmentation-models-pytorch's
DeepLabV3+ instead of timm's EfficientNet-B3 classifier. Deliberately
separate from classifier.py/model.py rather than replacing them in place —
master's binary classification path is untouched; this only exists on
feature/segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from src.utils.bcss_classes import BcssTaxonomy, load_bcss_classes
from src.utils.config import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: dict[str, nn.Module] = {}


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray  # [H, W] uint8, 0-based model class indices (see BcssClass.model_index)
    class_pixel_fractions: dict[str, float]  # name_en -> fraction of image area (0-1); absent classes omitted
    mean_confidence: float  # mean softmax probability of each pixel's predicted class


def load_segmentation_model(settings: Settings | None = None) -> nn.Module:
    """Load (and process-wide cache) the segmentation model with configured weights."""
    settings = settings or get_settings()
    cfg = settings.segmentation_model
    taxonomy = load_bcss_classes()
    weights_path = settings.resolve_path(cfg.weights_path)
    cache_key = str(weights_path)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Segmentation model weights not found at {weights_path}. "
            "Train one with src/training/train.py (see notebooks/train_segmentation_colab.ipynb) "
            "and place the checkpoint there."
        )

    model = smp.DeepLabV3Plus(
        encoder_name=cfg.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=taxonomy.num_classes,
    )
    state_dict = torch.load(weights_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.to(cfg.device)
    model.eval()

    logger.info(
        "Loaded DeepLabV3+/%s segmentation weights from %s onto %s",
        cfg.encoder_name, weights_path, cfg.device,
    )
    _MODEL_CACHE[cache_key] = model
    return model


def build_transform(settings: Settings) -> transforms.Compose:
    cfg = settings.segmentation_model
    return transforms.Compose(
        [
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalize_mean, cfg.normalize_std),
        ]
    )


class PatchSegmenter:
    """Wraps the loaded segmentation model and its preprocessing for single/batched inference."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.taxonomy: BcssTaxonomy = load_bcss_classes()
        self.model = load_segmentation_model(self.settings)
        self.transform = build_transform(self.settings)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Return a [1, C, H, W] tensor ready for the model, on the configured device."""
        tensor = self.transform(image.convert("RGB"))
        return tensor.unsqueeze(0).to(self.settings.segmentation_model.device)

    def _result_from_probs(self, probs: np.ndarray) -> SegmentationResult:
        """`probs` is [num_classes, H, W] softmax output for one image."""
        mask = probs.argmax(axis=0).astype(np.uint8)
        total_pixels = mask.size

        fractions: dict[str, float] = {}
        for bcss_class in self.taxonomy.classes:
            count = int((mask == bcss_class.model_index).sum())
            if count:
                fractions[bcss_class.name_en] = count / total_pixels

        predicted_class_probs = np.take_along_axis(probs, mask[np.newaxis, :, :], axis=0)[0]
        mean_confidence = float(predicted_class_probs.mean())

        return SegmentationResult(
            mask=mask,
            class_pixel_fractions=fractions,
            mean_confidence=mean_confidence,
        )

    @torch.no_grad()
    def segment(self, image: Image.Image) -> SegmentationResult:
        """Segment a single PIL image."""
        logits = self.model(self.preprocess(image))
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        return self._result_from_probs(probs)

    @torch.no_grad()
    def segment_tensor_batch(self, batch: torch.Tensor) -> list[SegmentationResult]:
        """Segment a pre-stacked [N, C, H, W] batch (used by the WSI pipeline)."""
        logits = self.model(batch.to(self.settings.segmentation_model.device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        return [self._result_from_probs(probs[i]) for i in range(probs.shape[0])]
