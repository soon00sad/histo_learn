"""Background/tissue filtering: discard glass-only tiles before expensive inference.

Whole-slide images are typically 60-80% empty glass. Stained tissue is far
more saturated in HSV color space than glass or whitespace, so we threshold
the saturation channel via Otsu's method (Otsu, 1979) to separate the two
automatically, without a hand-tuned per-scanner constant.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.utils.config import PreprocessingConfig


def _to_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    return image


def saturation_channel(image: Image.Image | np.ndarray) -> np.ndarray:
    """Return the HSV saturation channel (uint8, 0-255) of an RGB image."""
    rgb = _to_rgb_array(image)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 1]


def otsu_threshold(saturation: np.ndarray) -> float:
    """Otsu's automatic threshold separating tissue (high saturation) from
    background glass (low saturation)."""
    threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(threshold)


def tissue_mask(
    image: Image.Image | np.ndarray, threshold: float | None = None
) -> tuple[np.ndarray, float]:
    """Return (boolean tissue mask, threshold used). `threshold=None` runs Otsu."""
    saturation = saturation_channel(image)
    resolved_threshold = threshold if threshold is not None else otsu_threshold(saturation)
    mask = saturation > resolved_threshold
    return mask, resolved_threshold


def tissue_fraction(image: Image.Image | np.ndarray, threshold: float | None = None) -> float:
    """Fraction of pixels classified as tissue (vs. background glass)."""
    mask, _ = tissue_mask(image, threshold)
    return float(mask.mean())


def has_sufficient_tissue(image: Image.Image | np.ndarray, config: PreprocessingConfig) -> bool:
    """Whether a tile has enough tissue to be worth running through the model."""
    fraction = tissue_fraction(image, config.tissue_saturation_threshold)
    return fraction >= config.min_tissue_fraction
