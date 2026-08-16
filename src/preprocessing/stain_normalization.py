"""Macenko stain normalization (Macenko et al., 2009).

Corrects H&E color variation between scanners/labs by decomposing each tile
into hematoxylin/eosin stain concentrations in optical-density space, then
re-composing those concentrations against a fixed reference stain profile.
See docs/ALGORITHMS.md for the full derivation and formulas.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.utils.config import MacenkoReferenceConfig

_OD_BACKGROUND_THRESHOLD = 0.15  # beta: OD below this is background/glass, not tissue
_ANGULAR_PERCENTILE = 1.0        # alpha: percentile used to find robust stain extremes
_MIN_TISSUE_PIXELS = 50          # too few tissue pixels to fit stain vectors reliably


def _rgb_to_od(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB pixels [..., 3] to optical density (Beer-Lambert law)."""
    normalized = rgb.astype(np.float64) + 1.0  # +1 avoids log(0) on pure-white pixels
    return -np.log10(normalized / 256.0)


def _od_to_rgb(od: np.ndarray) -> np.ndarray:
    """Inverse of _rgb_to_od, clipped to a valid uint8 image range."""
    rgb = 256.0 * np.power(10.0, -od) - 1.0
    return np.clip(rgb, 0, 255)


def estimate_stain_vectors(
    od_tissue: np.ndarray, angular_percentile: float = _ANGULAR_PERCENTILE
) -> np.ndarray:
    """Estimate the two dominant stain directions from tissue OD pixels.

    Projects OD pixels onto the plane of largest variance (top-2 eigenvectors
    of their covariance), then takes the two angular extremes (robust to
    outliers via a percentile cutoff) as the hematoxylin/eosin directions.
    Returns a [3, 2] matrix of unit stain vectors, ordered [hematoxylin, eosin].
    """
    _, eigvecs = np.linalg.eigh(np.cov(od_tissue.T))
    plane = eigvecs[:, -2:]  # two directions of largest OD variance

    projected = od_tissue @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])

    min_angle = np.percentile(angles, angular_percentile)
    max_angle = np.percentile(angles, 100 - angular_percentile)

    v_min = plane @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v_max = plane @ np.array([np.cos(max_angle), np.sin(max_angle)])

    # Hematoxylin (stains nuclei blue-violet) has a larger red OD component
    # than eosin under this projection convention.
    ordered = np.stack([v_min, v_max] if v_min[0] > v_max[0] else [v_max, v_min], axis=1)
    return ordered / np.linalg.norm(ordered, axis=0, keepdims=True)


def estimate_concentrations(od: np.ndarray, stain_vectors: np.ndarray) -> np.ndarray:
    """Least-squares solve `stain_vectors @ concentrations.T = od.T` per pixel."""
    concentrations, *_ = np.linalg.lstsq(stain_vectors, od.T, rcond=None)
    return concentrations.T  # [N, 2]


def normalize(
    image: Image.Image,
    reference: MacenkoReferenceConfig,
    background_threshold: float = _OD_BACKGROUND_THRESHOLD,
) -> Image.Image:
    """Re-stain an H&E tile against a fixed reference stain profile.

    Returns the original image unchanged if too few non-background pixels are
    available to fit stain vectors reliably (e.g. a near-empty tile that
    slipped past tissue filtering) — normalization noise on background is
    worse than leaving it untouched.
    """
    rgb = np.array(image.convert("RGB"))
    height, width = rgb.shape[:2]
    od = _rgb_to_od(rgb.reshape(-1, 3))

    tissue_pixels = od[np.all(od > background_threshold, axis=1)]
    if tissue_pixels.shape[0] < _MIN_TISSUE_PIXELS:
        return image

    source_vectors = estimate_stain_vectors(tissue_pixels)
    concentrations = estimate_concentrations(od, source_vectors)

    reference_vectors = np.array(reference.stain_vectors)
    reference_max = np.array(reference.max_concentrations)

    source_max = np.percentile(concentrations, 99, axis=0)
    source_max[source_max == 0] = 1e-6  # guard against a degenerate flat tile
    normalized_concentrations = concentrations / source_max * reference_max

    normalized_od = normalized_concentrations @ reference_vectors.T
    normalized_rgb = _od_to_rgb(normalized_od).reshape(height, width, 3).astype(np.uint8)
    return Image.fromarray(normalized_rgb)
