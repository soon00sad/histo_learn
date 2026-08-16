"""Extraction of the top-K diagnostically significant regions from a GradCAM map."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as connected_components

from src.utils.config import XaiConfig


@dataclass(frozen=True)
class AttentionRegion:
    x: int
    y: int
    width: int
    height: int
    score: float


def top_k_regions(cam_map: np.ndarray, config: XaiConfig) -> list[AttentionRegion]:
    """Find the K highest-scoring connected components of a GradCAM heatmap.

    Thresholds at a fraction of the map's peak activation, labels connected
    components, and ranks them by mean activation — the same approach as the
    validated prototype's region extraction.
    """
    if cam_map.size == 0 or float(cam_map.max()) <= 0:
        return []

    threshold = cam_map.max() * config.region_threshold_ratio
    binary_mask = (cam_map > threshold).astype(np.uint8)
    labeled, num_components = connected_components(binary_mask)

    regions: list[AttentionRegion] = []
    for component_id in range(1, num_components + 1):
        mask = labeled == component_id
        ys, xs = np.where(mask)
        if len(xs) < config.min_region_pixels:
            continue
        regions.append(
            AttentionRegion(
                x=int(xs.min()),
                y=int(ys.min()),
                width=int(xs.max() - xs.min()),
                height=int(ys.max() - ys.min()),
                score=float(cam_map[mask].mean()),
            )
        )

    regions.sort(key=lambda r: r.score, reverse=True)
    return regions[: config.top_k_regions]
