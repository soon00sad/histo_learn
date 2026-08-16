"""Tests for GradCAM++ top-K region extraction."""
from __future__ import annotations

import numpy as np

from src.utils.config import XaiConfig
from src.xai.regions import top_k_regions


def _config(top_k: int = 3, threshold_ratio: float = 0.5, min_pixels: int = 4) -> XaiConfig:
    return XaiConfig(
        top_k_regions=top_k, region_threshold_ratio=threshold_ratio, min_region_pixels=min_pixels
    )


def test_top_k_regions_finds_distinct_hotspots_ranked_by_score():
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[10:20, 10:20] = 0.9  # strong hotspot, top-left
    cam[60:75, 60:75] = 0.6  # medium hotspot, bottom-right
    cam[40:42, 40:42] = 0.55  # small hotspot, exactly at the min pixel count (4 px)

    regions = top_k_regions(cam, _config(top_k=3))

    assert len(regions) == 3
    assert regions[0].score > regions[1].score > regions[2].score
    assert regions[0].x <= 15 and regions[0].y <= 15


def test_top_k_regions_respects_min_pixel_count():
    cam = np.zeros((50, 50), dtype=np.float32)
    cam[0, 0] = 1.0  # single pixel, below min_region_pixels=4

    assert top_k_regions(cam, _config(min_pixels=4)) == []


def test_top_k_regions_handles_all_zero_map():
    cam = np.zeros((32, 32), dtype=np.float32)
    assert top_k_regions(cam, _config()) == []


def test_top_k_regions_caps_at_top_k():
    cam = np.zeros((100, 100), dtype=np.float32)
    for i in range(5):
        cam[i * 15 : i * 15 + 6, i * 15 : i * 15 + 6] = 0.5 + i * 0.05

    regions = top_k_regions(cam, _config(top_k=3))

    assert len(regions) == 3
