"""Tests for WSI tiling helpers.

plan_tiles()/iter_tile_images() need a real OpenSlide-backed WsiReader and
are exercised as integration tests in the Docker environment (see
README.md — OpenSlide's native library isn't installable via pip on
Windows). This module tests the pure tissue-coverage logic directly.
"""
from __future__ import annotations

import numpy as np

from src.utils.config import get_settings
from src.wsi.tiler import TileCoordinate, _tile_has_tissue


def test_tile_coordinate_fields():
    coord = TileCoordinate(x=10, y=20, index=3)
    assert (coord.x, coord.y, coord.index) == (10, 20, 3)


def test_tile_has_tissue_true_for_fully_covered_mask():
    mask = np.ones((100, 100), dtype=bool)
    config = get_settings().preprocessing
    assert _tile_has_tissue(mask, 0, 0, 256, 256, 1.0, 1.0, config) is True


def test_tile_has_tissue_false_for_empty_mask():
    mask = np.zeros((100, 100), dtype=bool)
    config = get_settings().preprocessing
    assert _tile_has_tissue(mask, 0, 0, 256, 256, 1.0, 1.0, config) is False


def test_tile_has_tissue_respects_min_fraction_threshold():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:20, :20] = True  # 400 / 10000 = 4% tissue coverage

    base = get_settings().preprocessing
    lenient = base.model_copy(update={"min_tissue_fraction": 0.01})
    strict = base.model_copy(update={"min_tissue_fraction": 0.5})

    assert _tile_has_tissue(mask, 0, 0, 256, 256, 1.0, 1.0, lenient) is True
    assert _tile_has_tissue(mask, 0, 0, 256, 256, 1.0, 1.0, strict) is False
