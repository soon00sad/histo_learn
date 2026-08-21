"""Tests for the pure mask-stitching math in src/inference/wsi_segmenter.py.

analyze_wsi_segmentation() itself needs a real OpenSlide-backed WsiReader and
is exercised as an integration test in the Docker environment only (same
constraint as src/wsi/tiler.py's plan_tiles/iter_tile_images — see
tests/test_tiler.py). This module tests the tile-accumulation/blending logic
directly with synthetic probability arrays, no WSI or model involved.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.inference.wsi_segmenter import UNCOVERED, accumulate_tile, hann_window, mask_from_accumulators
from src.utils.bcss_classes import load_bcss_classes


def test_hann_window_peaks_at_center_and_tapers_at_edges():
    w = hann_window(16)
    assert w.shape == (16, 16)
    assert w[8, 8] == pytest.approx(w.max())
    assert w[0, 0] < w[8, 8]
    assert (w > 0).all()  # clipped away from true zero, per the docstring


def test_hann_window_size_one_is_a_single_full_weight_pixel():
    w = hann_window(1)
    assert w.shape == (1, 1)
    assert w[0, 0] == 1.0


def test_accumulate_tile_places_prediction_at_correct_canvas_offset():
    num_classes, canvas = 3, 8
    prob_accum = np.zeros((num_classes, canvas, canvas), dtype=np.float32)
    weight_accum = np.zeros((canvas, canvas), dtype=np.float32)

    tile_probs = np.zeros((num_classes, 4, 4), dtype=np.float32)
    tile_probs[1] = 1.0  # tile is 100% confident in class 1
    taper = np.ones((2, 2), dtype=np.float32)  # no tapering, for a clean arithmetic check

    # level-0 tile at x=8,y=0 with downsample=4 -> canvas offset (2,0), canvas tile size 2
    accumulate_tile(prob_accum, weight_accum, tile_probs, tile_x=8, tile_y=0, downsample=4, tile_canvas_size=2, taper=taper)

    assert weight_accum[0:2, 2:4].sum() == pytest.approx(4.0)  # 2x2 region, weight 1 each
    assert weight_accum[0:2, 0:2].sum() == 0  # untouched region stays zero
    assert prob_accum[1, 0:2, 2:4].sum() == pytest.approx(4.0)
    assert prob_accum[0, 0:2, 2:4].sum() == 0


def test_accumulate_tile_clips_at_canvas_boundary_without_erroring():
    num_classes, canvas = 2, 4
    prob_accum = np.zeros((num_classes, canvas, canvas), dtype=np.float32)
    weight_accum = np.zeros((canvas, canvas), dtype=np.float32)
    tile_probs = np.ones((num_classes, 4, 4), dtype=np.float32)
    taper = hann_window(4)

    # Tile position pushes part of it off the canvas edge — must not raise/crash,
    # and the in-bounds corner (canvas x/y 2..3) should still get a contribution.
    accumulate_tile(prob_accum, weight_accum, tile_probs, tile_x=2, tile_y=2, downsample=1, tile_canvas_size=4, taper=taper)

    assert weight_accum.sum() > 0
    assert weight_accum[3, 3] > 0  # in-bounds corner of the straddling tile
    assert weight_accum.shape == (canvas, canvas)  # accumulator itself never resized/overflowed


def test_overlapping_tiles_blend_instead_of_hard_switching():
    """Two overlapping tiles voting for different classes should produce a
    weighted blend in the overlap region, not one tile silently winning —
    this is the seam-avoidance behavior the whole module exists for."""
    num_classes, canvas = 2, 8
    prob_accum = np.zeros((num_classes, canvas, canvas), dtype=np.float32)
    weight_accum = np.zeros((canvas, canvas), dtype=np.float32)
    taper = np.ones((4, 4), dtype=np.float32)

    tile_a = np.zeros((num_classes, 4, 4), dtype=np.float32)
    tile_a[0] = 1.0  # votes class 0
    tile_b = np.zeros((num_classes, 4, 4), dtype=np.float32)
    tile_b[1] = 1.0  # votes class 1

    accumulate_tile(prob_accum, weight_accum, tile_a, tile_x=0, tile_y=0, downsample=1, tile_canvas_size=4, taper=taper)
    accumulate_tile(prob_accum, weight_accum, tile_b, tile_x=2, tile_y=0, downsample=1, tile_canvas_size=4, taper=taper)

    # Overlap column x=2..3: both tiles contributed equally -> tied probabilities.
    overlap_weight = weight_accum[0, 2]
    assert overlap_weight == pytest.approx(2.0)
    assert prob_accum[0, 0, 2] == pytest.approx(prob_accum[1, 0, 2])


def test_mask_from_accumulators_argmax_and_fractions():
    taxonomy = load_bcss_classes()
    tumor_idx = next(c.model_index for c in taxonomy.classes if c.name_en == "tumor")
    stroma_idx = next(c.model_index for c in taxonomy.classes if c.name_en == "stroma")

    num_classes, canvas = taxonomy.num_classes, 4
    prob_accum = np.zeros((num_classes, canvas, canvas), dtype=np.float32)
    weight_accum = np.zeros((canvas, canvas), dtype=np.float32)

    # Left half confidently tumor, right half confidently stroma, all covered.
    prob_accum[tumor_idx, :, :2] = 1.0
    prob_accum[stroma_idx, :, 2:] = 1.0
    weight_accum[:, :] = 1.0

    mask, fractions = mask_from_accumulators(prob_accum, weight_accum, taxonomy)

    assert (mask[:, :2] == tumor_idx).all()
    assert (mask[:, 2:] == stroma_idx).all()
    assert fractions["tumor"] == pytest.approx(0.5)
    assert fractions["stroma"] == pytest.approx(0.5)
    assert sum(fractions.values()) == pytest.approx(1.0)


def test_mask_from_accumulators_excludes_uncovered_pixels_from_fractions():
    taxonomy = load_bcss_classes()
    tumor_idx = next(c.model_index for c in taxonomy.classes if c.name_en == "tumor")

    num_classes, canvas = taxonomy.num_classes, 4
    prob_accum = np.zeros((num_classes, canvas, canvas), dtype=np.float32)
    weight_accum = np.zeros((canvas, canvas), dtype=np.float32)

    prob_accum[tumor_idx, :2, :] = 1.0
    weight_accum[:2, :] = 1.0  # only the top half was ever covered by a tile

    mask, fractions = mask_from_accumulators(prob_accum, weight_accum, taxonomy)

    assert fractions == {"tumor": pytest.approx(1.0)}  # bottom (uncovered) half doesn't dilute this
    assert (mask[2:, :] == UNCOVERED).all()  # uncovered pixels get the sentinel, not a real class
    assert (mask[:2, :] == tumor_idx).all()


def test_mask_from_accumulators_uncovered_sentinel_does_not_collide_with_class_zero():
    """Regression test: a naive `np.zeros(...)` default for uncovered pixels
    would be indistinguishable from model_index 0 (tumor, the first class in
    the taxonomy) — confirmed visually on a real WSI, where background
    rendered as solid "tumor" colored regions. UNCOVERED (255) must never
    equal any real model_index."""
    taxonomy = load_bcss_classes()
    real_indices = {c.model_index for c in taxonomy.classes}
    assert UNCOVERED not in real_indices

    num_classes, canvas = taxonomy.num_classes, 2
    prob_accum = np.zeros((num_classes, canvas, canvas), dtype=np.float32)
    weight_accum = np.zeros((canvas, canvas), dtype=np.float32)  # nothing covered at all

    mask, fractions = mask_from_accumulators(prob_accum, weight_accum, taxonomy)

    assert (mask == UNCOVERED).all()
    assert fractions == {}
