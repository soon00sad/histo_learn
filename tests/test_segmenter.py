"""Tests for patch-level semantic segmentation.

Uses the stub_load_segmentation_model fixture (tests/conftest.py) to build
an untrained DeepLabV3+ instead of loading models/segmentation.pth, so this
suite needs no network access and doesn't depend on the trained checkpoint
(which isn't committed to git — see docs/MODEL.md).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.inference.segmenter import PatchSegmenter
from src.utils.bcss_classes import load_bcss_classes


def test_segment_returns_full_resolution_mask(stub_load_segmentation_model, synthetic_he_patch):
    segmenter = PatchSegmenter()
    result = segmenter.segment(synthetic_he_patch)

    size = segmenter.settings.segmentation_model.input_size
    assert result.mask.shape == (size, size)
    assert result.mask.dtype == np.uint8


def test_segment_class_fractions_are_valid_distribution(stub_load_segmentation_model, synthetic_he_patch):
    segmenter = PatchSegmenter()
    result = segmenter.segment(synthetic_he_patch)

    taxonomy = load_bcss_classes()
    assert set(result.class_pixel_fractions).issubset({c.name_en for c in taxonomy.classes})
    assert sum(result.class_pixel_fractions.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(0.0 < frac <= 1.0 for frac in result.class_pixel_fractions.values())
    assert 0.0 <= result.mean_confidence <= 1.0


def test_preprocess_produces_expected_tensor_shape(stub_load_segmentation_model, synthetic_he_patch):
    segmenter = PatchSegmenter()
    tensor = segmenter.preprocess(synthetic_he_patch)

    size = segmenter.settings.segmentation_model.input_size
    assert tensor.shape == (1, 3, size, size)


def test_segment_tensor_batch_matches_individual_calls(stub_load_segmentation_model, synthetic_he_patch):
    segmenter = PatchSegmenter()
    single = segmenter.segment(synthetic_he_patch)

    batch_tensor = torch.cat([segmenter.preprocess(synthetic_he_patch)] * 2, dim=0)
    batch_results = segmenter.segment_tensor_batch(batch_tensor)

    assert len(batch_results) == 2
    assert np.array_equal(batch_results[0].mask, single.mask)
    assert batch_results[0].class_pixel_fractions == pytest.approx(single.class_pixel_fractions, abs=1e-4)


def test_load_segmentation_model_missing_weights_raises(settings, tmp_path):
    from src.inference.segmenter import load_segmentation_model

    bad_settings = settings.model_copy(
        update={
            "segmentation_model": settings.segmentation_model.model_copy(
                update={"weights_path": str(tmp_path / "does-not-exist.pth")}
            )
        }
    )
    with pytest.raises(FileNotFoundError):
        load_segmentation_model(bad_settings)
