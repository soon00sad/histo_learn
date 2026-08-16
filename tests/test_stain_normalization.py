"""Tests for Macenko stain normalization."""
from __future__ import annotations

import numpy as np

from src.preprocessing.stain_normalization import (
    _od_to_rgb,
    _rgb_to_od,
    estimate_concentrations,
    estimate_stain_vectors,
    normalize,
)
from src.utils.config import get_settings


def test_rgb_od_roundtrip_is_approximately_identity():
    rgb = np.array([[10, 120, 250], [0, 255, 128]], dtype=np.uint8)
    recovered = _od_to_rgb(_rgb_to_od(rgb))
    assert np.allclose(recovered, rgb, atol=1.0)


def test_estimate_stain_vectors_are_unit_norm():
    rng = np.random.default_rng(0)
    hematoxylin_like = rng.normal(loc=[0.6, 0.7, 0.5], scale=0.03, size=(500, 3))
    eosin_like = rng.normal(loc=[0.25, 0.6, 0.4], scale=0.03, size=(500, 3))
    od_tissue = np.clip(np.vstack([hematoxylin_like, eosin_like]), 0.16, None)

    vectors = estimate_stain_vectors(od_tissue)

    assert vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=0), 1.0, atol=1e-6)
    assert not np.isnan(vectors).any()


def test_estimate_concentrations_reconstructs_exact_linear_mixture():
    stain_vectors = np.array([[0.6, 0.2], [0.7, 0.8], [0.4, 0.55]])
    true_concentrations = np.array([[1.0, 0.5], [0.2, 1.3]])
    od = true_concentrations @ stain_vectors.T

    recovered = estimate_concentrations(od, stain_vectors)

    assert np.allclose(recovered, true_concentrations, atol=1e-6)


def test_normalize_returns_unchanged_image_for_blank_glass(blank_glass_patch):
    reference = get_settings().preprocessing.macenko_reference
    result = normalize(blank_glass_patch, reference)
    assert list(result.getdata()) == list(blank_glass_patch.convert("RGB").getdata())


def test_normalize_preserves_image_size_and_mode(synthetic_he_patch):
    reference = get_settings().preprocessing.macenko_reference
    result = normalize(synthetic_he_patch, reference)
    assert result.size == synthetic_he_patch.size
    assert result.mode == "RGB"
