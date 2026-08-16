"""Tests for Otsu-based tissue/background filtering."""
from __future__ import annotations

from src.preprocessing.tissue_filter import has_sufficient_tissue, tissue_fraction, tissue_mask
from src.utils.config import get_settings


def test_blank_glass_has_low_tissue_fraction(blank_glass_patch):
    fraction = tissue_fraction(blank_glass_patch)
    assert fraction < 0.05


def test_stained_patch_has_higher_tissue_fraction_than_blank_glass(
    synthetic_he_patch, blank_glass_patch
):
    assert tissue_fraction(synthetic_he_patch) > tissue_fraction(blank_glass_patch)


def test_tissue_mask_shape_matches_image(synthetic_he_patch):
    mask, threshold = tissue_mask(synthetic_he_patch)
    assert mask.shape == (256, 256)
    assert mask.dtype == bool
    assert 0 <= threshold <= 255


def test_has_sufficient_tissue_rejects_blank_glass(blank_glass_patch):
    config = get_settings().preprocessing
    assert has_sufficient_tissue(blank_glass_patch, config) is False


def test_has_sufficient_tissue_accepts_stained_patch(synthetic_he_patch):
    config = get_settings().preprocessing
    assert has_sufficient_tissue(synthetic_he_patch, config) is True


def test_explicit_threshold_overrides_otsu(synthetic_he_patch):
    # A threshold of 255 means "nothing can exceed it" -> zero tissue fraction.
    assert tissue_fraction(synthetic_he_patch, threshold=255) == 0.0
