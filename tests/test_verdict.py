"""Tests for verdict derivation from tissue-class-area fractions."""
from __future__ import annotations

import pytest

from src.inference.verdict import derive_verdict


def test_malignant_when_tumor_fraction_exceeds_threshold(settings):
    threshold = settings.segmentation_verdict.malignant_area_threshold
    result = derive_verdict({"tumor": threshold + 0.01, "stroma": 0.5})

    assert result.is_malignant is True
    assert result.verdict_label == "Злокачественная"
    assert result.tumor_area_fraction == pytest.approx(threshold + 0.01)


def test_benign_when_tumor_fraction_below_threshold(settings):
    threshold = settings.segmentation_verdict.malignant_area_threshold
    result = derive_verdict({"tumor": threshold / 2, "stroma": 0.9})

    assert result.is_malignant is False
    assert result.verdict_label == "Доброкачественная"


def test_tumor_and_dcis_fractions_are_combined(settings):
    threshold = settings.segmentation_verdict.malignant_area_threshold
    # Neither alone crosses the threshold, but together they do.
    result = derive_verdict({"tumor": threshold * 0.6, "dcis": threshold * 0.6})

    assert result.is_malignant is True
    assert result.tumor_area_fraction == pytest.approx(threshold * 1.2)


def test_no_malignant_classes_present_is_benign_with_zero_fraction():
    result = derive_verdict({"stroma": 0.7, "fat": 0.3})

    assert result.is_malignant is False
    assert result.tumor_area_fraction == 0.0


def test_empty_fractions_is_benign():
    result = derive_verdict({})

    assert result.is_malignant is False
    assert result.tumor_area_fraction == 0.0


def test_unknown_class_name_raises():
    with pytest.raises(ValueError, match="not-a-real-class"):
        derive_verdict({"not-a-real-class": 0.5})


def test_exactly_at_threshold_counts_as_malignant(settings):
    """>=, not >, matches the docstring: even a borderline focus should not
    be silently rounded down to benign."""
    threshold = settings.segmentation_verdict.malignant_area_threshold
    result = derive_verdict({"tumor": threshold})

    assert result.is_malignant is True
