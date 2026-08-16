"""Tests for patch-level classification.

Uses the stub_load_model fixture (tests/conftest.py) to build an untrained
EfficientNet-B3 instead of loading models/model.pth, so this suite needs no
network access and doesn't depend on the downloaded checkpoint.
"""
from __future__ import annotations

import pytest
import torch

from src.inference.classifier import PatchClassifier


def test_classify_returns_valid_probability_distribution(stub_load_model, synthetic_he_patch):
    classifier = PatchClassifier()
    result = classifier.classify(synthetic_he_patch)

    assert result.predicted_label in classifier.settings.model.class_names
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert 0.0 <= result.confidence <= 1.0
    assert result.probabilities[result.predicted_label] == pytest.approx(result.confidence)


def test_preprocess_produces_expected_tensor_shape(stub_load_model, synthetic_he_patch):
    classifier = PatchClassifier()
    tensor = classifier.preprocess(synthetic_he_patch)

    size = classifier.settings.model.input_size
    assert tensor.shape == (1, 3, size, size)


def test_classify_tensor_batch_matches_individual_calls(stub_load_model, synthetic_he_patch):
    classifier = PatchClassifier()
    single = classifier.classify(synthetic_he_patch)

    batch_tensor = torch.cat([classifier.preprocess(synthetic_he_patch)] * 2, dim=0)
    batch_results = classifier.classify_tensor_batch(batch_tensor)

    assert len(batch_results) == 2
    assert batch_results[0].predicted_label == single.predicted_label
    assert batch_results[0].probabilities == pytest.approx(single.probabilities, abs=1e-4)
