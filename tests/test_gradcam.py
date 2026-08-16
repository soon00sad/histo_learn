"""Tests for GradCAM++ explanation generation (untrained architecture via
the stub_load_model fixture — see tests/conftest.py)."""
from __future__ import annotations

import numpy as np

from src.inference.classifier import PatchClassifier
from src.xai.gradcam import GradCamExplainer


def test_explain_returns_heatmap_matching_input_resolution(stub_load_model, synthetic_he_patch):
    classifier = PatchClassifier()
    explainer = GradCamExplainer(classifier)

    result = classifier.classify(synthetic_he_patch)
    grayscale_cam, overlay = explainer.explain(synthetic_he_patch, target_class=result.predicted_class)

    size = classifier.settings.model.input_size
    assert grayscale_cam.shape == (size, size)
    assert overlay.shape == (size, size, 3)
    assert overlay.dtype == np.uint8
    assert grayscale_cam.min() >= 0.0
    assert grayscale_cam.max() <= 1.0 + 1e-4
