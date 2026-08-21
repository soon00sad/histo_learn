"""Shared pytest fixtures: synthetic H&E-like images and a fast, weights-free
test model so the suite never needs to download or load the real 43 MB
checkpoint."""
from __future__ import annotations

import numpy as np
import pytest
import segmentation_models_pytorch as smp
import timm
from PIL import Image

from src.utils.bcss_classes import load_bcss_classes
from src.utils.config import get_settings


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture
def synthetic_he_patch() -> Image.Image:
    """A 256x256 RGB image with saturated purple/pink blobs (tissue) on a
    near-white background (glass) — enough structure to exercise tissue
    filtering, stain normalization, and GradCAM without a real slide."""
    rng = np.random.default_rng(42)
    canvas = np.full((256, 256, 3), 245, dtype=np.uint8)  # near-white glass

    for center, color, radius in [
        ((80, 80), (150, 60, 140), 40),   # hematoxylin-like purple
        ((170, 150), (210, 110, 150), 50),  # eosin-like pink
    ]:
        yy, xx = np.ogrid[:256, :256]
        mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2
        noise = rng.integers(-10, 10, size=(mask.sum(), 3))
        canvas[mask] = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(canvas)


@pytest.fixture
def blank_glass_patch() -> Image.Image:
    """A tile that is entirely background glass — should be filtered out."""
    return Image.fromarray(np.full((256, 256, 3), 250, dtype=np.uint8))


@pytest.fixture
def stub_load_model(monkeypatch, settings):
    """Monkeypatch model loading to build an untrained architecture instead
    of reading models/model.pth, so ML-layer tests don't depend on the
    downloaded checkpoint being present."""

    def _fake_load_model(_settings=None):
        cfg = settings.model
        model = timm.create_model(cfg.architecture, pretrained=False, num_classes=cfg.num_classes)
        model.eval()
        return model

    monkeypatch.setattr("src.inference.classifier.load_model", _fake_load_model)
    return _fake_load_model


@pytest.fixture
def stub_load_segmentation_model(monkeypatch, settings):
    """Same idea as stub_load_model, for the DeepLabV3+ segmentation model —
    untrained (encoder_weights=None) so tests don't need models/segmentation.pth."""

    def _fake_load_segmentation_model(_settings=None):
        cfg = settings.segmentation_model
        num_classes = load_bcss_classes().num_classes
        model = smp.DeepLabV3Plus(
            encoder_name=cfg.encoder_name, encoder_weights=None, in_channels=3, classes=num_classes
        )
        model.eval()
        return model

    monkeypatch.setattr("src.inference.segmenter.load_segmentation_model", _fake_load_segmentation_model)
    return _fake_load_segmentation_model
