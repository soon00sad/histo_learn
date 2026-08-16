"""Process-wide singletons for the ML runtime (model + GradCAM++ explainer).

Loading EfficientNet-B3 and registering GradCAM++ hooks is expensive, and
the hooks must not be re-registered per request — pytorch_grad_cam attaches
forward/backward hooks to the underlying model on construction and only
detaches them via explicit release(), so creating a new explainer per
request would leak hooks. Both are built once per process and reused.

This means a single process handles inference serially per request; scaling
past that is a multi-worker deployment concern (see docs/ARCHITECTURE.md),
not something this module needs to solve for the on-premise MVP.
"""
from __future__ import annotations

import threading
from typing import Optional

from src.inference.classifier import PatchClassifier
from src.xai.gradcam import GradCamExplainer

_lock = threading.Lock()
_classifier: Optional[PatchClassifier] = None
_explainer: Optional[GradCamExplainer] = None


def get_classifier() -> PatchClassifier:
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                _classifier = PatchClassifier()
    return _classifier


def get_explainer() -> GradCamExplainer:
    global _explainer
    if _explainer is None:
        with _lock:
            if _explainer is None:
                _explainer = GradCamExplainer(get_classifier())
    return _explainer
