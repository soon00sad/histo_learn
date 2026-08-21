"""Process-wide singleton for the segmentation model.

Loading DeepLabV3+ is expensive — built once per process and reused, so a
request never pays that cost. Simpler than the old binary path's
classifier+explainer pair (see git history) since segmentation needs no
GradCAM hooks: the mask itself is the explanation.

This means a single process handles inference serially per request; scaling
past that is a multi-worker deployment concern (see docs/ARCHITECTURE.md),
not something this module needs to solve for the on-premise MVP.
"""
from __future__ import annotations

import threading
from typing import Optional

from src.inference.segmenter import PatchSegmenter

_lock = threading.Lock()
_segmenter: Optional[PatchSegmenter] = None


def get_segmenter() -> PatchSegmenter:
    global _segmenter
    if _segmenter is None:
        with _lock:
            if _segmenter is None:
                _segmenter = PatchSegmenter()
    return _segmenter
