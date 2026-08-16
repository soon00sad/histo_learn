"""Loading and caching of the EfficientNet-B3 malignant/benign classifier."""
from __future__ import annotations

import timm
import torch
from torch import nn

from src.utils.config import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MODEL_CACHE: dict[str, nn.Module] = {}


def load_model(settings: Settings | None = None) -> nn.Module:
    """Load (and process-wide cache) the classifier with configured weights."""
    settings = settings or get_settings()
    cfg = settings.model
    weights_path = settings.resolve_path(cfg.weights_path)
    cache_key = str(weights_path)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_path}. "
            "Run `python scripts/download_weights.py` first."
        )

    model = timm.create_model(
        cfg.architecture, pretrained=False, num_classes=cfg.num_classes
    )
    state_dict = torch.load(weights_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.to(cfg.device)
    model.eval()

    logger.info(
        "Loaded %s weights from %s onto %s", cfg.architecture, weights_path, cfg.device
    )
    _MODEL_CACHE[cache_key] = model
    return model


def resolve_target_layer(model: nn.Module, layer_path: str):
    """Resolve a dotted path with numeric indices (e.g. 'blocks.-1.-1') to a submodule.

    Matches the prototype's `model.blocks[-1][-1]` GradCAM++ target layer.
    """
    obj = model
    for part in layer_path.split("."):
        if part.lstrip("-").isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj
