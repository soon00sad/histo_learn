"""Download the EfficientNet-B3 classifier weights from HuggingFace into models/.

Usage:
    python scripts/download_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_settings  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

WEIGHTS_URL = (
    "https://huggingface.co/spaces/ViktoriaVladlenovna1/histovision/"
    "resolve/main/model.pth"
)


def download_weights(force: bool = False) -> Path:
    """Download model.pth to the path configured in config.yaml (model.weights_path)."""
    settings = get_settings()
    dest = settings.resolve_path(settings.model.weights_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        logger.info("Weights already present at %s, skipping download.", dest)
        return dest

    logger.info("Downloading model weights from %s", WEIGHTS_URL)
    response = requests.get(WEIGHTS_URL, stream=True, timeout=120)
    response.raise_for_status()

    tmp_path = dest.with_suffix(".part")
    with tmp_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    tmp_path.replace(dest)

    logger.info("Saved weights to %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


if __name__ == "__main__":
    download_weights(force="--force" in sys.argv)
