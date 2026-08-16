"""Patch-level classification: H&E image tile -> malignant/benign probabilities."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms

from src.inference.model import load_model
from src.utils.config import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ClassificationResult:
    predicted_class: int
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


def build_transform(settings: Settings) -> transforms.Compose:
    cfg = settings.model
    return transforms.Compose(
        [
            transforms.Resize((cfg.input_size, cfg.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalize_mean, cfg.normalize_std),
        ]
    )


class PatchClassifier:
    """Wraps the loaded model and its preprocessing for single/batched inference."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.model = load_model(self.settings)
        self.transform = build_transform(self.settings)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Return a [1, C, H, W] tensor ready for the model, on the configured device."""
        tensor = self.transform(image.convert("RGB"))
        return tensor.unsqueeze(0).to(self.settings.model.device)

    def _result_from_probs(self, probs: torch.Tensor) -> ClassificationResult:
        class_names = self.settings.model.class_names
        predicted_class = int(probs.argmax().item())
        return ClassificationResult(
            predicted_class=predicted_class,
            predicted_label=class_names[predicted_class],
            confidence=float(probs[predicted_class].item()),
            probabilities={name: float(probs[i].item()) for i, name in enumerate(class_names)},
        )

    @torch.no_grad()
    def classify(self, image: Image.Image) -> ClassificationResult:
        """Classify a single PIL image."""
        logits = self.model(self.preprocess(image))
        probs = torch.softmax(logits, dim=1)[0]
        return self._result_from_probs(probs)

    @torch.no_grad()
    def classify_tensor_batch(self, batch: torch.Tensor) -> list[ClassificationResult]:
        """Classify a pre-stacked [N, C, H, W] batch (used by the WSI pipeline)."""
        logits = self.model(batch.to(self.settings.model.device))
        probs = torch.softmax(logits, dim=1)
        return [self._result_from_probs(probs[i]) for i in range(probs.shape[0])]
