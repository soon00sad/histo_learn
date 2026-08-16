"""GradCAM++ explanation heatmaps for the patch classifier.

Reuses the prototype's exact target layer (model.blocks[-1][-1]) so the
visual explanations match the validated HuggingFace demo.
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.inference.classifier import PatchClassifier
from src.inference.model import resolve_target_layer
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _denormalize(tensor: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    denorm = (tensor.cpu() * std_t + mean_t).clamp(0, 1)
    return denorm.permute(1, 2, 0).numpy()


class GradCamExplainer:
    """Produces a GradCAM++ heatmap + RGB overlay for a classified patch."""

    def __init__(self, classifier: PatchClassifier):
        self.classifier = classifier
        target_layer = resolve_target_layer(
            classifier.model, classifier.settings.model.gradcam_target_layer
        )
        self._cam = GradCAMPlusPlus(model=classifier.model, target_layers=[target_layer])

    def explain(self, image: Image.Image, target_class: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (grayscale_cam [H,W] in [0,1], rgb_overlay uint8 [H,W,3])."""
        input_tensor = self.classifier.preprocess(image)
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = self._cam(input_tensor=input_tensor, targets=targets)[0]

        model_cfg = self.classifier.settings.model
        img_np = _denormalize(input_tensor[0], model_cfg.normalize_mean, model_cfg.normalize_std)
        overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        return grayscale_cam, overlay
