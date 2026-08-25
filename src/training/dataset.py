"""PyTorch Dataset for HistoVision's BCSS segmentation training.

Expects data prepared by `src.training.prepare_bcss` in the layout:

    <root>/images/<id>.png   RGB region/tile
    <root>/masks/<id>.png    single-channel, raw BCSS GT codes as pixel values
    <root>/splits/train.txt  one <id> per line
    <root>/splits/val.txt    one <id> per line

Raw GT codes are remapped to HistoVision's 16-class taxonomy
(config/bcss_classes.yaml) via `remap_mask`; pixels with no mapped class
(the raw "outside_roi" ignore code, or anything unexpected) become
`IGNORE_INDEX` and are excluded from loss and metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from src.preprocessing.stain_normalization import normalize as macenko_normalize
from src.utils.bcss_classes import BcssTaxonomy, load_bcss_classes
from src.utils.config import Settings, get_settings

# BCSS regions run up to ~11800x11200px (132M pixels) — real values seen on a
# live download, comfortably past Pillow's default 89M-pixel
# "DecompressionBombWarning" threshold (meant to flag untrusted uploads, not
# regions from a dataset we just downloaded ourselves via prepare_bcss.py).
Image.MAX_IMAGE_PIXELS = None

IGNORE_INDEX = 255

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class BcssSample:
    id: str
    image_path: Path
    mask_path: Path


def list_samples(root: Path, split: str) -> list[BcssSample]:
    """Read <root>/splits/<split>.txt and pair each id with its image/mask files."""
    split_file = root / "splits" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}. Run prepare_bcss.py first."
        )
    samples: list[BcssSample] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        sample_id = line.strip()
        if not sample_id:
            continue
        image_path = root / "images" / f"{sample_id}.png"
        mask_path = root / "masks" / f"{sample_id}.png"
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(
                f"Missing image/mask for sample '{sample_id}' under {root}"
            )
        samples.append(BcssSample(sample_id, image_path, mask_path))
    return samples


def remap_mask(raw_mask: np.ndarray, taxonomy: BcssTaxonomy) -> np.ndarray:
    """Map raw BCSS GT codes to 0-based model class indices.

    Any code not present in the taxonomy (including the outside_roi ignore
    code) becomes IGNORE_INDEX, matching BCSS's own guidance that
    "don't care" pixels get zero weight during training.
    """
    remapped = np.full(raw_mask.shape, IGNORE_INDEX, dtype=np.uint8)
    for raw_code, model_index in taxonomy.raw_code_to_model_index().items():
        remapped[raw_mask == raw_code] = model_index
    return remapped


class BcssDataset(Dataset):
    """Yields (image_tensor[3,H,W] float, mask_tensor[H,W] long) pairs.

    `crop_size` tiles are randomly cropped (train) or center-cropped (val)
    from each region; regions smaller than `crop_size` are padded (image
    with black, mask with IGNORE_INDEX) rather than skipped, so a tiny
    smoke-test dataset with small sample images still works.
    """

    def __init__(
        self,
        samples: list[BcssSample],
        taxonomy: BcssTaxonomy | None = None,
        crop_size: int = 256,
        augment: bool = False,
        stain_normalize: bool = False,
        settings: Settings | None = None,
    ) -> None:
        self.samples = samples
        self.taxonomy = taxonomy or load_bcss_classes()
        self.crop_size = crop_size
        self.augment = augment
        self.stain_normalize = stain_normalize
        self.settings = settings or get_settings()

        geometric: list[T.Transform] = [
            T.RandomCrop(crop_size, pad_if_needed=True, fill={tv_tensors.Image: 0, tv_tensors.Mask: IGNORE_INDEX}),
        ]
        if augment:
            geometric += [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=90),
            ]
        self._geometric = T.Compose(geometric)
        self._color_jitter = T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02) if augment else None
        self._normalize = T.Compose([
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        raw_mask = np.array(Image.open(sample.mask_path).convert("L"))
        mask = remap_mask(raw_mask, self.taxonomy)

        image_t = tv_tensors.Image(image)
        mask_t = tv_tensors.Mask(torch.from_numpy(mask).to(torch.uint8))

        image_t, mask_t = self._geometric(image_t, mask_t)

        if self.stain_normalize:
            # Applied to the already-cropped patch, not the full BCSS region:
            # this must match inference-time granularity (Macenko runs
            # per-tile in src/inference/wsi_segmenter.py, never on a whole
            # WSI/region) for train/inference consistency. It also matters a
            # lot for speed — BCSS regions run up to ~11800x11200px
            # (median ~4800x3900), and Macenko's cost scales with pixel
            # count: ~10s for a 4500x4500 image vs ~0.04s for a 256x256 crop
            # in practice. Normalizing the full region on every __getitem__
            # call (every sample, every epoch, nothing cached) was what made
            # a training run look hung for hours with no visible progress —
            # see docs/MODEL.md.
            cropped_image = Image.fromarray(image_t.permute(1, 2, 0).numpy())
            normalized_image = macenko_normalize(cropped_image, self.settings.preprocessing.macenko_reference)
            image_t = tv_tensors.Image(torch.from_numpy(np.array(normalized_image)).permute(2, 0, 1))

        if self._color_jitter is not None:
            image_t = self._color_jitter(image_t)
        image_t = self._normalize(image_t)

        return image_t, mask_t.to(torch.int64)
