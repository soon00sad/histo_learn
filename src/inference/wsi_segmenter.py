"""WSI-level semantic segmentation: stitches per-tile DeepLabV3+ predictions
into a single, seam-free class mask for the whole slide.

Reuses src/wsi/ (reader, tiler) and src/preprocessing/ (Macenko normalization)
completely unchanged — only the per-tile inference and the
aggregation-into-one-mask step differ from the binary classification path
(src/inference/wsi_aggregator.py, still used as-is on master).

Seam avoidance: unlike the classification path (non-overlapping tiles, one
scalar per tile), segmentation tiles here overlap (`segmentation_wsi.tile_stride
< tile_size`) and each tile's per-class probability map is accumulated into a
whole-slide canvas through a 2D Hann taper — full weight at a tile's center,
tapering to near-zero at its edges — so adjacent tiles blend instead of
showing a grid of hard borders (a well-known artifact of naive tile-wise CNN
inference). The final mask is downsampled from level 0
(`segmentation_wsi.mask_output_downsample`) since a full-resolution canvas for
a gigapixel slide won't fit in memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.inference.segmenter import PatchSegmenter
from src.preprocessing.stain_normalization import normalize as macenko_normalize
from src.utils.bcss_classes import BcssTaxonomy, load_bcss_classes
from src.utils.config import Settings, get_settings
from src.utils.logging import get_logger
from src.wsi.reader import WsiReader
from src.wsi.tiler import TileCoordinate, iter_tile_images, plan_tiles

logger = get_logger(__name__)

# (stage_key, progress_fraction[0-1], human-readable message)
ProgressCallback = Callable[[str, float, str], None]

# Sentinel for canvas pixels no tile ever covered (background/glass, filtered
# out by plan_tiles' tissue check before reaching the model). Model class
# indices are 0..num_classes-1 — index 0 is a real class (tumor), so it can't
# double as "no data" without pixels outside the tissue silently rendering as
# a real prediction. Matches src/training/dataset.py's IGNORE_INDEX.
UNCOVERED = 255


def _noop_progress(stage: str, fraction: float, message: str) -> None:
    return


@dataclass(frozen=True)
class WsiSegmentationResult:
    mask: np.ndarray  # [canvas_h, canvas_w] uint8, 0-based model class indices, or UNCOVERED (255)
    mask_downsample: float  # level-0 pixels per canvas pixel
    class_area_fractions: dict[str, float]  # name_en -> fraction of *tissue-covered* canvas area
    thumbnail: Image.Image
    tiles_total: int
    tiles_analyzed: int


def hann_window(size: int) -> np.ndarray:
    """2D separable Hann taper: 1.0 at a tile's center, tapering to ~0 at its
    edges, so overlapping tiles blend smoothly at their borders instead of
    producing visible seams. Clipped away from exact zero so a tile with no
    overlapping neighbor (e.g. at the tissue boundary) still contributes."""
    if size == 1:
        return np.ones((1, 1), dtype=np.float32)
    w1d = np.clip(np.hanning(size), 1e-3, None)
    return np.outer(w1d, w1d).astype(np.float32)


def accumulate_tile(
    prob_accum: np.ndarray,
    weight_accum: np.ndarray,
    tile_probs: np.ndarray,
    tile_x: int,
    tile_y: int,
    downsample: float,
    tile_canvas_size: int,
    taper: np.ndarray,
) -> None:
    """Downsample one tile's [num_classes, in_size, in_size] probability map
    to canvas resolution and add it (Hann-weighted) into `prob_accum`
    ([num_classes, canvas_h, canvas_w]) / `weight_accum` ([canvas_h, canvas_w])
    at its level-0 position (tile_x, tile_y). Pure array math — no model or
    WSI reader involved, so this is unit-testable on its own.
    """
    num_classes, canvas_h, canvas_w = prob_accum.shape
    resized = np.stack(
        [
            cv2.resize(tile_probs[c], (tile_canvas_size, tile_canvas_size), interpolation=cv2.INTER_AREA)
            for c in range(num_classes)
        ]
    )

    cx, cy = round(tile_x / downsample), round(tile_y / downsample)
    x0, y0 = max(0, cx), max(0, cy)
    x1, y1 = min(canvas_w, cx + tile_canvas_size), min(canvas_h, cy + tile_canvas_size)
    if x1 <= x0 or y1 <= y0:
        return  # tile falls entirely outside the canvas (rounding at the far edge)

    sx0, sy0 = x0 - cx, y0 - cy
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    w = taper[sy0:sy1, sx0:sx1]
    prob_accum[:, y0:y1, x0:x1] += resized[:, sy0:sy1, sx0:sx1] * w
    weight_accum[y0:y1, x0:x1] += w


def mask_from_accumulators(
    prob_accum: np.ndarray, weight_accum: np.ndarray, taxonomy: BcssTaxonomy
) -> tuple[np.ndarray, dict[str, float]]:
    """Argmax the blended probabilities into a class mask, and compute
    per-class area fractions over tissue-covered pixels only. Pixels no tile
    ever touched (weight==0) get UNCOVERED (255), not 0 — 0 is the real
    "tumor" class, so defaulting to it would make uncovered background
    silently render as a tumor prediction."""
    covered = weight_accum > 0
    mask = np.full(weight_accum.shape, UNCOVERED, dtype=np.uint8)
    mask[covered] = prob_accum[:, covered].argmax(axis=0).astype(np.uint8)

    total_covered = int(covered.sum())
    fractions: dict[str, float] = {}
    if total_covered:
        covered_mask = mask[covered]
        for bcss_class in taxonomy.classes:
            count = int((covered_mask == bcss_class.model_index).sum())
            if count:
                fractions[bcss_class.name_en] = count / total_covered
    return mask, fractions


def analyze_wsi_segmentation(
    slide_path: str,
    settings: Optional[Settings] = None,
    segmenter: Optional[PatchSegmenter] = None,
    on_progress: ProgressCallback = _noop_progress,
) -> WsiSegmentationResult:
    """Run the full WSI segmentation pipeline and return a stitched slide-level mask."""
    settings = settings or get_settings()
    segmenter = segmenter or PatchSegmenter(settings)
    taxonomy = load_bcss_classes()
    wsi_cfg, pre_cfg = settings.segmentation_wsi, settings.preprocessing
    device = settings.segmentation_model.device

    with WsiReader(slide_path) as reader:
        on_progress("tiling", 0.0, "Разбиение препарата на перекрывающиеся фрагменты")
        tiles, thumbnail = plan_tiles(reader, wsi_cfg, pre_cfg)
        tiles_total = len(tiles)
        if tiles_total == 0:
            raise ValueError("На препарате не обнаружено ткани (весь кадр — фон/стекло).")
        on_progress("tiling", 1.0, f"Препарат разбит на {tiles_total} перекрывающихся фрагментов")

        downsample = wsi_cfg.mask_output_downsample
        width, height = reader.dimensions
        canvas_w = max(1, round(width / downsample))
        canvas_h = max(1, round(height / downsample))
        num_classes = taxonomy.num_classes

        prob_accum = np.zeros((num_classes, canvas_h, canvas_w), dtype=np.float32)
        weight_accum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        tile_canvas_size = max(1, round(wsi_cfg.tile_size / downsample))
        taper = hann_window(tile_canvas_size)

        batch_tiles: list[TileCoordinate] = []
        batch_tensors: list[torch.Tensor] = []
        tiles_analyzed = 0
        progress_stride = max(tiles_total // 20, 1)

        def flush_batch() -> None:
            if not batch_tensors:
                return
            batch = torch.cat(batch_tensors, dim=0)
            with torch.no_grad():
                logits = segmenter.model(batch.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # [N, C, in, in]
            for coord, tile_probs in zip(batch_tiles, probs):
                accumulate_tile(
                    prob_accum, weight_accum, tile_probs,
                    coord.x, coord.y, downsample, tile_canvas_size, taper,
                )
            batch_tiles.clear()
            batch_tensors.clear()

        on_progress("inference", 0.0, "Нормализация окраски и посегментная сегментация")
        for i, (coord, tile_image) in enumerate(iter_tile_images(reader, tiles, wsi_cfg)):
            if pre_cfg.stain_normalization_enabled:
                tile_image = macenko_normalize(tile_image, pre_cfg.macenko_reference)

            batch_tiles.append(coord)
            batch_tensors.append(segmenter.preprocess(tile_image))
            if len(batch_tensors) >= wsi_cfg.batch_size:
                flush_batch()

            if (i + 1) % progress_stride == 0 or i + 1 == tiles_total:
                tiles_analyzed = i + 1
                on_progress(
                    "inference", tiles_analyzed / tiles_total,
                    f"Обработано {tiles_analyzed} из {tiles_total} фрагментов",
                )
        flush_batch()
        tiles_analyzed = tiles_total

        on_progress("stitching", 0.0, "Сборка маски препарата из фрагментов")
        mask, fractions = mask_from_accumulators(prob_accum, weight_accum, taxonomy)
        on_progress("stitching", 1.0, "Маска препарата готова")

    return WsiSegmentationResult(
        mask=mask,
        mask_downsample=float(downsample),
        class_area_fractions=fractions,
        thumbnail=thumbnail,
        tiles_total=tiles_total,
        tiles_analyzed=tiles_analyzed,
    )
