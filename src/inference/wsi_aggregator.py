"""WSI-level orchestration: tiling -> tissue filter -> stain normalization ->
batched inference -> aggregation into a slide-level verdict and heatmap.

This is the "divide and analyze" pipeline described in docs/ARCHITECTURE.md.
Each stage reports progress through a callback so the API can surface it to
the "Обработка" (Processing) screen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from src.inference.classifier import ClassificationResult, PatchClassifier
from src.preprocessing.stain_normalization import normalize as macenko_normalize
from src.utils.config import Settings
from src.utils.logging import get_logger
from src.wsi.reader import WsiReader
from src.wsi.tiler import TileCoordinate, iter_tile_images, plan_tiles

logger = get_logger(__name__)

# (stage_key, progress_fraction[0-1], human-readable message)
ProgressCallback = Callable[[str, float, str], None]


@dataclass(frozen=True)
class TileResult:
    tile: TileCoordinate
    result: ClassificationResult


@dataclass(frozen=True)
class WsiAnalysisResult:
    verdict_label: str
    confidence: float
    probabilities: dict[str, float]
    tile_results: list[TileResult]
    thumbnail: Image.Image
    heatmap_overlay: Image.Image
    tiles_total: int
    tiles_analyzed: int


def _noop_progress(stage: str, percent: float, message: str) -> None:
    return


def analyze_wsi(
    slide_path: str,
    settings: Settings,
    classifier: Optional[PatchClassifier] = None,
    on_progress: ProgressCallback = _noop_progress,
) -> WsiAnalysisResult:
    """Run the full WSI pipeline and return a slide-level verdict + heatmap."""
    classifier = classifier or PatchClassifier(settings)
    wsi_cfg, pre_cfg = settings.wsi, settings.preprocessing
    malignant_name = settings.model.class_names[settings.model.malignant_class_index]
    benign_name = next(n for n in settings.model.class_names if n != malignant_name)

    with WsiReader(slide_path) as reader:
        on_progress("tiling", 0.0, "Разбиение препарата на фрагменты")
        tiles, thumbnail = plan_tiles(reader, wsi_cfg, pre_cfg)
        tiles_total = len(tiles)
        if tiles_total == 0:
            raise ValueError("На препарате не обнаружено ткани (весь кадр — фон/стекло).")
        on_progress("tiling", 1.0, f"Препарат разбит на {tiles_total} фрагментов")

        tile_results = _classify_tiles(
            reader, tiles, classifier, wsi_cfg, pre_cfg, tiles_total, on_progress
        )

        on_progress("heatmap", 0.0, "Построение тепловой карты")
        heatmap_overlay = _build_heatmap_overlay(
            thumbnail, reader.dimensions, tile_results, wsi_cfg, malignant_name
        )
        on_progress("heatmap", 1.0, "Тепловая карта готова")

    malignant_probs = np.array([tr.result.probabilities[malignant_name] for tr in tile_results])
    # Slide-level malignancy: mean of the most suspicious decile of tiles, so
    # one strongly atypical focus drives the verdict instead of being diluted
    # by thousands of unremarkable tiles (matches how a pathologist scans).
    top_decile_count = max(1, len(malignant_probs) // 10)
    slide_malignant_prob = float(np.sort(malignant_probs)[-top_decile_count:].mean())

    if slide_malignant_prob >= 0.5:
        verdict_label, confidence = malignant_name, slide_malignant_prob
    else:
        verdict_label, confidence = benign_name, 1 - slide_malignant_prob

    return WsiAnalysisResult(
        verdict_label=verdict_label,
        confidence=confidence,
        probabilities={malignant_name: slide_malignant_prob, benign_name: 1 - slide_malignant_prob},
        tile_results=tile_results,
        thumbnail=thumbnail,
        heatmap_overlay=heatmap_overlay,
        tiles_total=tiles_total,
        tiles_analyzed=len(tile_results),
    )


def _classify_tiles(
    reader: WsiReader,
    tiles: list[TileCoordinate],
    classifier: PatchClassifier,
    wsi_cfg,
    pre_cfg,
    tiles_total: int,
    on_progress: ProgressCallback,
) -> list[TileResult]:
    tile_results: list[TileResult] = []
    batch_tensors: list[torch.Tensor] = []
    batch_coords: list[TileCoordinate] = []
    progress_stride = max(tiles_total // 20, 1)

    def flush_batch() -> None:
        if not batch_tensors:
            return
        stacked = torch.cat(batch_tensors, dim=0)
        for coord, result in zip(batch_coords, classifier.classify_tensor_batch(stacked)):
            tile_results.append(TileResult(tile=coord, result=result))
        batch_tensors.clear()
        batch_coords.clear()

    on_progress("normalization", 0.0, "Нормализация окраски и анализ нейросетью")
    for i, (coord, tile_image) in enumerate(iter_tile_images(reader, tiles, wsi_cfg)):
        if pre_cfg.stain_normalization_enabled:
            tile_image = macenko_normalize(tile_image, pre_cfg.macenko_reference)

        batch_tensors.append(classifier.preprocess(tile_image))
        batch_coords.append(coord)
        if len(batch_tensors) >= wsi_cfg.batch_size:
            flush_batch()

        if (i + 1) % progress_stride == 0 or i + 1 == tiles_total:
            on_progress(
                "inference",
                (i + 1) / tiles_total,
                f"Обработано {i + 1} из {tiles_total} фрагментов",
            )
    flush_batch()
    return tile_results


def _build_heatmap_overlay(
    thumbnail: Image.Image,
    slide_dimensions: tuple[int, int],
    tile_results: list[TileResult],
    wsi_config,
    malignant_name: str,
) -> Image.Image:
    """Paint each tile's malignancy probability onto a thumbnail-resolution
    heatmap (blue = low, red = high), alpha-blended over the tissue thumbnail."""
    width, height = slide_dimensions
    thumb_w, thumb_h = thumbnail.size
    scale_x, scale_y = thumb_w / width, thumb_h / height

    heat = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    coverage = np.zeros((thumb_h, thumb_w), dtype=np.float32)

    for tr in tile_results:
        malignant_prob = tr.result.probabilities[malignant_name]
        x0 = int(tr.tile.x * scale_x)
        y0 = int(tr.tile.y * scale_y)
        x1 = max(int((tr.tile.x + wsi_config.tile_size) * scale_x), x0 + 1)
        y1 = max(int((tr.tile.y + wsi_config.tile_size) * scale_y), y0 + 1)
        heat[y0:y1, x0:x1] += malignant_prob
        coverage[y0:y1, x0:x1] += 1

    tissue_present = coverage > 0
    coverage[~tissue_present] = 1  # avoid divide-by-zero; result is masked out below
    heat = heat / coverage

    heat_uint8 = np.uint8(np.clip(heat, 0, 1) * 255)
    colored_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    base = np.array(thumbnail.convert("RGB")).astype(np.float32)
    alpha = 0.45
    blended = base.copy()
    mask3 = np.repeat(tissue_present[:, :, None], 3, axis=2)
    blended[mask3] = (1 - alpha) * base[mask3] + alpha * colored_rgb[mask3]
    return Image.fromarray(blended.astype(np.uint8))
