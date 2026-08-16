"""Tiling of whole-slide images: splits the tissue region into model-sized tiles.

Two-pass approach so the expensive per-tile work never touches empty glass:
1. Build a coarse tissue mask from a cheap thumbnail (preprocessing.tissue_filter).
2. Only emit level-0 tile coordinates whose corresponding thumbnail region has
   sufficient tissue coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from PIL import Image

from src.preprocessing.tissue_filter import tissue_mask
from src.utils.config import PreprocessingConfig, WsiConfig
from src.wsi.reader import WsiReader


@dataclass(frozen=True)
class TileCoordinate:
    x: int  # level-0 pixel x of the tile's top-left corner
    y: int  # level-0 pixel y of the tile's top-left corner
    index: int


def plan_tiles(
    reader: WsiReader,
    wsi_config: WsiConfig,
    preprocessing_config: PreprocessingConfig,
) -> tuple[list[TileCoordinate], Image.Image]:
    """Return (tissue-containing tile coordinates, thumbnail used for masking)."""
    width, height = reader.dimensions
    downsample = max(wsi_config.tissue_mask_level_downsample, 1)
    thumbnail = reader.read_thumbnail(
        (max(width // downsample, 1), max(height // downsample, 1))
    )
    mask, _ = tissue_mask(thumbnail)

    thumb_w, thumb_h = thumbnail.size
    scale_x = thumb_w / width
    scale_y = thumb_h / height

    tiles: list[TileCoordinate] = []
    index = 0
    for y in range(0, height, wsi_config.tile_stride):
        for x in range(0, width, wsi_config.tile_stride):
            tile_w = min(wsi_config.tile_size, width - x)
            tile_h = min(wsi_config.tile_size, height - y)
            if _tile_has_tissue(
                mask, x, y, tile_w, tile_h, scale_x, scale_y, preprocessing_config
            ):
                tiles.append(TileCoordinate(x=x, y=y, index=index))
                index += 1
    return tiles, thumbnail


def _tile_has_tissue(
    mask: np.ndarray,
    x: int,
    y: int,
    tile_w: int,
    tile_h: int,
    scale_x: float,
    scale_y: float,
    preprocessing_config: PreprocessingConfig,
) -> bool:
    mask_h, mask_w = mask.shape
    x0 = min(int(x * scale_x), mask_w - 1)
    y0 = min(int(y * scale_y), mask_h - 1)
    x1 = max(min(int((x + tile_w) * scale_x), mask_w), x0 + 1)
    y1 = max(min(int((y + tile_h) * scale_y), mask_h), y0 + 1)
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return False
    return float(region.mean()) >= preprocessing_config.min_tissue_fraction


def iter_tile_images(
    reader: WsiReader, tiles: list[TileCoordinate], wsi_config: WsiConfig
) -> Iterator[tuple[TileCoordinate, Image.Image]]:
    """Lazily read each planned tile as an RGB PIL image at level 0."""
    for tile in tiles:
        image = reader.read_region_rgb(
            (tile.x, tile.y), level=0, size=(wsi_config.tile_size, wsi_config.tile_size)
        )
        yield tile, image
