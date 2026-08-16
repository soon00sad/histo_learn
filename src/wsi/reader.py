"""OpenSlide-backed reading of whole-slide images (WSI).

Reads the multi-resolution pyramid without loading the full gigapixel image
into memory — only requested regions/levels are decoded on demand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image

from src.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import openslide

    _OPENSLIDE_IMPORT_ERROR: Optional[Exception] = None
except (ImportError, OSError) as exc:  # OSError: native libopenslide missing
    openslide = None  # type: ignore[assignment]
    _OPENSLIDE_IMPORT_ERROR = exc


class OpenSlideUnavailableError(RuntimeError):
    """The OpenSlide native library isn't installed on this host.

    `pip install openslide-python` only installs the Python bindings — the
    OpenSlide C library itself must be present separately. On Windows,
    download the binaries from https://openslide.org/download/ and add
    them to PATH; on Linux, install `libopenslide-dev` (already handled in
    the project's Docker image). See README.md for details.
    """


class UnsupportedSlideError(ValueError):
    """A file could not be opened as a whole-slide image (corrupt/unsupported format)."""


class WsiReader:
    """Thin, defensive wrapper around openslide.OpenSlide."""

    def __init__(self, path: str | Path):
        if openslide is None:
            raise OpenSlideUnavailableError(
                "OpenSlide native library is not available on this host "
                f"(original error: {_OPENSLIDE_IMPORT_ERROR})."
            )

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"WSI file not found: {self.path}")

        try:
            self._slide = openslide.OpenSlide(str(self.path))
        except openslide.OpenSlideError as exc:
            raise UnsupportedSlideError(
                f"Could not open '{self.path.name}' as a whole-slide image: {exc}"
            ) from exc

    @property
    def dimensions(self) -> tuple[int, int]:
        """Full-resolution (width, height) in pixels at level 0."""
        return self._slide.dimensions

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    @property
    def level_dimensions(self) -> tuple[tuple[int, int], ...]:
        return self._slide.level_dimensions

    @property
    def level_downsamples(self) -> tuple[float, ...]:
        return self._slide.level_downsamples

    def best_level_for_downsample(self, downsample: float) -> int:
        """Pick the pyramid level whose downsample factor is closest to (but
        not greater than) the requested factor, to avoid upsampling."""
        return self._slide.get_best_level_for_downsample(downsample)

    def read_region_rgb(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> Image.Image:
        """Read a region (level-0 pixel coordinates for `location`) as RGB."""
        region = self._slide.read_region(location, level, size)
        return region.convert("RGB")

    def read_thumbnail(self, max_size: tuple[int, int] = (1024, 1024)) -> Image.Image:
        """Read a low-resolution thumbnail of the whole slide, for tissue masking."""
        return self._slide.get_thumbnail(max_size)

    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> "WsiReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
