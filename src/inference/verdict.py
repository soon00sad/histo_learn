"""Verdict derivation from tissue-class-area fractions.

Shared between single-patch (src/inference/segmenter.py's SegmentationResult)
and WSI-level (src/inference/wsi_segmenter.py's WsiSegmentationResult)
results — both expose the same `dict[name_en, fraction]` shape, so this only
needs to be written once. Class names/threshold are config-driven
(config.yaml -> segmentation_verdict), not hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.utils.bcss_classes import BcssTaxonomy, load_bcss_classes
from src.utils.config import Settings, get_settings


@dataclass(frozen=True)
class SegmentationVerdict:
    verdict_label: str
    is_malignant: bool
    tumor_area_fraction: float  # combined area fraction of the configured malignant classes


def derive_verdict(
    class_fractions: dict[str, float],
    settings: Settings | None = None,
    taxonomy: BcssTaxonomy | None = None,
) -> SegmentationVerdict:
    """`class_fractions` — output of SegmentationResult.class_pixel_fractions
    or WsiSegmentationResult.class_area_fractions (absent classes = 0)."""
    settings = settings or get_settings()
    taxonomy = taxonomy or load_bcss_classes()
    cfg = settings.segmentation_verdict

    known = {c.name_en for c in taxonomy.classes}
    unknown = set(class_fractions) - known
    if unknown:
        raise ValueError(f"Unknown class name(s) in class_fractions: {sorted(unknown)}")

    tumor_fraction = sum(class_fractions.get(name, 0.0) for name in cfg.malignant_class_names)
    is_malignant = tumor_fraction >= cfg.malignant_area_threshold

    return SegmentationVerdict(
        verdict_label="Злокачественная" if is_malignant else "Доброкачественная",
        is_malignant=is_malignant,
        tumor_area_fraction=tumor_fraction,
    )
