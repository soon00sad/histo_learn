"""Loader for config/bcss_classes.yaml — the HistoVision segmentation class
taxonomy (16 classes merged from the official BCSS 21-class + ignore scheme).

Single source of truth shared by dataset preparation, training, inference,
WSI aggregation, and the frontend legend. See docs/MODEL.md for the mapping
rationale.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from src.utils.config import PROJECT_ROOT

BCSS_CLASSES_PATH = PROJECT_ROOT / "config" / "bcss_classes.yaml"


class BcssClass(BaseModel):
    id: int  # 1-based, matches the table in config/bcss_classes.yaml
    name_en: str
    name_ru: str
    color: str
    raw_codes: list[int]

    @property
    def model_index(self) -> int:
        """0-based index into the model's class-probability/logit dimension."""
        return self.id - 1


class BcssTaxonomy(BaseModel):
    ignore_raw_code: int
    classes: list[BcssClass]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def raw_code_to_model_index(self) -> dict[int, int]:
        """Every raw BCSS GT code -> 0-based model class index (ignore_raw_code excluded)."""
        mapping: dict[int, int] = {}
        for bcss_class in self.classes:
            for raw_code in bcss_class.raw_codes:
                mapping[raw_code] = bcss_class.model_index
        return mapping

    def class_by_model_index(self, index: int) -> BcssClass:
        for bcss_class in self.classes:
            if bcss_class.model_index == index:
                return bcss_class
        raise KeyError(f"No BCSS class with model_index={index}")


@lru_cache(maxsize=1)
def load_bcss_classes() -> BcssTaxonomy:
    """Load and cache the BCSS class taxonomy for the process lifetime."""
    if not BCSS_CLASSES_PATH.exists():
        raise FileNotFoundError(
            f"BCSS class taxonomy not found at {BCSS_CLASSES_PATH}."
        )
    with BCSS_CLASSES_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return BcssTaxonomy(**raw)
