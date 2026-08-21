"""Shared helpers for building Case rows and converting between the ORM,
API schemas, and PDF report representations. Used by the analysis, cases,
and reports routers so this logic lives in exactly one place.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.api.db import Case
from src.api.schemas import CaseDetail, CaseSummary, ClassAreaOut
from src.inference.wsi_segmenter import UNCOVERED
from src.report.pdf_report import ReportClassArea, ReportData
from src.utils.bcss_classes import BcssTaxonomy, load_bcss_classes
from src.utils.config import Settings


def class_areas_to_json(fractions: dict[str, float]) -> str:
    return json.dumps(fractions)


def json_to_class_areas(raw: str, taxonomy: Optional[BcssTaxonomy] = None) -> list[ClassAreaOut]:
    taxonomy = taxonomy or load_bcss_classes()
    by_name = {c.name_en: c for c in taxonomy.classes}
    fractions: dict[str, float] = json.loads(raw)

    result: list[ClassAreaOut] = []
    for name, fraction in sorted(fractions.items(), key=lambda item: -item[1]):
        bcss_class = by_name.get(name)
        if bcss_class is None:
            continue  # defensive: ignore an unrecognized class name rather than 500
        result.append(
            ClassAreaOut(name_en=bcss_class.name_en, name_ru=bcss_class.name_ru,
                         color=bcss_class.color, fraction=fraction)
        )
    return result


def save_image(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=90)
    return path


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def save_mask_png(mask: np.ndarray, path: Path, taxonomy: Optional[BcssTaxonomy] = None) -> Path:
    """Lossless indexed PNG: each pixel's byte value *is* its model class
    index — JPEG's lossy compression would corrupt exact class IDs, which a
    heatmap-style scalar overlay never needed to worry about. The palette
    maps indices to config/bcss_classes.yaml's colors; UNCOVERED (255,
    background/glass no tile ever touched — see wsi_segmenter.py) gets a
    fully transparent palette entry via a tRNS chunk, so the frontend's
    mix-blend-mode overlay shows the tissue image through it instead of a
    stray color (this is also the pixel value single-patch segmentation
    never produces, since the whole input patch is always covered).
    """
    taxonomy = taxonomy or load_bcss_classes()
    path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(mask, mode="P")
    palette = [0, 0, 0] * 256
    alpha = bytearray([255] * 256)
    for bcss_class in taxonomy.classes:
        r, g, b = _hex_to_rgb(bcss_class.color)
        offset = bcss_class.model_index * 3
        palette[offset:offset + 3] = [r, g, b]
    alpha[UNCOVERED] = 0
    img.putpalette(palette)
    img.save(path, format="PNG", transparency=bytes(alpha))
    return path


def case_to_summary(case: Case) -> CaseSummary:
    return CaseSummary(
        id=case.id,
        created_at=case.created_at,
        tissue_type=case.tissue_type,
        verdict_label=case.verdict_label,
        is_malignant=case.is_malignant,
        tumor_area_fraction=case.tumor_area_fraction,
        status=case.status.value,
    )


def case_to_detail(case: Case) -> CaseDetail:
    summary = case_to_summary(case)
    return CaseDetail(
        **summary.model_dump(),
        source_filename=case.source_filename,
        analysis_mode=case.analysis_mode,
        class_areas=json_to_class_areas(case.class_areas_json),
        ki67=case.ki67,
        er_status=case.er_status,
        pr_status=case.pr_status,
        her2_status=case.her2_status,
        report_available=bool(case.report_pdf_path and Path(case.report_pdf_path).exists()),
    )


def build_report_data(case: Case, settings: Settings, doctor_name: str) -> ReportData:
    class_areas = [
        ReportClassArea(name_ru=c.name_ru, color=c.color, fraction=c.fraction)
        for c in json_to_class_areas(case.class_areas_json)
    ]

    return ReportData(
        case_id=case.id,
        created_at=case.created_at.strftime("%d.%m.%Y, %H:%M"),
        tissue_type=case.tissue_type,
        source_filename=case.source_filename,
        analysis_mode="Полный препарат" if case.analysis_mode == "wsi" else "Живой анализ",
        verdict_label=case.verdict_label,
        is_malignant=case.is_malignant,
        tumor_area_fraction=case.tumor_area_fraction,
        class_areas=class_areas,
        source_image_path=settings.resolve_path(case.source_image_path),
        mask_image_path=settings.resolve_path(case.mask_image_path),
        disclaimer=settings.app.disclaimer,
        model_version=settings.app.version,
        generated_at=dt.datetime.utcnow().strftime("%d.%m.%Y, %H:%M"),
        doctor_name=doctor_name,
        logo_path=None,
    )
