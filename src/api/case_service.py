"""Shared helpers for building Case rows and converting between the ORM,
API schemas, and PDF report representations. Used by the analysis, cases,
and reports routers so this logic lives in exactly one place.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from src.api.db import Case
from src.api.schemas import CaseDetail, CaseSummary, RegionOut
from src.report.pdf_report import ReportData, ReportRegion
from src.utils.config import Settings
from src.xai.regions import AttentionRegion

_VERDICT_MALIGNANT = "Злокачественная"
_VERDICT_BENIGN = "Доброкачественная"


def verdict_label_ru(is_malignant: bool) -> str:
    return _VERDICT_MALIGNANT if is_malignant else _VERDICT_BENIGN


def regions_to_json(regions: Sequence[AttentionRegion]) -> str:
    return json.dumps(
        [{"rank": i + 1, "x": r.x, "y": r.y, "score": r.score} for i, r in enumerate(regions)]
    )


def json_to_region_out(raw: str) -> list[RegionOut]:
    return [RegionOut(**item) for item in json.loads(raw)]


def save_image(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=90)
    return path


def case_to_summary(case: Case) -> CaseSummary:
    return CaseSummary(
        id=case.id,
        created_at=case.created_at,
        tissue_type=case.tissue_type,
        verdict_label=case.verdict_label,
        is_malignant=case.is_malignant,
        confidence=case.confidence,
        status=case.status.value,
    )


def case_to_detail(case: Case) -> CaseDetail:
    summary = case_to_summary(case)
    return CaseDetail(
        **summary.model_dump(),
        source_filename=case.source_filename,
        analysis_mode=case.analysis_mode,
        malignant_probability=case.malignant_probability,
        benign_probability=case.benign_probability,
        top_regions=json_to_region_out(case.top_regions_json),
        ki67=case.ki67,
        er_status=case.er_status,
        pr_status=case.pr_status,
        her2_status=case.her2_status,
        report_available=bool(case.report_pdf_path and Path(case.report_pdf_path).exists()),
    )


def build_report_data(case: Case, settings: Settings, doctor_name: str) -> ReportData:
    regions = json_to_region_out(case.top_regions_json)
    return ReportData(
        case_id=case.id,
        created_at=case.created_at.strftime("%d.%m.%Y, %H:%M"),
        tissue_type=case.tissue_type,
        source_filename=case.source_filename,
        analysis_mode="Полный препарат" if case.analysis_mode == "wsi" else "Живой анализ",
        verdict_label=case.verdict_label,
        is_malignant=case.is_malignant,
        confidence=case.confidence,
        malignant_probability=case.malignant_probability,
        benign_probability=case.benign_probability,
        top_regions=[ReportRegion(rank=r.rank, x=r.x, y=r.y, score=r.score) for r in regions],
        heatmap_image_path=settings.resolve_path(case.heatmap_image_path),
        disclaimer=settings.app.disclaimer,
        model_version=settings.app.version,
        generated_at=dt.datetime.utcnow().strftime("%d.%m.%Y, %H:%M"),
        doctor_name=doctor_name,
        logo_path=None,
    )
