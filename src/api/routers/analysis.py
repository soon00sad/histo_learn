"""Analysis endpoints: single-patch (synchronous, seconds) and whole-slide
(asynchronous job, up to ~15 min on CPU — see docs/ALGORITHMS.md).

Segmentation path: the mask itself is the explanation (no GradCAM boxes —
src/xai/ stays untouched but unused here, still needed by the binary path
on master)."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from src.api.case_service import class_areas_to_json, save_image, save_mask_png
from src.api.db import Case, Job, JobStatus, User, generate_case_id, generate_job_id, session_scope
from src.api.deps import current_user, db_session, settings_dep
from src.api.ml_runtime import get_segmenter
from src.api.schemas import AnalysisResult, ClassAreaOut, JobAccepted
from src.inference.verdict import derive_verdict
from src.inference.wsi_segmenter import analyze_wsi_segmentation
from src.utils.bcss_classes import load_bcss_classes
from src.utils.config import Settings, get_settings
from src.utils.logging import get_logger
from src.wsi.reader import OpenSlideUnavailableError, UnsupportedSlideError

logger = get_logger(__name__)
router = APIRouter(prefix="/analyze", tags=["analysis"])

_DEFAULT_TISSUE_TYPE = "Молочная железа, биопсия"


def _class_areas_out(fractions: dict[str, float]) -> list[ClassAreaOut]:
    taxonomy = load_bcss_classes()
    by_name = {c.name_en: c for c in taxonomy.classes}
    return [
        ClassAreaOut(name_en=by_name[name].name_en, name_ru=by_name[name].name_ru,
                     color=by_name[name].color, fraction=fraction)
        for name, fraction in sorted(fractions.items(), key=lambda item: -item[1])
    ]


@router.post("/patch", response_model=AnalysisResult)
async def analyze_patch(
    file: UploadFile = File(...),
    ki67: Optional[float] = Form(None),
    er_status: Optional[str] = Form(None),
    pr_status: Optional[str] = Form(None),
    her2_status: Optional[str] = Form(None),
    tissue_type: str = Form(_DEFAULT_TISSUE_TYPE),
    session: Session = Depends(db_session),
    settings: Settings = Depends(settings_dep),
    user: User = Depends(current_user),
) -> AnalysisResult:
    try:
        image = Image.open(file.file)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не удалось прочитать файл изображения. Поддерживаются форматы PNG и JPEG.",
        ) from exc

    segmenter = get_segmenter()
    result = segmenter.segment(image)
    verdict = derive_verdict(result.class_pixel_fractions, settings)

    case_id = generate_case_id(session)
    source_rel = f"{settings.paths.uploads_dir}/{case_id}.jpg"
    mask_rel = f"{settings.paths.heatmaps_dir}/{case_id}.png"
    save_image(image, settings.resolve_path(source_rel))
    save_mask_png(result.mask, settings.resolve_path(mask_rel))

    case = Case(
        id=case_id,
        doctor_id=user.id,
        tissue_type=tissue_type,
        source_filename=file.filename or f"{case_id}.jpg",
        analysis_mode="patch",
        verdict_label=verdict.verdict_label,
        is_malignant=verdict.is_malignant,
        tumor_area_fraction=verdict.tumor_area_fraction,
        class_areas_json=class_areas_to_json(result.class_pixel_fractions),
        source_image_path=source_rel,
        mask_image_path=mask_rel,
        ki67=ki67,
        er_status=er_status,
        pr_status=pr_status,
        her2_status=her2_status,
    )
    session.add(case)
    session.commit()
    logger.info("Patch segmentation complete for case %s (%s)", case_id, case.verdict_label)

    return AnalysisResult(
        case_id=case.id,
        verdict_label=case.verdict_label,
        is_malignant=case.is_malignant,
        tumor_area_fraction=case.tumor_area_fraction,
        class_areas=_class_areas_out(result.class_pixel_fractions),
        analysis_mode="patch",
    )


@router.post("/wsi", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def analyze_wsi_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ki67: Optional[float] = Form(None),
    er_status: Optional[str] = Form(None),
    pr_status: Optional[str] = Form(None),
    her2_status: Optional[str] = Form(None),
    tissue_type: str = Form(_DEFAULT_TISSUE_TYPE),
    session: Session = Depends(db_session),
    settings: Settings = Depends(settings_dep),
    user: User = Depends(current_user),
) -> JobAccepted:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Файл препарата не передан.")

    job_id = generate_job_id()
    upload_rel = f"{settings.paths.uploads_dir}/{job_id}_{file.filename}"
    upload_path = settings.resolve_path(upload_rel)
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    with upload_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job = Job(id=job_id, status=JobStatus.QUEUED, stage="queued", progress=0.0, message="В очереди")
    session.add(job)
    session.commit()

    background_tasks.add_task(
        _run_wsi_job,
        job_id=job_id,
        slide_path=upload_path,
        source_filename=file.filename,
        tissue_type=tissue_type,
        doctor_id=user.id,
        ki67=ki67,
        er_status=er_status,
        pr_status=pr_status,
        her2_status=her2_status,
    )
    logger.info("Queued WSI segmentation job %s for %s", job_id, file.filename)
    return JobAccepted(job_id=job_id)


def _run_wsi_job(
    job_id: str,
    slide_path: Path,
    source_filename: str,
    tissue_type: str,
    doctor_id: int,
    ki67: Optional[float],
    er_status: Optional[str],
    pr_status: Optional[str],
    her2_status: Optional[str],
) -> None:
    """Runs in a FastAPI BackgroundTasks worker thread: the full WSI
    segmentation pipeline with progress written back to the `jobs` table for
    polling."""
    settings = get_settings()

    with session_scope() as session:
        job = session.get(Job, job_id)
        job.status = JobStatus.RUNNING
        session.commit()

    def on_progress(stage: str, percent: float, message: str) -> None:
        with session_scope() as progress_session:
            progress_job = progress_session.get(Job, job_id)
            progress_job.stage = stage
            progress_job.progress = percent
            progress_job.message = message
            progress_session.commit()

    try:
        result = analyze_wsi_segmentation(
            str(slide_path), settings, segmenter=get_segmenter(), on_progress=on_progress
        )
    except (OpenSlideUnavailableError, UnsupportedSlideError, FileNotFoundError, ValueError) as exc:
        logger.warning("WSI segmentation job %s failed: %s", job_id, exc)
        with session_scope() as session:
            job = session.get(Job, job_id)
            job.status = JobStatus.FAILED
            job.error = str(exc)
            session.commit()
        return
    except Exception as exc:  # unexpected failure — still surface it, don't hang the UI forever
        logger.exception("WSI segmentation job %s failed unexpectedly", job_id)
        with session_scope() as session:
            job = session.get(Job, job_id)
            job.status = JobStatus.FAILED
            job.error = f"Внутренняя ошибка анализа: {exc}"
            session.commit()
        return

    with session_scope() as session:
        case_id = generate_case_id(session)
        thumb_rel = f"{settings.paths.uploads_dir}/{case_id}_thumb.jpg"
        mask_rel = f"{settings.paths.heatmaps_dir}/{case_id}.png"
        save_image(result.thumbnail, settings.resolve_path(thumb_rel))
        save_mask_png(result.mask, settings.resolve_path(mask_rel))

        case = Case(
            id=case_id,
            doctor_id=doctor_id,
            tissue_type=tissue_type,
            source_filename=source_filename,
            analysis_mode="wsi",
            verdict_label=result.verdict.verdict_label,
            is_malignant=result.verdict.is_malignant,
            tumor_area_fraction=result.verdict.tumor_area_fraction,
            class_areas_json=class_areas_to_json(result.class_area_fractions),
            source_image_path=thumb_rel,
            mask_image_path=mask_rel,
            ki67=ki67,
            er_status=er_status,
            pr_status=pr_status,
            her2_status=her2_status,
        )
        session.add(case)
        session.flush()

        job = session.get(Job, job_id)
        job.status = JobStatus.DONE
        job.stage = "done"
        job.progress = 1.0
        job.message = "Анализ завершён"
        job.case_id = case_id
        session.commit()

    logger.info(
        "WSI segmentation job %s complete -> case %s (%s, %d/%d tiles analyzed)",
        job_id, case_id, result.verdict.verdict_label, result.tiles_analyzed, result.tiles_total,
    )
