"""PDF report generation and retrieval for a case."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.api.case_service import build_report_data
from src.api.db import Case, User
from src.api.deps import current_user, db_session, settings_dep
from src.api.schemas import ReportUrl
from src.report.pdf_report import generate_pdf_report
from src.utils.config import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/cases", tags=["reports"])


def _get_case_or_404(session: Session, case_id: str) -> Case:
    case = session.get(Case, case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Случай не найден")
    return case


@router.post("/{case_id}/report", response_model=ReportUrl)
def create_report(
    case_id: str,
    session: Session = Depends(db_session),
    settings: Settings = Depends(settings_dep),
    user: User = Depends(current_user),
) -> ReportUrl:
    case = _get_case_or_404(session, case_id)

    report_rel = f"{settings.paths.reports_dir}/{case.id}.pdf"
    report_path = settings.resolve_path(report_rel)
    data = build_report_data(case, settings, doctor_name=user.full_name)
    generate_pdf_report(data, report_path)

    case.report_pdf_path = report_rel
    session.commit()
    logger.info("Report generated for case %s by %s", case.id, user.email)

    return ReportUrl(report_url=f"/api/v1/cases/{case.id}/report.pdf")


@router.get("/{case_id}/report.pdf")
def download_report(
    case_id: str,
    session: Session = Depends(db_session),
    settings: Settings = Depends(settings_dep),
    user: User = Depends(current_user),
) -> FileResponse:
    case = _get_case_or_404(session, case_id)

    report_rel = case.report_pdf_path or f"{settings.paths.reports_dir}/{case.id}.pdf"
    report_path = settings.resolve_path(report_rel)
    if not report_path.exists():
        # Generate on first request rather than requiring the client to call
        # POST /report first — matches the design's single "Сформировать
        # PDF-отчёт" button.
        data = build_report_data(case, settings, doctor_name=user.full_name)
        generate_pdf_report(data, report_path)
        case.report_pdf_path = report_rel
        session.commit()

    return FileResponse(
        report_path, media_type="application/pdf", filename=f"HistoVision_{case.id}.pdf"
    )
