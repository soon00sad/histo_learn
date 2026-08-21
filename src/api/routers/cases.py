"""Case history endpoints: listing, detail, status updates, and image retrieval."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from src.api.case_service import case_to_detail, case_to_summary
from src.api.db import Case, CaseStatus, User
from src.api.deps import current_user, db_session, settings_dep
from src.api.schemas import CaseDetail, CaseStatusUpdate, CaseSummary
from src.utils.config import Settings

router = APIRouter(prefix="/cases", tags=["cases"])


def _get_case_or_404(session: Session, case_id: str) -> Case:
    case = session.get(Case, case_id)
    if case is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Случай не найден")
    return case


@router.get("", response_model=list[CaseSummary])
def list_cases(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    verdict: Optional[str] = Query(default=None, description="malignant | benign"),
    search: Optional[str] = Query(default=None, description="Поиск по ID случая"),
    session: Session = Depends(db_session),
    _user: User = Depends(current_user),
) -> list[CaseSummary]:
    query = session.query(Case)
    if status_filter:
        query = query.filter(Case.status == CaseStatus(status_filter))
    if verdict == "malignant":
        query = query.filter(Case.is_malignant.is_(True))
    elif verdict == "benign":
        query = query.filter(Case.is_malignant.is_(False))
    if search:
        query = query.filter(Case.id.ilike(f"%{search}%"))

    cases = query.order_by(Case.created_at.desc()).all()
    return [case_to_summary(c) for c in cases]


@router.get("/{case_id}", response_model=CaseDetail)
def get_case(
    case_id: str, session: Session = Depends(db_session), _user: User = Depends(current_user)
) -> CaseDetail:
    return case_to_detail(_get_case_or_404(session, case_id))


@router.patch("/{case_id}", response_model=CaseSummary)
def update_case_status(
    case_id: str,
    payload: CaseStatusUpdate,
    session: Session = Depends(db_session),
    _user: User = Depends(current_user),
) -> CaseSummary:
    case = _get_case_or_404(session, case_id)
    try:
        case.status = CaseStatus(payload.status)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Статус должен быть 'pending' или 'confirmed'",
        ) from exc
    session.commit()
    return case_to_summary(case)


@router.get("/{case_id}/image")
def get_case_image(
    case_id: str,
    session: Session = Depends(db_session),
    settings: Settings = Depends(settings_dep),
    _user: User = Depends(current_user),
) -> FileResponse:
    case = _get_case_or_404(session, case_id)
    path = settings.resolve_path(case.source_image_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Изображение недоступно")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{case_id}/mask")
def get_case_mask(
    case_id: str,
    session: Session = Depends(db_session),
    settings: Settings = Depends(settings_dep),
    _user: User = Depends(current_user),
) -> FileResponse:
    case = _get_case_or_404(session, case_id)
    path = settings.resolve_path(case.mask_image_path)
    if not path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Маска сегментации недоступна")
    return FileResponse(path, media_type="image/png")
