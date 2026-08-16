"""Job status polling for asynchronous WSI analysis (see routers/analysis.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.db import Job, User
from src.api.deps import current_user, db_session
from src.api.schemas import JobStatusOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusOut)
def get_job(
    job_id: str,
    session: Session = Depends(db_session),
    _user: User = Depends(current_user),
) -> JobStatusOut:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Задача не найдена")

    return JobStatusOut(
        id=job.id,
        status=job.status.value,
        stage=job.stage,
        progress=job.progress,
        message=job.message,
        case_id=job.case_id,
        error=job.error,
    )
