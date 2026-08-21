"""Pydantic request/response models for the HistoVision API."""
from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    email: str
    password: str


class UserOut(BaseModel):
    email: str
    full_name: str
    role: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class ClassAreaOut(BaseModel):
    """One tissue class' share of the analyzed area — see
    config/bcss_classes.yaml for the taxonomy and src/inference/verdict.py
    for how the verdict is derived from these."""

    name_en: str
    name_ru: str
    color: str
    fraction: float


class IhcMarkers(BaseModel):
    ki67: Optional[float] = Field(default=None, ge=0, le=100)
    er_status: Optional[str] = None
    pr_status: Optional[str] = None
    her2_status: Optional[str] = None


class AnalysisResult(BaseModel):
    case_id: str
    verdict_label: str
    is_malignant: bool
    tumor_area_fraction: float
    class_areas: list[ClassAreaOut]
    analysis_mode: str


class CaseSummary(BaseModel):
    id: str
    created_at: dt.datetime
    tissue_type: str
    verdict_label: str
    is_malignant: bool
    tumor_area_fraction: float
    status: str


class CaseDetail(CaseSummary):
    source_filename: str
    analysis_mode: str
    class_areas: list[ClassAreaOut]
    ki67: Optional[float]
    er_status: Optional[str]
    pr_status: Optional[str]
    her2_status: Optional[str]
    report_available: bool


class CaseStatusUpdate(BaseModel):
    status: str  # "pending" | "confirmed"


class JobStatusOut(BaseModel):
    id: str
    status: str
    stage: str
    progress: float
    message: str
    case_id: Optional[str] = None
    error: Optional[str] = None


class JobAccepted(BaseModel):
    job_id: str


class ReportUrl(BaseModel):
    report_url: str
