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


class RegionOut(BaseModel):
    rank: int
    x: int
    y: int
    score: float


class IhcMarkers(BaseModel):
    ki67: Optional[float] = Field(default=None, ge=0, le=100)
    er_status: Optional[str] = None
    pr_status: Optional[str] = None
    her2_status: Optional[str] = None


class AnalysisResult(BaseModel):
    case_id: str
    verdict_label: str
    is_malignant: bool
    confidence: float
    malignant_probability: float
    benign_probability: float
    top_regions: list[RegionOut]
    analysis_mode: str


class CaseSummary(BaseModel):
    id: str
    created_at: dt.datetime
    tissue_type: str
    verdict_label: str
    is_malignant: bool
    confidence: float
    status: str


class CaseDetail(CaseSummary):
    source_filename: str
    analysis_mode: str
    malignant_probability: float
    benign_probability: float
    top_regions: list[RegionOut]
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
