"""SQLAlchemy engine/session setup and ORM models for HistoVision.

SQLite is a deliberate choice for the on-premise single-instance MVP: no
extra service to operate, file-based so it lives on the same volume as
uploaded slides, and swappable for Postgres later without touching callers
(they only see Session objects). See docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import datetime as dt
import enum
import uuid
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from src.utils.config import Settings, get_settings


class Base(DeclarativeBase):
    pass


class CaseStatus(str, enum.Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    doctor_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)

    tissue_type: Mapped[str] = mapped_column(String(255), default="Молочная железа, биопсия")
    source_filename: Mapped[str] = mapped_column(String(512))
    analysis_mode: Mapped[str] = mapped_column(String(32))  # "patch" | "wsi"

    verdict_label: Mapped[str] = mapped_column(String(64))
    is_malignant: Mapped[bool] = mapped_column()
    tumor_area_fraction: Mapped[float] = mapped_column(Float)
    # JSON {name_en: fraction}, only classes actually present in the mask —
    # see src/inference/verdict.py / src/api/case_service.py. Segmentation
    # replaces the old binary malignant_probability/benign_probability pair.
    class_areas_json: Mapped[str] = mapped_column(Text, default="{}")

    source_image_path: Mapped[str] = mapped_column(String(1024))
    # Lossless indexed PNG (exact per-pixel class IDs), not the old JPEG
    # heatmap — see case_service.save_mask_png.
    mask_image_path: Mapped[str] = mapped_column(String(1024))
    report_pdf_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)

    # "model" (real segmentation output) vs "bcss_ground_truth" (pathologist-
    # annotated reference mask, used for exhibition demo cases seeded by
    # scripts/seed_demo_cases.py while the model is still undertrained — see
    # docs/MODEL.md). Never hardcoded per-case in application code; the
    # frontend shows a reference-data badge whenever this isn't "model".
    mask_source: Mapped[str] = mapped_column(String(32), default="model")

    ki67: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    er_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    pr_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    her2_status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    status: Mapped[CaseStatus] = mapped_column(Enum(CaseStatus), default=CaseStatus.PENDING)


class CaseReview(Base):
    """A doctor's agree/disagree decision on a case's system-generated
    verdict. Written once per review action (a case can be reviewed more
    than once if reopened), kept even after Case.status changes again, so
    the history isn't lost. Disagreements (agreed=False) are the intended
    seed for future retraining data — this table only *collects* them;
    nothing here triggers retraining automatically.
    """
    __tablename__ = "case_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id"), index=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    agreed: Mapped[bool] = mapped_column()
    # System's verdict/tumor fraction at review time — kept alongside the
    # doctor's correction so a later retraining pass doesn't need to guess
    # what the model originally predicted from the (possibly since-changed) Case row.
    system_verdict_label: Mapped[str] = mapped_column(String(64))
    system_tumor_area_fraction: Mapped[float] = mapped_column(Float)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    corrected_verdict_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow
    )

    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.QUEUED)
    stage: Mapped[str] = mapped_column(String(64), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(String(512), default="")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    case_id: Mapped[Optional[str]] = mapped_column(ForeignKey("cases.id"), nullable=True)


_engine = None
_SessionLocal: Optional[sessionmaker] = None


def init_db(settings: Optional[Settings] = None) -> None:
    """Create the SQLite engine/schema and seed the default clinician user.

    Safe to call multiple times (e.g. in tests) — table creation and user
    seeding are both idempotent.
    """
    global _engine, _SessionLocal
    settings = settings or get_settings()
    settings.ensure_runtime_dirs()
    db_path = settings.resolve_path(settings.paths.db_path)
    _engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(_engine)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    _seed_default_user(settings)


def _seed_default_user(settings: Settings) -> None:
    from src.api.security import hash_password

    with get_session() as session:
        seed = settings.auth.seed_user
        if session.query(User).filter_by(email=seed.email).first():
            return
        session.add(
            User(
                email=seed.email,
                full_name=seed.full_name,
                role=seed.role,
                password_hash=hash_password(seed.password),
            )
        )
        session.commit()


def get_session() -> Session:
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized; call init_db() first.")
    return _SessionLocal()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context-managed session for code outside the request/response cycle
    (e.g. FastAPI BackgroundTasks running the WSI pipeline)."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def generate_case_id(session: Session) -> str:
    """Sequential, human-friendly case id in the HV-YYYY-NNNNN format used by the design."""
    year = dt.datetime.utcnow().year
    count = session.query(Case).count() + 1
    return f"HV-{year}-{4800 + count:05d}"


def generate_job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"
