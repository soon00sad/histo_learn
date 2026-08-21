"""HistoVision FastAPI application entrypoint.

Run with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.db import init_db
from src.api.ml_runtime import get_segmenter
from src.api.routers import analysis, auth, cases, jobs, reports
from src.utils.config import get_settings
from src.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    description=(
        "Система поддержки принятия диагностических решений для "
        "патологоанатома. Не заменяет врача — окончательное заключение "
        "формулирует врач-патологоанатом."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(analysis.router, prefix=API_PREFIX)
app.include_router(jobs.router, prefix=API_PREFIX)
app.include_router(cases.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)


@app.on_event("startup")
def on_startup() -> None:
    init_db(settings)
    logger.info("Database ready")
    try:
        get_segmenter()
        logger.info("Segmentation model pre-loaded")
    except FileNotFoundError as exc:
        logger.warning(
            "Segmentation model weights unavailable at startup (%s). Analysis endpoints "
            "will fail until a trained checkpoint is placed at models/segmentation.pth "
            "(see notebooks/train_segmentation_colab.ipynb).", exc,
        )


@app.get(f"{API_PREFIX}/health", tags=["health"])
def health() -> dict:
    return {"status": "ok", "app": settings.app.name, "version": settings.app.version}
