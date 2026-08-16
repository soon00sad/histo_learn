"""Application-wide logging setup. Import get_logger(__name__) instead of print()."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from src.utils.config import get_settings

_CONFIGURED = False


def configure_logging() -> None:
    """Idempotently attach console + rotating-file handlers to the root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    settings.ensure_runtime_dirs()
    log_cfg = settings.logging
    log_path = settings.resolve_path(log_cfg.file)

    root = logging.getLogger()
    root.setLevel(log_cfg.level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=log_cfg.max_bytes,
        backupCount=log_cfg.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring handlers on first use."""
    configure_logging()
    return logging.getLogger(name)
