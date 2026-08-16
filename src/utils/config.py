"""Centralized configuration loading for HistoVision.

All tunable values (paths, model parameters, thresholds) live in
config/config.yaml. Two secrets may be overridden via environment variables
so they never need to be committed: HISTOVISION_JWT_SECRET and
HISTOVISION_SEED_USER_PASSWORD.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class AppConfig(BaseModel):
    name: str
    version: str
    disclaimer: str


class ServerConfig(BaseModel):
    host: str
    port: int
    cors_origins: list[str]


class SeedUserConfig(BaseModel):
    email: str
    full_name: str
    role: str
    password: str


class AuthConfig(BaseModel):
    jwt_algorithm: str
    access_token_expire_minutes: int
    jwt_secret: str
    seed_user: SeedUserConfig


class ModelConfig(BaseModel):
    architecture: str
    num_classes: int
    class_names: list[str]
    malignant_class_index: int
    weights_path: str
    device: str
    input_size: int
    normalize_mean: list[float]
    normalize_std: list[float]
    gradcam_target_layer: str
    metrics_summary: str


class XaiConfig(BaseModel):
    top_k_regions: int
    region_threshold_ratio: float
    min_region_pixels: int


class MacenkoReferenceConfig(BaseModel):
    stain_vectors: list[list[float]]
    max_concentrations: list[float]


class PreprocessingConfig(BaseModel):
    tissue_saturation_threshold: Optional[float] = None
    min_tissue_fraction: float
    stain_normalization_enabled: bool
    macenko_reference: MacenkoReferenceConfig


class WsiConfig(BaseModel):
    tile_size: int
    tile_stride: int
    tissue_mask_level_downsample: int
    batch_size: int
    max_processing_minutes: int


class PathsConfig(BaseModel):
    data_dir: str
    uploads_dir: str
    heatmaps_dir: str
    reports_dir: str
    db_path: str
    log_dir: str


class LoggingConfig(BaseModel):
    level: str
    file: str
    max_bytes: int
    backup_count: int


class Settings(BaseModel):
    app: AppConfig
    server: ServerConfig
    auth: AuthConfig
    model: ModelConfig
    xai: XaiConfig
    preprocessing: PreprocessingConfig
    wsi: WsiConfig
    paths: PathsConfig
    logging: LoggingConfig

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config-relative path against the project root."""
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def ensure_runtime_dirs(self) -> None:
        """Create the data directories the app writes to, if missing."""
        for relative in (
            self.paths.uploads_dir,
            self.paths.heatmaps_dir,
            self.paths.reports_dir,
            self.resolve_path(self.paths.db_path).parent,
            self.resolve_path(self.paths.log_dir),
        ):
            self.resolve_path(str(relative)).mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. HistoVision requires config/config.yaml."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(raw: dict) -> dict:
    jwt_secret = os.environ.get("HISTOVISION_JWT_SECRET")
    if jwt_secret:
        raw["auth"]["jwt_secret"] = jwt_secret
    seed_password = os.environ.get("HISTOVISION_SEED_USER_PASSWORD")
    if seed_password:
        raw["auth"]["seed_user"]["password"] = seed_password
    return raw


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings for the process lifetime."""
    raw = _apply_env_overrides(_load_yaml(CONFIG_PATH))
    return Settings(**raw)
