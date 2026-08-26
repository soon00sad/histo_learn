"""Seed exhibition demo cases from BCSS ground-truth (pathologist) masks,
standing in for real model inference while the segmentation model is still
undertrained (see docs/MODEL.md) — everything downstream of the mask
(area-fraction aggregation, verdict derivation, mask-on-tissue overlay,
PDF report, doctor agree/disagree review) is the real, unchanged product
code path. Case.mask_source is set to "bcss_ground_truth" for these cases
so the frontend and PDF report both show a "reference data" notice —
never silently indistinguishable from a real model result.

Which BCSS regions to use and their display metadata come from
config/demo_cases.yaml, not this script — see that file for how the three
regions were chosen.

Usage (run once per deployment; needs internet the first time, to fetch
the BCSS regions themselves — safe to re-run afterwards without network,
since already-downloaded regions are skipped, and the API/frontend need
no internet access at demo time):

    python scripts/seed_demo_cases.py
    # or, against the running Docker stack:
    docker compose exec api python scripts/seed_demo_cases.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.case_service import build_report_data, class_areas_to_json, save_image, save_mask_png  # noqa: E402
from src.api.db import Case, User, init_db, session_scope  # noqa: E402
from src.inference.verdict import derive_verdict  # noqa: E402
from src.report.pdf_report import generate_pdf_report  # noqa: E402
from src.training import prepare_bcss  # noqa: E402
from src.training.dataset import IGNORE_INDEX, remap_mask  # noqa: E402
from src.utils.bcss_classes import load_bcss_classes  # noqa: E402
from src.utils.config import get_settings  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

DEMO_CONFIG_PATH = PROJECT_ROOT / "config" / "demo_cases.yaml"
DEMO_RAW_DATA_DIR = Path("data") / "bcss_demo"
# BCSS regions run up to ~130 megapixels — downsized so the saved source
# JPEG/mask PNG, the frontend viewer, and the PDF stay a sane size. NEAREST
# resampling for the mask (not LANCZOS/bilinear) is required to keep exact
# per-pixel class ids intact — a blurred mask would invent nonexistent
# in-between classes at boundaries.
MAX_DIMENSION = 1600


def _load_demo_entries() -> list[dict]:
    with open(DEMO_CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["cases"]


def _find_downloaded_file(directory: Path, row_id: str) -> Path:
    matches = sorted(directory.glob(f"{row_id}_*.png"))
    if not matches:
        raise FileNotFoundError(f"No downloaded file for BCSS region '{row_id}' under {directory}")
    return matches[0]


def _resize_together(image: Image.Image, mask: np.ndarray, max_dim: int) -> tuple[Image.Image, np.ndarray]:
    width, height = image.size
    scale = min(1.0, max_dim / max(width, height))
    if scale >= 1.0:
        return image, mask
    new_size = (round(width * scale), round(height * scale))
    resized_image = image.resize(new_size, Image.LANCZOS)
    resized_mask = np.array(Image.fromarray(mask).resize(new_size, Image.NEAREST))
    return resized_image, resized_mask


def seed_demo_cases() -> None:
    entries = _load_demo_entries()
    row_ids = {entry["bcss_row_id"] for entry in entries}

    logger.info("Fetching %d BCSS demo region(s) from the official source (network needed once)...", len(row_ids))
    prepare_bcss.cmd_download_official(DEMO_RAW_DATA_DIR, limit=None, val_fraction=0.0, seed=0, sample_ids=row_ids)

    taxonomy = load_bcss_classes()
    settings = get_settings()
    init_db(settings)

    with session_scope() as session:
        seed_user = session.query(User).filter_by(email=settings.auth.seed_user.email).first()
        if seed_user is None:
            raise RuntimeError("Seed clinician user not found — start the API once first so init_db seeds it.")

        for entry in entries:
            case_id, row_id = entry["case_id"], entry["bcss_row_id"]

            image_path = _find_downloaded_file(DEMO_RAW_DATA_DIR / "images", row_id)
            mask_path = _find_downloaded_file(DEMO_RAW_DATA_DIR / "masks", row_id)

            image = Image.open(image_path).convert("RGB")
            raw_mask = np.array(Image.open(mask_path).convert("L"))
            remapped = remap_mask(raw_mask, taxonomy)  # raw BCSS codes -> 0-based model indices; outside_roi -> IGNORE_INDEX
            image, remapped = _resize_together(image, remapped, MAX_DIMENSION)

            valid = remapped[remapped != IGNORE_INDEX]
            total = int(valid.size)
            fractions = {
                c.name_en: int((valid == c.model_index).sum()) / total
                for c in taxonomy.classes
                if int((valid == c.model_index).sum())
            }
            verdict = derive_verdict(fractions, settings, taxonomy)

            source_rel = f"{settings.paths.uploads_dir}/{case_id}.jpg"
            mask_rel = f"{settings.paths.heatmaps_dir}/{case_id}.png"
            save_image(image, settings.resolve_path(source_rel))
            # IGNORE_INDEX (255) == wsi_segmenter.UNCOVERED (255) by design —
            # save_mask_png renders it as fully transparent, same as an
            # un-analyzed WSI background pixel would be.
            save_mask_png(remapped, settings.resolve_path(mask_rel), taxonomy)

            existing = session.get(Case, case_id)
            if existing is not None:
                session.delete(existing)
                session.flush()

            case = Case(
                id=case_id,
                doctor_id=seed_user.id,
                tissue_type=entry.get("tissue_type", "Молочная железа, биопсия"),
                source_filename=f"{row_id}.svs",
                analysis_mode="wsi",
                verdict_label=verdict.verdict_label,
                is_malignant=verdict.is_malignant,
                tumor_area_fraction=verdict.tumor_area_fraction,
                class_areas_json=class_areas_to_json(fractions),
                source_image_path=source_rel,
                mask_image_path=mask_rel,
                mask_source="bcss_ground_truth",
            )
            session.add(case)
            session.flush()

            # Pre-generate the PDF so "Открыть PDF-отчёт" works instantly at
            # the booth instead of generating on first click.
            report_rel = f"{settings.paths.reports_dir}/{case.id}.pdf"
            report_data = build_report_data(case, settings, doctor_name=seed_user.full_name)
            generate_pdf_report(report_data, settings.resolve_path(report_rel))
            case.report_pdf_path = report_rel
            session.commit()

            logger.info(
                "Seeded demo case %s (%s, tumor_area=%.3f, %d classes) from BCSS region %s",
                case_id, verdict.verdict_label, verdict.tumor_area_fraction, len(fractions), row_id,
            )

    logger.info("Done — %d demo case(s) seeded.", len(entries))


if __name__ == "__main__":
    seed_demo_cases()
