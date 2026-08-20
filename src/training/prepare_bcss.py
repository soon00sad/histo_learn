"""Prepare BCSS data for HistoVision segmentation training/dataset.py.

Produces the on-disk layout `dataset.py` expects:

    <out>/images/<id>.png    RGB region
    <out>/masks/<id>.png     single-channel raw BCSS GT codes (uint8)
    <out>/splits/train.txt   one <id> per line
    <out>/splits/val.txt     one <id> per line

Three subcommands:

  synthetic   Generate a tiny procedural dataset (no network, no real BCSS
              data) — for smoke-testing the training pipeline's *code*
              (shapes, loss, checkpointing, mIoU) on a machine with no GPU
              and no BCSS download, exactly like tests/conftest.py's
              synthetic_he_patch fixture avoids needing a real model
              checkpoint. This is what CI / this dev machine should use.

  from-source Reorganize an already-downloaded real BCSS export (expects
              <source-dir>/images/*.png + <source-dir>/masks/*.png with
              matching filenames, which is the layout BCSS's own official
              download scripts produce) into the split layout above.

  download    Best-effort full download of the official BCSS Google Drive
              folder via `gdown`, then reorganize like from-source. This
              pulls the WHOLE dataset (multi-GB) — intended to run on a
              GPU/cloud machine that will also do the real training, not
              this project's low-RAM/no-GPU dev machine. Not exercised in
              CI; treat network failures as expected and retry manually.

Usage:
    python -m src.training.prepare_bcss synthetic --out data/bcss --count 8
    python -m src.training.prepare_bcss from-source --source-dir /path/to/bcss --out data/bcss
    python -m src.training.prepare_bcss download --out data/bcss --limit 40
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.utils.bcss_classes import load_bcss_classes
from src.utils.logging import get_logger

logger = get_logger(__name__)

BCSS_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss"
)


def _write_split(out_dir: Path, ids: list[str], val_fraction: float, seed: int = 0) -> None:
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction)) if len(shuffled) > 1 else 0
    val_ids, train_ids = shuffled[:n_val], shuffled[n_val:]
    if not train_ids:  # degenerate tiny-sample case: keep at least one train sample
        train_ids, val_ids = shuffled, []

    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text("\n".join(sorted(train_ids)) + "\n", encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(sorted(val_ids)) + "\n", encoding="utf-8")
    logger.info("Wrote split: %d train / %d val samples", len(train_ids), len(val_ids))


def cmd_synthetic(out: Path, count: int, region_size: int, val_fraction: float, seed: int) -> None:
    """Procedural regions with a stroma background and blobs of several raw
    BCSS codes, so remap_mask + class-imbalance weighting have more than one
    class to exercise. Not real tissue — proves the pipeline runs, not that
    it's accurate (see docs/MODEL.md smoke-test disclaimer).
    """
    taxonomy = load_bcss_classes()
    valid_raw_codes = sorted(taxonomy.raw_code_to_model_index().keys())
    # A handful of representative raw codes, always including some that map
    # to distinct HistoVision classes (tumor=1, stroma=2, lymphocytic=3,
    # necrosis=4, fat=9, blood_vessel=18) plus the outside_roi ignore ring.
    blob_codes = [c for c in (1, 2, 3, 4, 9, 18) if c in valid_raw_codes]

    images_dir, masks_dir = out / "images", out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    ids: list[str] = []
    for i in range(count):
        sample_id = f"synthetic-{i:03d}"
        ids.append(sample_id)

        # RGB: pale pink/purple H&E-like background with colored blobs.
        rgb = np.full((region_size, region_size, 3), (235, 210, 220), dtype=np.uint8)
        mask = np.full((region_size, region_size), 2, dtype=np.uint8)  # default: stroma

        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        mask_img = Image.fromarray(mask)
        mask_draw = ImageDraw.Draw(mask_img)

        for code in blob_codes:
            n_blobs = rng.randint(1, 3)
            color = tuple(int(c) for c in np_rng.integers(60, 220, size=3))
            for _ in range(n_blobs):
                r = rng.randint(region_size // 16, region_size // 6)
                cx = rng.randint(r, region_size - r)
                cy = rng.randint(r, region_size - r)
                bbox = [cx - r, cy - r, cx + r, cy + r]
                draw.ellipse(bbox, fill=color)
                mask_draw.ellipse(bbox, fill=code)

        # outside_roi ignore ring around the border, like real BCSS ROI framing.
        border = max(2, region_size // 32)
        mask_arr = np.array(mask_img)
        mask_arr[:border, :] = 0
        mask_arr[-border:, :] = 0
        mask_arr[:, :border] = 0
        mask_arr[:, -border:] = 0

        image.save(images_dir / f"{sample_id}.png")
        Image.fromarray(mask_arr).save(masks_dir / f"{sample_id}.png")

    _write_split(out, ids, val_fraction, seed)
    logger.info("Wrote %d synthetic BCSS-shaped samples to %s", count, out)


# The official BCSS Google Drive folder uses "rgbs_colorNormalized" as the
# images subfolder name (not "images") — "images" is kept as an alias for
# hand-organized/renamed local exports.
_IMAGE_DIR_NAMES = ("images", "rgbs_colorNormalized")


def _find_images_dir(source_dir: Path) -> Path:
    for name in _IMAGE_DIR_NAMES:
        candidate = source_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"No images directory found under {source_dir} (tried: {', '.join(_IMAGE_DIR_NAMES)})"
    )


def cmd_from_source(source_dir: Path, out: Path, val_fraction: float, seed: int) -> None:
    src_images = _find_images_dir(source_dir)
    src_masks = source_dir / "masks"
    if not src_masks.is_dir():
        raise FileNotFoundError(f"Expected {source_dir}/masks/ — not found.")

    images_dir, masks_dir = out / "images", out / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    for image_path in sorted(src_images.glob("*.png")):
        mask_path = src_masks / image_path.name
        if not mask_path.exists():
            logger.warning("Skipping %s: no matching mask", image_path.name)
            continue
        sample_id = image_path.stem
        shutil.copy2(image_path, images_dir / f"{sample_id}.png")
        shutil.copy2(mask_path, masks_dir / f"{sample_id}.png")
        ids.append(sample_id)

    if not ids:
        raise ValueError(f"No matching image/mask pairs found under {source_dir}")

    _write_split(out, ids, val_fraction, seed)
    logger.info("Prepared %d real BCSS samples from %s into %s", len(ids), source_dir, out)


def pair_masks_and_images(all_files, limit: int | None = None) -> dict[str, tuple[object, object]]:
    """Pair BCSS Drive folder entries by filename stem across masks/ and
    rgbs_colorNormalized/ (the two subfolders that matter for training).

    `all_files` is whatever `gdown.download_folder(..., skip_download=True)`
    returns: objects with `.path` (relpath within the folder, e.g.
    "masks/TCGA-....png") and `.local_path`. Kept as a standalone function
    (rather than inline in cmd_download) so this pairing logic — the actual
    bug that broke the first live run, where naively slicing the flat file
    list grabbed only masks with zero matching images — is unit-testable
    without a real Google Drive call.

    Returns {sample_id: (image_entry, mask_entry)}, capped to `limit` pairs.
    """
    by_relpath = {Path(f.path): f for f in all_files}
    masks = {p.stem: f for p, f in by_relpath.items() if p.parts[0] == "masks" and p.suffix == ".png"}
    images = {p.stem: f for p, f in by_relpath.items() if p.parts[0] == "rgbs_colorNormalized" and p.suffix == ".png"}
    common_ids = sorted(set(masks) & set(images))
    if limit:
        common_ids = common_ids[:limit]
    return {sample_id: (images[sample_id], masks[sample_id]) for sample_id in common_ids}


def cmd_download(out: Path, limit: int | None, val_fraction: float, seed: int) -> None:
    """Best-effort: not exercised on this dev machine (see module docstring)."""
    import gdown

    raw_dir = out / "_gdrive_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.warning(
        "Downloading from Google Drive folder %s — this is the FULL BCSS dataset "
        "(multi-GB). Run this on the machine that will do the real training, not "
        "a constrained dev box.",
        BCSS_DRIVE_FOLDER_URL,
    )
    all_files = gdown.download_folder(url=BCSS_DRIVE_FOLDER_URL, output=str(raw_dir), skip_download=True)

    pairs = pair_masks_and_images(all_files, limit)
    if not pairs:
        raise ValueError(
            "No matching image/mask pairs found in the BCSS Drive folder listing "
            "(expected masks/ and rgbs_colorNormalized/ subfolders)."
        )
    logger.info("Downloading %d matched image/mask pairs...", len(pairs))

    for image_entry, mask_entry in pairs.values():
        for f in (image_entry, mask_entry):
            # f.local_path is already the full destination path (gdown's
            # download_folder joins `output` in when building it) — do not
            # join raw_dir in again. skip_download=True also means gdown
            # never created the directory structure, so mkdir first.
            dest = Path(f.local_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=f.id, output=str(dest), quiet=False)

    cmd_from_source(raw_dir, out, val_fraction, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_synth = sub.add_parser("synthetic", help="Generate a tiny procedural dataset (no network).")
    p_synth.add_argument("--out", type=Path, default=Path("data/bcss"))
    p_synth.add_argument("--count", type=int, default=8)
    p_synth.add_argument("--region-size", type=int, default=320)
    p_synth.add_argument("--val-fraction", type=float, default=0.25)
    p_synth.add_argument("--seed", type=int, default=0)

    p_src = sub.add_parser("from-source", help="Reorganize an already-downloaded BCSS export.")
    p_src.add_argument("--source-dir", type=Path, required=True)
    p_src.add_argument("--out", type=Path, default=Path("data/bcss"))
    p_src.add_argument("--val-fraction", type=float, default=0.15)
    p_src.add_argument("--seed", type=int, default=0)

    p_dl = sub.add_parser("download", help="Full BCSS download via gdown (run on the training machine).")
    p_dl.add_argument("--out", type=Path, default=Path("data/bcss"))
    p_dl.add_argument("--limit", type=int, default=None, help="Cap number of files fetched (omit for the full dataset).")
    p_dl.add_argument("--val-fraction", type=float, default=0.15)
    p_dl.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "synthetic":
        cmd_synthetic(args.out, args.count, args.region_size, args.val_fraction, args.seed)
    elif args.command == "from-source":
        cmd_from_source(args.source_dir, args.out, args.val_fraction, args.seed)
    elif args.command == "download":
        cmd_download(args.out, args.limit, args.val_fraction, args.seed)


if __name__ == "__main__":
    main()
