"""Smoke tests for the src/training/ segmentation pipeline.

Mirrors the project's existing test philosophy (see tests/conftest.py's
stub_load_model / synthetic_he_patch): no real BCSS download, no real
trained checkpoint, no network access — synthetic data + an untrained
(encoder_weights=None) model, just like the classifier tests use an
untrained timm model instead of models/model.pth. These tests prove the
training *code* is correct (shapes, loss, checkpointing, mIoU computation),
not that the model is accurate — see docs/MODEL.md for that distinction.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.training import prepare_bcss
from src.training.dataset import BcssDataset, IGNORE_INDEX, list_samples, remap_mask
from src.training.losses import WeightedCeDiceLoss, compute_class_weights
from src.training.metrics import ConfusionMatrix, count_class_pixels
from src.training import train as train_module
from src.training.train import build_model, evaluate, train_one_epoch
from src.utils.bcss_classes import load_bcss_classes

import numpy as np


def test_stain_normalize_runs_on_the_crop_not_the_full_region(tmp_path, monkeypatch):
    """Regression test for a real ~10-hour "hung" Colab run: BcssDataset used
    to run Macenko normalization on the full BCSS region (up to ~11800px
    per side) before cropping to crop_size, instead of after. Macenko's
    cost scales with pixel count (~10s for 4500x4500 vs ~0.04s for 256x256
    measured on this machine), so with that ordering every __getitem__ call
    on every sample of every epoch paid a many-seconds-per-image tax with
    zero progress logging — indistinguishable from a hang. This pins the
    fix: normalize() must see an image no larger than crop_size."""
    from src.training import dataset as dataset_module

    seen_sizes = []
    real_normalize = dataset_module.macenko_normalize

    def spy_normalize(image, reference):
        seen_sizes.append(image.size)  # PIL .size is (width, height)
        return real_normalize(image, reference)

    monkeypatch.setattr(dataset_module, "macenko_normalize", spy_normalize)

    out = tmp_path / "bcss_smoke"
    prepare_bcss.cmd_synthetic(out, count=2, region_size=800, val_fraction=0.5, seed=0)
    taxonomy = load_bcss_classes()
    samples = list_samples(out, "train") + list_samples(out, "val")

    crop_size = 96
    ds = BcssDataset(samples, taxonomy, crop_size=crop_size, augment=False, stain_normalize=True)
    for i in range(len(ds)):
        ds[i]

    assert seen_sizes, "macenko_normalize was never called"
    for width, height in seen_sizes:
        assert width <= crop_size and height <= crop_size, (
            f"macenko_normalize saw a {width}x{height} image — should be cropped to "
            f"{crop_size}x{crop_size} (or smaller, at the sample's own resolution) first"
        )


def test_remap_mask_maps_merged_raw_codes_to_same_class():
    taxonomy = load_bcss_classes()
    raw = np.array([[1, 19], [2, 0]], dtype=np.uint8)  # tumor, angioinvasion, stroma, outside_roi
    remapped = remap_mask(raw, taxonomy)
    tumor_index = next(c.model_index for c in taxonomy.classes if c.name_en == "tumor")
    stroma_index = next(c.model_index for c in taxonomy.classes if c.name_en == "stroma")
    assert remapped[0, 0] == tumor_index
    assert remapped[0, 1] == tumor_index  # angioinvasion merged into tumor
    assert remapped[1, 0] == stroma_index
    assert remapped[1, 1] == IGNORE_INDEX  # outside_roi -> ignore


def test_confusion_matrix_perfect_prediction_gives_iou_one():
    cm = ConfusionMatrix(num_classes=3)
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    cm.update(targets.clone(), targets)
    iou = cm.per_class_iou()
    assert torch.allclose(iou, torch.ones(3))
    assert cm.mean_iou() == 1.0


def test_confusion_matrix_ignores_ignore_index():
    cm = ConfusionMatrix(num_classes=2, ignore_index=IGNORE_INDEX)
    preds = torch.tensor([0, 1, 1])
    targets = torch.tensor([0, 0, IGNORE_INDEX])  # last pixel should be excluded entirely
    cm.update(preds, targets)
    assert cm.matrix.sum().item() == 2  # only the 2 non-ignored pixels counted


def test_confusion_matrix_nan_for_absent_class():
    cm = ConfusionMatrix(num_classes=3)
    cm.update(torch.tensor([0, 0]), torch.tensor([0, 0]))  # class 2 never appears
    iou = cm.per_class_iou()
    assert iou[0].item() == 1.0
    assert torch.isnan(iou[2])
    assert cm.mean_iou() == 1.0  # NaN class excluded from the mean, not treated as 0


def test_compute_class_weights_favors_rare_classes():
    counts = torch.tensor([1000, 10])
    weights = compute_class_weights(counts)
    assert weights[1] > weights[0]  # rarer class gets a bigger weight
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-4)


def test_compute_class_weights_caps_extreme_imbalance():
    """BCSS-scale imbalance (tumor/stroma vs. nerve/blood_vessel) can be
    orders of magnitude — uncapped inverse frequency would let a
    near-absent class dominate the loss. The heaviest weight must stay
    within max_imbalance_ratio of the lightest, not grow unbounded."""
    counts = torch.tensor([1_000_000, 5])  # 200,000x raw imbalance
    weights = compute_class_weights(counts, max_imbalance_ratio=50.0)
    assert weights.max().item() / weights.min().item() == pytest.approx(50.0, rel=1e-3)
    assert weights[1] > weights[0]  # still favors the rare class, just bounded


def _fake_drive_entry(path: str) -> SimpleNamespace:
    """Stands in for gdown's GoogleDriveFileToDownload (.path/.local_path/.id)."""
    return SimpleNamespace(path=path, local_path=f"/tmp/_gdrive_raw/{path}", id=f"id-{path}")


def test_pair_masks_and_images_ignores_logs_and_meta_and_matches_by_stem():
    # Mirrors the real BCSS Drive folder layout: logs/ and meta/ have no
    # counterpart to pair with; only masks/ and rgbs_colorNormalized/ matter.
    files = [
        _fake_drive_entry("logs/2021-05-14.log"),
        _fake_drive_entry("meta/gtruth_codes.tsv"),
        _fake_drive_entry("masks/TCGA-A1-1.png"),
        _fake_drive_entry("masks/TCGA-A1-2.png"),
        _fake_drive_entry("rgbs_colorNormalized/TCGA-A1-1.png"),
        _fake_drive_entry("rgbs_colorNormalized/TCGA-A1-2.png"),
    ]
    pairs = prepare_bcss.pair_masks_and_images(files)
    assert set(pairs) == {"TCGA-A1-1", "TCGA-A1-2"}
    image_entry, mask_entry = pairs["TCGA-A1-1"]
    assert image_entry.path == "rgbs_colorNormalized/TCGA-A1-1.png"
    assert mask_entry.path == "masks/TCGA-A1-1.png"


def test_pair_masks_and_images_naive_slice_would_have_grabbed_zero_images():
    """Regression test for the bug from the first live Colab run: masks/
    sorts before rgbs_colorNormalized/ in the real folder listing, so
    slicing the first N raw files (the old implementation) grabbed only
    masks. pair_masks_and_images must apply --limit to matched pairs, not
    to the flat file list."""
    files = (
        [_fake_drive_entry("logs/x.log")]
        + [_fake_drive_entry(f"masks/TCGA-{i}.png") for i in range(5)]
        + [_fake_drive_entry(f"rgbs_colorNormalized/TCGA-{i}.png") for i in range(5)]
    )
    naive_slice = files[:3]  # what the old buggy code effectively did
    assert all(Path(f.path).parts[0] != "rgbs_colorNormalized" for f in naive_slice), (
        "sanity check: confirms the bug scenario this test guards against"
    )

    pairs = prepare_bcss.pair_masks_and_images(files, limit=3)
    assert len(pairs) == 3
    for image_entry, mask_entry in pairs.values():
        assert Path(image_entry.path).parts[0] == "rgbs_colorNormalized"
        assert Path(mask_entry.path).parts[0] == "masks"


def test_pair_masks_and_images_empty_when_no_overlap():
    files = [_fake_drive_entry("masks/only-in-masks.png")]
    assert prepare_bcss.pair_masks_and_images(files) == {}


def test_find_images_dir_accepts_rgbs_color_normalized_alias(tmp_path):
    (tmp_path / "rgbs_colorNormalized").mkdir()
    (tmp_path / "masks").mkdir()
    assert prepare_bcss._find_images_dir(tmp_path).name == "rgbs_colorNormalized"


def test_find_images_dir_prefers_plain_images_name(tmp_path):
    (tmp_path / "images").mkdir()
    assert prepare_bcss._find_images_dir(tmp_path).name == "images"


def test_cmd_download_skips_failed_pairs_and_keeps_successful_ones(tmp_path, monkeypatch):
    """Regression test for the live run that hit Google Drive's per-file
    download quota (FileURLRetrievalError) partway through: one failing
    pair must not abort the whole dataset — the rest should still end up
    as a usable, split dataset."""
    import gdown

    out = tmp_path / "bcss"

    def fake_download_folder(url, output, skip_download):
        assert skip_download is True
        entries = []
        for i in range(3):
            for folder in ("masks", "rgbs_colorNormalized"):
                rel = f"{folder}/sample-{i}.png"
                entries.append(SimpleNamespace(path=rel, local_path=str(Path(output) / rel), id=f"id-{rel}"))
        return entries

    def fake_download(id, output, quiet=False):
        if "sample-1" in output:
            raise RuntimeError("simulated Google Drive quota error")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"fake-png-bytes")
        return output

    monkeypatch.setattr(gdown, "download_folder", fake_download_folder)
    monkeypatch.setattr(gdown, "download", fake_download)

    prepare_bcss.cmd_download(out, limit=None, val_fraction=0.5, seed=0)

    all_ids = {s.id for s in list_samples(out, "train") + list_samples(out, "val")}
    assert all_ids == {"sample-0", "sample-2"}  # sample-1's simulated failure excluded it, not fatal


def _fake_roi_row(index: int) -> dict:
    return {
        "": f"TCGA-XX-{index:04d}-DX1",
        "xmin": "0", "ymin": "0", "xmax": "8", "ymax": "8",
        "mask_link": f"https://example.invalid/mask-{index}",
    }


def _fake_mask_response(class_index: int = 2):
    mask = np.full((8, 8), class_index, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return SimpleNamespace(content=buf.getvalue(), raise_for_status=lambda: None)


def test_cmd_download_official_skips_one_failed_region_and_keeps_the_rest(tmp_path, monkeypatch):
    """Mirrors test_cmd_download_skips_failed_pairs_and_keeps_successful_ones,
    but for the Girder+Figshare download path (prepare_bcss.cmd_download_official)
    that replaces gdown/Google-Drive as the recommended full-dataset source —
    a single region's Girder tile request failing must not abort the run."""
    import girder_client

    out = tmp_path / "bcss"
    rows = [_fake_roi_row(i) for i in range(3)]
    monkeypatch.setattr(prepare_bcss, "fetch_roi_bounds", lambda: rows)

    class FakeGirderClient:
        def __init__(self, apiUrl):
            self.apiUrl = apiUrl

        def authenticate(self, apiKey):
            pass

        def get(self, path, jsonResp=True):
            if path.startswith("item?"):
                return [{"name": row[""] + "-01Z-00-DX1.svs", "_id": f"id-{row['']}"} for row in rows]
            if "id-TCGA-XX-0001" in path:
                raise RuntimeError("simulated Girder region failure")
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), color=(200, 180, 190)).save(buf, format="PNG")
            return SimpleNamespace(content=buf.getvalue())

    monkeypatch.setattr(girder_client, "GirderClient", FakeGirderClient)
    monkeypatch.setattr(prepare_bcss.requests, "get", lambda url, timeout=60: _fake_mask_response())

    prepare_bcss.cmd_download_official(out, limit=None, val_fraction=0.5, seed=0)

    all_ids = {s.id for s in list_samples(out, "train") + list_samples(out, "val")}
    assert all_ids == {"TCGA-XX-0000-DX1_xmin0_ymin0", "TCGA-XX-0002-DX1_xmin0_ymin0"}


def test_cmd_download_official_does_not_redownload_existing_pairs(tmp_path, monkeypatch):
    """Resumability: a region already fully downloaded from an earlier
    (possibly interrupted) run must be skipped, not re-fetched — critical
    for a free-tier Colab session that can drop mid-download."""
    import girder_client

    out = tmp_path / "bcss"
    rows = [_fake_roi_row(0)]
    monkeypatch.setattr(prepare_bcss, "fetch_roi_bounds", lambda: rows)

    (out / "images").mkdir(parents=True)
    (out / "masks").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(out / "images" / "TCGA-XX-0000-DX1_xmin0_ymin0.png")
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(out / "masks" / "TCGA-XX-0000-DX1_xmin0_ymin0.png")

    def must_not_be_called(*args, **kwargs):
        raise AssertionError("should not hit the network for an already-downloaded pair")

    class FakeGirderClient:
        def __init__(self, apiUrl):
            pass

        def authenticate(self, apiKey):
            pass

        def get(self, path, jsonResp=True):
            if path.startswith("item?"):
                return [{"name": rows[0][""] + "-01Z-00-DX1.svs", "_id": "id-0"}]
            return must_not_be_called()

    monkeypatch.setattr(girder_client, "GirderClient", FakeGirderClient)
    monkeypatch.setattr(prepare_bcss.requests, "get", must_not_be_called)

    prepare_bcss.cmd_download_official(out, limit=None, val_fraction=0.5, seed=0)

    all_ids = {s.id for s in list_samples(out, "train") + list_samples(out, "val")}
    assert all_ids == {"TCGA-XX-0000-DX1_xmin0_ymin0"}


def test_cmd_download_official_sample_ids_filters_to_exact_regions(tmp_path, monkeypatch):
    """scripts/seed_demo_cases.py needs to fetch a specific handful of
    regions on a fresh machine, not the full 151 — --sample-ids must
    restrict fetching to exactly the requested ids and reject unknown ones
    up front rather than silently returning fewer regions than asked."""
    import girder_client

    out = tmp_path / "bcss"
    rows = [_fake_roi_row(i) for i in range(3)]
    monkeypatch.setattr(prepare_bcss, "fetch_roi_bounds", lambda: rows)

    class FakeGirderClient:
        def __init__(self, apiUrl):
            pass

        def authenticate(self, apiKey):
            pass

        def get(self, path, jsonResp=True):
            if path.startswith("item?"):
                return [{"name": row[""] + "-01Z-00-DX1.svs", "_id": f"id-{row['']}"} for row in rows]
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), color=(200, 180, 190)).save(buf, format="PNG")
            return SimpleNamespace(content=buf.getvalue())

    monkeypatch.setattr(girder_client, "GirderClient", FakeGirderClient)
    monkeypatch.setattr(prepare_bcss.requests, "get", lambda url, timeout=60: _fake_mask_response())

    prepare_bcss.cmd_download_official(
        out, limit=None, val_fraction=0.5, seed=0, sample_ids={"TCGA-XX-0001-DX1"}
    )

    all_ids = {s.id for s in list_samples(out, "train") + list_samples(out, "val")}
    assert all_ids == {"TCGA-XX-0001-DX1_xmin0_ymin0"}

    with pytest.raises(ValueError, match="not found"):
        prepare_bcss.cmd_download_official(
            out, limit=None, val_fraction=0.5, seed=0, sample_ids={"does-not-exist"}
        )


def test_prepare_bcss_synthetic_layout(tmp_path):
    out = tmp_path / "bcss_smoke"
    prepare_bcss.cmd_synthetic(out, count=4, region_size=96, val_fraction=0.5, seed=1)

    train_samples = list_samples(out, "train")
    val_samples = list_samples(out, "val")
    assert len(train_samples) + len(val_samples) == 4
    assert len(val_samples) >= 1

    for sample in train_samples + val_samples:
        assert sample.image_path.exists()
        assert sample.mask_path.exists()


def test_training_pipeline_runs_end_to_end(tmp_path):
    """Full smoke run: generate data -> dataset -> one training step ->
    evaluate -> checkpoint. Uses encoder_weights=None (no pretrained
    download) to stay fully offline, matching stub_load_model's approach."""
    data_dir = tmp_path / "bcss_smoke"
    prepare_bcss.cmd_synthetic(data_dir, count=4, region_size=96, val_fraction=0.5, seed=2)

    taxonomy = load_bcss_classes()
    train_samples = list_samples(data_dir, "train")
    val_samples = list_samples(data_dir, "val")

    train_ds = BcssDataset(train_samples, taxonomy, crop_size=64, augment=True)
    val_ds = BcssDataset(val_samples, taxonomy, crop_size=64, augment=False)

    image, mask = train_ds[0]
    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert mask.shape == (64, 64)
    assert mask.dtype == torch.int64

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    counts = count_class_pixels(train_ds, taxonomy.num_classes)
    assert counts.shape == (taxonomy.num_classes,)
    class_weights = compute_class_weights(counts)

    device = torch.device("cpu")
    model = build_model("resnet18", taxonomy.num_classes, encoder_weights=None).to(device)
    criterion = WeightedCeDiceLoss(class_weights=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    assert loss == loss  # not NaN
    assert loss > 0

    cm = evaluate(model, val_loader, taxonomy.num_classes, device)
    miou = cm.mean_iou()
    assert 0.0 <= miou <= 1.0

    checkpoint_path = tmp_path / "checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    assert checkpoint_path.exists()

    # Loading the checkpoint back into a fresh model of the same shape must work.
    reloaded = build_model("resnet18", taxonomy.num_classes, encoder_weights=None)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))


def test_train_main_resumes_from_interrupted_run(tmp_path, monkeypatch):
    """Simulates a Colab session dropping partway through: run main() for 2
    epochs with --resume-path set, then run it again "cold" (as a fresh
    process would after a disconnect) with a higher --epochs cap. The
    second run must pick up at epoch 3, not restart training from epoch 1 —
    that's the whole point of --resume-path for a free-tier GPU that can
    disconnect at any time."""
    import sys

    data_dir = tmp_path / "bcss_smoke"
    prepare_bcss.cmd_synthetic(data_dir, count=6, region_size=96, val_fraction=0.34, seed=3)

    out = tmp_path / "checkpoint.pth"
    resume_path = tmp_path / "resume.pt"
    history_jsonl = tmp_path / "history.jsonl"

    base_argv = [
        "train.py",
        "--data-dir", str(data_dir),
        "--out", str(out),
        "--crop-size", "64",
        "--batch-size", "2",
        "--device", "cpu",
        "--resume-path", str(resume_path),
        "--history-jsonl", str(history_jsonl),
        "--min-epochs", "100",  # disable early stopping for this test
    ]

    monkeypatch.setattr(sys, "argv", base_argv + ["--epochs", "2"])
    train_module.main()

    assert resume_path.exists()
    first_state = torch.load(resume_path, map_location="cpu")
    assert first_state["epoch"] == 2
    assert len(first_state["history"]) == 2
    lines_after_first_run = history_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_first_run) == 2

    # "Fresh process" run: same resume-path already has state on disk from above.
    monkeypatch.setattr(sys, "argv", base_argv + ["--epochs", "4"])
    train_module.main()

    second_state = torch.load(resume_path, map_location="cpu")
    assert second_state["epoch"] == 4
    assert len(second_state["history"]) == 4
    assert [h["epoch"] for h in second_state["history"]] == [1, 2, 3, 4]
    lines_after_second_run = history_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_second_run) == 4  # appended, not overwritten


def test_train_main_early_stops_on_plateau(tmp_path, monkeypatch):
    """With --patience set low and an untrained model on a tiny synthetic
    dataset (mIoU won't meaningfully improve epoch to epoch), training must
    stop well before the --epochs hard cap."""
    import sys

    data_dir = tmp_path / "bcss_smoke"
    prepare_bcss.cmd_synthetic(data_dir, count=6, region_size=96, val_fraction=0.34, seed=4)

    out = tmp_path / "checkpoint.pth"
    history_jsonl = tmp_path / "history.jsonl"

    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--data-dir", str(data_dir),
        "--out", str(out),
        "--crop-size", "64",
        "--batch-size", "2",
        "--device", "cpu",
        "--history-jsonl", str(history_jsonl),
        "--epochs", "50",
        "--min-epochs", "1",
        "--patience", "1",
    ])
    train_module.main()

    lines = history_jsonl.read_text(encoding="utf-8").splitlines()
    assert 1 <= len(lines) < 50  # stopped well short of the hard cap
