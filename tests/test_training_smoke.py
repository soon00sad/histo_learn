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

import pytest
import torch
from torch.utils.data import DataLoader

from src.training import prepare_bcss
from src.training.dataset import BcssDataset, IGNORE_INDEX, list_samples, remap_mask
from src.training.losses import WeightedCeDiceLoss, compute_class_weights
from src.training.metrics import ConfusionMatrix, count_class_pixels
from src.training.train import build_model, evaluate, train_one_epoch
from src.utils.bcss_classes import load_bcss_classes

import numpy as np


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
