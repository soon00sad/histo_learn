"""Train HistoVision's BCSS segmentation model.

DeepLabV3+ (ResNet-18 encoder, ImageNet-pretrained) via segmentation-models-
pytorch — see docs/MODEL.md for why this architecture was chosen (CPU-inference
budget vs. accuracy trade-off). Training hyperparameters are CLI flags rather
than config.yaml entries: unlike the deployed app's runtime config (which has
one correct value per environment), training hyperparameters are inherently
varied per run — that's the deployed/training config split this project
already draws elsewhere (config.yaml has no training section).

Usage (real training — run on a GPU machine, not this dev box):
    python -m src.training.train \\
        --data-dir data/bcss --out models/segmentation.pth \\
        --epochs 40 --batch-size 16 --device cuda

Usage (CPU smoke test — proves the pipeline runs, not that it's accurate):
    python -m src.training.prepare_bcss synthetic --out data/bcss_smoke --count 8
    python -m src.training.train --data-dir data/bcss_smoke --out /tmp/smoke.pth \\
        --epochs 1 --batch-size 2 --crop-size 128 --device cpu
"""
from __future__ import annotations

import argparse
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.training.dataset import BcssDataset, list_samples
from src.training.losses import WeightedCeDiceLoss, compute_class_weights
from src.training.metrics import ConfusionMatrix, count_class_pixels
from src.utils.bcss_classes import load_bcss_classes
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_model(encoder_name: str, num_classes: int, encoder_weights: str | None = "imagenet") -> torch.nn.Module:
    return smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights, in_channels=3, classes=num_classes)


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate(model, loader, num_classes, device) -> ConfusionMatrix:
    model.eval()
    cm = ConfusionMatrix(num_classes)
    for images, masks in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()
        cm.update(preds, masks)
    return cm


def log_per_class_iou(cm: ConfusionMatrix) -> None:
    taxonomy = load_bcss_classes()
    iou = cm.per_class_iou()
    for bcss_class in taxonomy.classes:
        value = iou[bcss_class.model_index].item()
        label = f"{value:.3f}" if value == value else "n/a (not in val set)"  # NaN check
        logger.info("  %-24s IoU = %s", bcss_class.name_en, label)
    logger.info("mIoU = %.4f", cm.mean_iou())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Output of prepare_bcss.py")
    parser.add_argument("--out", type=Path, required=True, help="Checkpoint path (state_dict, like models/model.pth)")
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency CE weighting")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--stain-normalize", action="store_true", help="Apply Macenko normalization (matches inference-time preprocessing)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device)
    taxonomy = load_bcss_classes()

    train_samples = list_samples(args.data_dir, "train")
    val_samples = list_samples(args.data_dir, "val")
    logger.info("Loaded %d train / %d val samples from %s", len(train_samples), len(val_samples), args.data_dir)

    train_ds = BcssDataset(train_samples, taxonomy, args.crop_size, augment=not args.no_augment, stain_normalize=args.stain_normalize)
    # Validation always uses the un-augmented, un-stain-normalized path: metrics
    # should reflect the raw model, not preprocessing choices under test.
    val_ds = BcssDataset(val_samples, taxonomy, args.crop_size, augment=False, stain_normalize=False) if val_samples else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_ds else None

    class_weights = None
    if not args.no_class_weights:
        logger.info("Scanning training set for class pixel counts (for weighted CE)...")
        counts = count_class_pixels(train_ds, taxonomy.num_classes)
        class_weights = compute_class_weights(counts).to(device)
        logger.info("Class weights: %s", {c.name_en: round(class_weights[c.model_index].item(), 3) for c in taxonomy.classes})

    model = build_model(args.encoder, taxonomy.num_classes).to(device)
    criterion = WeightedCeDiceLoss(class_weights=class_weights, dice_weight=args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_miou = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.info("Epoch %d/%d — train loss %.4f", epoch, args.epochs, train_loss)

        if val_loader is not None:
            cm = evaluate(model, val_loader, taxonomy.num_classes, device)
            miou = cm.mean_iou()
            logger.info("Epoch %d — val mIoU %.4f", epoch, miou)
            if miou >= best_miou:
                best_miou = miou
                torch.save(model.state_dict(), args.out)
                logger.info("Saved new best checkpoint to %s (mIoU=%.4f)", args.out, miou)
        else:
            # No val split (e.g. a 1-2 sample smoke run): save every epoch so
            # `--out` always ends up with something to inspect.
            torch.save(model.state_dict(), args.out)

    if val_loader is not None:
        logger.info("Final per-class IoU (best checkpoint's last validated epoch):")
        cm = evaluate(model, val_loader, taxonomy.num_classes, device)
        log_per_class_iou(cm)
    logger.info("Training complete. Best val mIoU: %.4f. Checkpoint: %s", max(best_miou, 0.0), args.out)


if __name__ == "__main__":
    main()
