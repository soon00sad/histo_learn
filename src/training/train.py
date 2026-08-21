"""Train HistoVision's BCSS segmentation model.

DeepLabV3+ (ResNet-18 encoder, ImageNet-pretrained) via segmentation-models-
pytorch — see docs/MODEL.md for why this architecture was chosen (CPU-inference
budget vs. accuracy trade-off). Training hyperparameters are CLI flags rather
than config.yaml entries: unlike the deployed app's runtime config (which has
one correct value per environment), training hyperparameters are inherently
varied per run — that's the deployed/training config split this project
already draws elsewhere (config.yaml has no training section).

Usage (real training — run on a GPU machine, not this dev box). --resume-path
and --history-jsonl should point at a persistent (e.g. Drive-mounted) path on
a free-tier Colab GPU, so re-running this exact command after a dropped
session resumes from the last completed epoch instead of starting over:
    python -m src.training.train \\
        --data-dir data/bcss --out models/segmentation.pth \\
        --resume-path /content/drive/MyDrive/histovision/checkpoints/resume.pt \\
        --history-jsonl /content/drive/MyDrive/histovision/logs/history.jsonl \\
        --stain-normalize --device cuda

Usage (CPU smoke test — proves the pipeline runs, not that it's accurate):
    python -m src.training.prepare_bcss synthetic --out data/bcss_smoke --count 8
    python -m src.training.train --data-dir data/bcss_smoke --out /tmp/smoke.pth \\
        --epochs 1 --batch-size 2 --crop-size 128 --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from src.training.dataset import BcssDataset, list_samples
from src.training.losses import WeightedCeDiceLoss, compute_class_weights
from src.training.metrics import ConfusionMatrix, count_class_pixels
from src.utils.bcss_classes import BcssTaxonomy, load_bcss_classes
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Classes a pathologist would consider diagnostically central, as opposed to
# rare/ambiguous ones (nerve, undetermined, other, ...) that BCSS's own
# long-tailed distribution makes hard to learn well from ~150 regions. The
# full 16-class mIoU is the honest headline number, but this subset is what
# actually matters for the "is this a plausible demo" judgment call — see
# docs/MODEL.md.
CLINICAL_CLASS_NAMES = (
    "tumor", "dcis", "stroma", "lymphocytic_infiltrate",
    "necrosis_or_debris", "normal_acinus_or_duct",
)


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


def log_class_histogram(counts: torch.Tensor, taxonomy: BcssTaxonomy) -> None:
    """Per-class labeled-pixel counts and share of the training set, sorted
    by frequency. Answers "how many of the 16 classes actually showed up,
    and how skewed is the distribution" before training even starts —
    needed to judge whether compute_class_weights' imbalance cap is
    reasonable for this particular data pull."""
    total = int(counts.sum().item())
    order = torch.argsort(counts, descending=True)
    present = int((counts > 0).sum().item())
    logger.info(
        "Class pixel histogram: %d/%d classes present, %d total labeled pixels:",
        present, taxonomy.num_classes, total,
    )
    for idx in order.tolist():
        bcss_class = taxonomy.class_by_model_index(idx)
        count = int(counts[idx].item())
        pct = 100.0 * count / total if total else 0.0
        flag = "" if count > 0 else "  <- ABSENT from this data pull"
        logger.info("  %-24s %12d px  (%6.2f%%)%s", bcss_class.name_en, count, pct, flag)


def clinical_mean_iou(cm: ConfusionMatrix, taxonomy: BcssTaxonomy) -> float:
    """mIoU restricted to CLINICAL_CLASS_NAMES — see the module-level
    constant's docstring for why this is reported alongside the full
    16-class mIoU rather than instead of it."""
    indices = [c.model_index for c in taxonomy.classes if c.name_en in CLINICAL_CLASS_NAMES]
    iou = cm.per_class_iou()[indices]
    valid = iou[~torch.isnan(iou)]
    return valid.mean().item() if valid.numel() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="Output of prepare_bcss.py")
    parser.add_argument("--out", type=Path, required=True, help="Best-mIoU checkpoint path (state_dict only, like models/model.pth)")
    parser.add_argument("--encoder", default="resnet18")
    parser.add_argument("--epochs", type=int, default=100, help="Hard cap; --patience usually stops training sooner")
    parser.add_argument("--patience", type=int, default=10, help="Stop after this many epochs with no val mIoU improvement")
    parser.add_argument("--min-epochs", type=int, default=8, help="Never early-stop before this many epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency CE weighting")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--stain-normalize", action="store_true", help="Apply Macenko normalization (matches inference-time preprocessing)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume-path", type=Path, default=None,
        help="Full training state (weights+optimizer+epoch+history), saved every epoch and reloaded "
             "from here on restart if it exists. Point this at a Drive-mounted path in Colab so a "
             "dropped session resumes instead of restarting from scratch.",
    )
    parser.add_argument(
        "--history-jsonl", type=Path, default=None,
        help="Append one JSON line per epoch (train_loss, val mIoU, clinical mIoU) here — a durable, "
             "human-tailable record independent of --resume-path's binary checkpoint.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    taxonomy = load_bcss_classes()

    if args.history_jsonl:
        args.history_jsonl.parent.mkdir(parents=True, exist_ok=True)
        log_file = str(args.history_jsonl.with_suffix(".log").resolve())
        # Guard against duplicate handlers (and duplicated log lines) if a
        # notebook cell calling main() gets re-run without restarting the
        # kernel, which is the normal way to resume after a Colab drop.
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_file for h in logger.handlers):
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(file_handler)

    train_samples = list_samples(args.data_dir, "train")
    val_samples = list_samples(args.data_dir, "val")
    logger.info("Loaded %d train / %d val samples from %s", len(train_samples), len(val_samples), args.data_dir)

    train_ds = BcssDataset(train_samples, taxonomy, args.crop_size, augment=not args.no_augment, stain_normalize=args.stain_normalize)
    # Validation always uses the un-augmented, un-stain-normalized path: metrics
    # should reflect the raw model, not preprocessing choices under test.
    val_ds = BcssDataset(val_samples, taxonomy, args.crop_size, augment=False, stain_normalize=False) if val_samples else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=len(train_ds) > args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if val_ds else None

    logger.info("Scanning training set for class pixel counts...")
    counts = count_class_pixels(train_ds, taxonomy.num_classes)
    log_class_histogram(counts, taxonomy)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(counts).to(device)
        logger.info("Class weights (capped, mean=1): %s", {c.name_en: round(class_weights[c.model_index].item(), 3) for c in taxonomy.classes})

    model = build_model(args.encoder, taxonomy.num_classes).to(device)
    criterion = WeightedCeDiceLoss(class_weights=class_weights, dice_weight=args.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_miou = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []

    if args.resume_path and args.resume_path.exists():
        logger.info("Resuming from checkpoint: %s", args.resume_path)
        state = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        start_epoch = state["epoch"] + 1
        best_miou = state["best_miou"]
        epochs_without_improvement = state["epochs_without_improvement"]
        history = state["history"]
        logger.info(
            "Resumed at epoch %d (best mIoU so far: %.4f, %d epochs without improvement)",
            start_epoch, best_miou, epochs_without_improvement,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        logger.info("Epoch %d/%d — train loss %.4f", epoch, args.epochs, train_loss)
        record = {"epoch": epoch, "train_loss": train_loss}

        if val_loader is not None:
            cm = evaluate(model, val_loader, taxonomy.num_classes, device)
            miou = cm.mean_iou()
            c_miou = clinical_mean_iou(cm, taxonomy)
            record.update({"val_miou": miou, "val_clinical_miou": c_miou})
            logger.info("Epoch %d — val mIoU %.4f (clinical-subset mIoU %.4f)", epoch, miou, c_miou)

            if miou > best_miou:
                best_miou = miou
                epochs_without_improvement = 0
                torch.save(model.state_dict(), args.out)
                logger.info("Saved new best checkpoint to %s (mIoU=%.4f)", args.out, miou)
            else:
                epochs_without_improvement += 1
        else:
            # No val split (e.g. a 1-2 sample smoke run): save every epoch so
            # `--out` always ends up with something to inspect.
            torch.save(model.state_dict(), args.out)

        history.append(record)
        if args.history_jsonl:
            with open(args.history_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        if args.resume_path:
            args.resume_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_miou": best_miou,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            }, args.resume_path)

        if val_loader is not None and epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            logger.info(
                "Early stopping at epoch %d: no val mIoU improvement in %d epochs (plateau).",
                epoch, args.patience,
            )
            break

    if val_loader is not None:
        logger.info("Reloading best checkpoint (mIoU=%.4f) for final evaluation...", best_miou)
        model.load_state_dict(torch.load(args.out, map_location=device))
        cm = evaluate(model, val_loader, taxonomy.num_classes, device)
        log_per_class_iou(cm)
        logger.info(
            "Clinical-subset mIoU (%s): %.4f",
            ", ".join(CLINICAL_CLASS_NAMES), clinical_mean_iou(cm, taxonomy),
        )
    logger.info("Training complete. Best val mIoU: %.4f. Checkpoint: %s", max(best_miou, 0.0), args.out)


if __name__ == "__main__":
    main()
