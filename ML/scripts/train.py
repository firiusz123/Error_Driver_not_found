#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsiness.config import TrainConfig
from drowsiness.data import DrowsinessDataset
from drowsiness.models import build_model
from drowsiness.train_utils import evaluate, train_one_epoch
from drowsiness.utils import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default="data/manifests/train.csv")
    parser.add_argument("--val-manifest", default="data/manifests/val.csv")
    parser.add_argument("--test-manifest", default="data/manifests/test.csv")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-name", choices=["mobilenet_v2", "custom_cnn"], default="mobilenet_v2")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(**vars(args))

    device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    use_amp = cfg.amp and device == "cuda"

    set_seed(cfg.seed)
    ensure_dir(cfg.out_dir)

    train_ds = DrowsinessDataset(cfg.train_manifest, image_size=cfg.image_size, train=True)
    val_ds = DrowsinessDataset(cfg.val_manifest, image_size=cfg.image_size, train=False)
    test_ds = DrowsinessDataset(cfg.test_manifest, image_size=cfg.image_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device == "cuda"),
    )

    model = build_model(cfg.model_name, pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc = 0.0
    best_path = os.path.join(cfg.out_dir, f"best_{cfg.model_name}.pt")

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch={epoch} train_loss={train_stats['loss']:.4f} train_acc={train_stats['acc']:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": cfg.model_name,
                    "image_size": cfg.image_size,
                    "best_val_acc": best_val_acc,
                },
                best_path,
            )

    print(f"best checkpoint: {best_path} (val_acc={best_val_acc:.4f})")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
