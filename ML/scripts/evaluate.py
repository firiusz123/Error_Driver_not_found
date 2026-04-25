#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsiness.data import DrowsinessDataset
from drowsiness.models import build_model
from drowsiness.train_utils import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    ds = DrowsinessDataset(args.manifest, image_size=args.image_size, train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", "mobilenet_v2")
    model = build_model(model_name=model_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, dl, criterion, device)
    print(f"loss={loss:.4f} acc={acc:.4f}")


if __name__ == "__main__":
    main()
