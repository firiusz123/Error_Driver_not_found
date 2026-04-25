#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def normalize_label_name(name: str):
    s = str(name).lower().replace("_", " ").replace("-", " ").strip()
    if "non drowsy" in s or "not drowsy" in s:
        return 0
    if "drowsy" in s or "sleep" in s or "fatigue" in s or "yawn" in s:
        return 1
    if "alert" in s or "awake" in s or "open" in s or "normal" in s:
        return 0
    return None


def discover_data_files(root: Path) -> Dict[str, List[str]]:
    files: Dict[str, List[str]] = {}
    for p in sorted((root / "data").glob("*.parquet")):
        split = p.name.split("-")[0]
        files.setdefault(split, []).append(str(p))
    return files


class HFSplitDataset(Dataset):
    def __init__(self, split_ds, transform):
        self.ds = split_ds
        self.transform = transform

        label_names = split_ds.features["label"].names
        self.label_map = {i: normalize_label_name(name) for i, name in enumerate(label_names)}
        self.valid_idx = [i for i in range(len(split_ds)) if self.label_map.get(int(split_ds[i]["label"])) is not None]

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        row = self.ds[self.valid_idx[idx]]
        image = row["image"].convert("RGB")
        label = self.label_map[int(row["label"])]
        x = self.transform(image)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


def build_model(checkpoint: dict, device: str):
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def eval_loader(model, loader, device: str):
    cm = torch.zeros((2, 2), dtype=torch.long)
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)

            total += y.size(0)
            correct += (pred == y).sum().item()

            for t, p in zip(y.cpu(), pred.cpu()):
                cm[int(t), int(p)] += 1

    acc = correct / total if total > 0 else 0.0
    return acc, cm, total


def print_metrics(name: str, acc: float, cm: torch.Tensor, total: int):
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    print(f"\n=== {name} ===")
    print(f"samples={total}")
    print(f"accuracy={acc:.4f}")
    print("confusion_matrix (rows=true, cols=pred) [alert, drowsy]:")
    print(cm.numpy())
    print(f"TN={tn} FP={fp} FN={fn} TP={tp}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/mobilenetv2_minimal_baseline.pt")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "data/raw/n7i5x9__driver-drowsiness-dataset",
            "data/raw/akahana__Driver-Drowsiness-Dataset",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    image_size = int(checkpoint.get("img_size", args.image_size))

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = build_model(checkpoint, device)

    overall_cm = torch.zeros((2, 2), dtype=torch.long)
    overall_total = 0
    overall_correct = 0

    for root_str in args.roots:
        root = Path(root_str)
        data_files = discover_data_files(root)
        if "test" not in data_files:
            print(f"Skipping {root}: no test split parquet files found.")
            continue

        ds = load_dataset("parquet", data_files={"test": data_files["test"]})
        test_ds = HFSplitDataset(ds["test"], transform)
        loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        acc, cm, total = eval_loader(model, loader, device)
        print_metrics(root.name, acc, cm, total)

        overall_cm += cm
        overall_total += total
        overall_correct += int(cm.trace().item())

    if overall_total == 0:
        raise RuntimeError("No test samples were evaluated. Check dataset roots.")

    overall_acc = overall_correct / overall_total
    print_metrics("OVERALL", overall_acc, overall_cm, overall_total)


if __name__ == "__main__":
    main()
