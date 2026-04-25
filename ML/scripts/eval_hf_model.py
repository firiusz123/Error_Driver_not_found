#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageClassification

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label_name(name: str) -> Optional[int]:
    s = str(name).strip().lower().replace("-", " ").replace("_", " ")

    # Drowsy-like
    if any(k in s for k in ["drowsy", "sleep", "closed", "yawn", "fatigue", "tired"]):
        if "non drowsy" in s or "not drowsy" in s:
            return 0
        return 1

    # Alert-like
    if any(k in s for k in ["alert", "awake", "non drowsy", "not drowsy", "open", "normal"]):
        return 0

    return None


def infer_gt_label_from_path(path: Path) -> Optional[int]:
    return normalize_label_name(str(path))


def list_folder_samples(roots: List[str]) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            label = infer_gt_label_from_path(path)
            if label is None:
                continue
            samples.append((path, label))
    return samples


def discover_parquet_data_files(root: Path) -> Dict[str, List[str]]:
    data_dir = root / "data"
    if not data_dir.exists():
        return {}

    files_by_split: Dict[str, List[str]] = {}
    for p in sorted(data_dir.glob("*.parquet")):
        # Expects names like: train-00000-of-00002.parquet
        split = p.name.split("-")[0]
        files_by_split.setdefault(split, []).append(str(p))
    return files_by_split


def build_hf_label_map(split_ds) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    features = getattr(split_ds, "features", None)
    if not features or "label" not in features:
        return mapping

    label_feature = features["label"]
    names = getattr(label_feature, "names", None)
    if not names:
        return mapping

    for idx, name in enumerate(names):
        mapped = normalize_label_name(name)
        if mapped is not None:
            mapping[idx] = mapped
    return mapping


def image_from_hf_cell(cell) -> Optional[Image.Image]:
    # datasets.Image may decode directly into PIL.Image
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")

    # Or it can be a dict with bytes/path.
    if isinstance(cell, dict):
        if cell.get("bytes"):
            try:
                from io import BytesIO

                return Image.open(BytesIO(cell["bytes"])).convert("RGB")
            except Exception:
                return None
        if cell.get("path"):
            try:
                return Image.open(cell["path"]).convert("RGB")
            except Exception:
                return None

    return None


def build_pred_map(model) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    id2label = getattr(model.config, "id2label", {})
    for idx, label_name in id2label.items():
        try:
            idx_int = int(idx)
        except Exception:
            idx_int = idx
        mapped = normalize_label_name(str(label_name))
        if mapped is not None:
            mapping[idx_int] = mapped
    return mapping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "data/raw/n7i5x9__driver-drowsiness-dataset",
            "data/raw/akahana__Driver-Drowsiness-Dataset",
        ],
        help="Dataset roots (HF parquet snapshots and/or image folders).",
    )
    parser.add_argument("--model-id", default="chbh7051/driver-drowsiness-detection")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="Max items for quick test.")
    return parser.parse_args()


def load_processor(model_id: str):
    # Older checkpoints (like this one) may publish ViTFeatureExtractor-only configs.
    try:
        from transformers import AutoImageProcessor

        return AutoImageProcessor.from_pretrained(model_id)
    except Exception as e:
        from transformers import AutoFeatureExtractor

        print(f"AutoImageProcessor failed, falling back to AutoFeatureExtractor: {e}")
        return AutoFeatureExtractor.from_pretrained(model_id)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    processor = load_processor(args.model_id)
    model = AutoModelForImageClassification.from_pretrained(args.model_id).to(device)
    model.eval()

    pred_map = build_pred_map(model)
    if not pred_map:
        print("Warning: model id2label could not be mapped; will fallback to raw class ids (0/1).")

    total = 0
    correct = 0
    skipped_unreadable = 0
    skipped_unmappable = 0
    scanned = 0

    # Confusion matrix (positive class = drowsy)
    tp = tn = fp = fn = 0

    # 1) Evaluate parquet-style HF snapshots if present.
    for root_str in args.roots:
        root = Path(root_str)
        if not root.exists():
            continue

        parquet_files = discover_parquet_data_files(root)
        if not parquet_files:
            continue

        from datasets import load_dataset

        ds_dict = load_dataset("parquet", data_files=parquet_files)
        for split_name, split_ds in ds_dict.items():
            hf_label_map = build_hf_label_map(split_ds)
            if not hf_label_map:
                print(f"Warning: no HF label map inferred for {root.name}:{split_name}; skipping split")
                continue

            n = len(split_ds)
            if args.limit > 0:
                n = min(n, args.limit)

            for i in tqdm(range(0, n, args.batch_size), desc=f"{root.name}:{split_name}"):
                batch = split_ds[i : min(i + args.batch_size, n)]
                images_raw = batch.get("image", [])
                labels_raw = batch.get("label", [])

                images = []
                labels = []
                for img_cell, y_raw in zip(images_raw, labels_raw):
                    scanned += 1
                    img = image_from_hf_cell(img_cell)
                    if img is None:
                        skipped_unreadable += 1
                        continue

                    if isinstance(y_raw, str):
                        y_true = normalize_label_name(y_raw)
                    else:
                        y_true = hf_label_map.get(int(y_raw))

                    if y_true is None:
                        skipped_unmappable += 1
                        continue

                    images.append(img)
                    labels.append(y_true)

                if not images:
                    continue

                inputs = processor(images=images, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    logits = model(**inputs).logits
                    pred_ids = torch.argmax(logits, dim=1).tolist()

                for y_true, pred_id in zip(labels, pred_ids):
                    if pred_id in pred_map:
                        y_pred = pred_map[pred_id]
                    elif pred_id in (0, 1):
                        y_pred = pred_id
                    else:
                        skipped_unmappable += 1
                        continue

                    total += 1
                    if y_pred == y_true:
                        correct += 1

                    if y_true == 1 and y_pred == 1:
                        tp += 1
                    elif y_true == 0 and y_pred == 0:
                        tn += 1
                    elif y_true == 0 and y_pred == 1:
                        fp += 1
                    else:
                        fn += 1

    # 2) Also evaluate plain image-folder samples (if any).
    folder_samples = list_folder_samples(args.roots)
    if args.limit > 0:
        folder_samples = folder_samples[: args.limit]

    for i in tqdm(range(0, len(folder_samples), args.batch_size), desc="folders"):
        batch = folder_samples[i : i + args.batch_size]

        images = []
        labels = []
        for path, y_true in batch:
            scanned += 1
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                skipped_unreadable += 1
                continue
            images.append(img)
            labels.append(y_true)

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=1).tolist()

        for y_true, pred_id in zip(labels, pred_ids):
            if pred_id in pred_map:
                y_pred = pred_map[pred_id]
            elif pred_id in (0, 1):
                y_pred = pred_id
            else:
                skipped_unmappable += 1
                continue

            total += 1
            if y_pred == y_true:
                correct += 1

            if y_true == 1 and y_pred == 1:
                tp += 1
            elif y_true == 0 and y_pred == 0:
                tn += 1
            elif y_true == 0 and y_pred == 1:
                fp += 1
            else:
                fn += 1

    if scanned == 0:
        raise ValueError("No dataset items found. Check --roots paths.")
    if total == 0:
        raise RuntimeError("No comparable predictions produced (all samples skipped/unmappable).")

    acc = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n=== Hugging Face Model Evaluation ===")
    print(f"model_id={args.model_id}")
    print(f"device={device}")
    print(f"scanned={scanned}")
    print(f"evaluated={total}")
    print(f"skipped_unreadable={skipped_unreadable}")
    print(f"skipped_unmappable={skipped_unmappable}")
    print(f"accuracy={acc:.4f}")
    print(f"precision_drowsy={precision:.4f}")
    print(f"recall_drowsy={recall:.4f}")
    print(f"f1_drowsy={f1:.4f}")
    print(f"confusion_matrix: TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    main()
