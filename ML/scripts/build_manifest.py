#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_label(path: Path) -> Optional[int]:
    p = str(path).lower()
    alert_keys = ["alert", "awake", "open_eye", "non_drowsy", "not_drowsy"]
    drowsy_keys = ["drowsy", "sleep", "closed_eye", "yawn", "fatigue"]

    if any(k in p for k in drowsy_keys):
        return 1
    if any(k in p for k in alert_keys):
        return 0
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", default="data/manifests/raw_manifest.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rows = []
    skipped = 0

    for root in args.roots:
        root_path = Path(root)
        for path in root_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            label = infer_label(path)
            if label is None:
                skipped += 1
                continue
            rows.append(
                {
                    "image_path": str(path.resolve()),
                    "label": label,
                    "source": root_path.name,
                }
            )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    drowsy = sum(1 for r in rows if r["label"] == 1)
    alert = total - drowsy
    print(f"saved {total} rows to {args.output}")
    print(f"class distribution: alert={alert}, drowsy={drowsy}, skipped_unknown={skipped}")


if __name__ == "__main__":
    main()
