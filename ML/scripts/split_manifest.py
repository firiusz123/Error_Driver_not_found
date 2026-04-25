#!/usr/bin/env python3
import argparse
import csv
import os
import random
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", default="data/manifests")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_csv(path: str, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.manifest, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    random.seed(args.seed)
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row)

    train_rows, val_rows, test_rows = [], [], []

    for label, label_rows in by_class.items():
        random.shuffle(label_rows)
        n = len(label_rows)
        n_test = int(n * args.test_ratio)
        n_val = int(n * args.val_ratio)

        test_rows.extend(label_rows[:n_test])
        val_rows.extend(label_rows[n_test : n_test + n_val])
        train_rows.extend(label_rows[n_test + n_val :])

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    train_path = os.path.join(args.out_dir, "train.csv")
    val_path = os.path.join(args.out_dir, "val.csv")
    test_path = os.path.join(args.out_dir, "test.csv")

    write_csv(train_path, train_rows)
    write_csv(val_path, val_rows)
    write_csv(test_path, test_rows)

    print(f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    main()
