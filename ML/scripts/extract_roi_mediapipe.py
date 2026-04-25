#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import List

import cv2
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsiness.roi import FaceRoiExtractor, visualize_pair

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="data/raw")
    parser.add_argument("--output-subdir", default="crop")
    parser.add_argument("--margin-ratio", type=float, default=0.2)
    parser.add_argument("--min-det-conf", type=float, default=0.5)
    parser.add_argument("--mp-model", default=None, help="Path to MediaPipe face detector model file.")
    parser.add_argument("--window", default="ROI Screening")
    return parser.parse_args()


def list_images(dataset_root: Path, output_subdir: str) -> List[Path]:
    image_paths: List[Path] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        # Skip already-cropped files
        if output_subdir in path.parts:
            continue
        image_paths.append(path)
    image_paths.sort()
    return image_paths


def build_save_path(image_path: Path, dataset_root: Path, output_subdir: str) -> Path:
    rel = image_path.relative_to(dataset_root)
    return dataset_root / output_subdir / rel


def run():
    args = parse_args()
    input_root = Path(args.input_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    datasets = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not datasets:
        print(f"No dataset directories in {input_root}")
        return

    stats = {
        "seen": 0,
        "saved": 0,
        "skipped": 0,
        "no_face": 0,
        "empty_crop": 0,
        "read_error": 0,
    }

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)

    with FaceRoiExtractor(
        min_detection_confidence=args.min_det_conf,
        margin_ratio=args.margin_ratio,
        model_path=args.mp_model,
    ) as extractor:
        for dataset_dir in datasets:
            images = list_images(dataset_dir, args.output_subdir)
            if not images:
                continue

            print(f"\nDataset: {dataset_dir.name} | images to review: {len(images)}")

            for image_path in tqdm(images, desc=f"{dataset_dir.name}"):
                stats["seen"] += 1
                image = cv2.imread(str(image_path))
                if image is None:
                    stats["read_error"] += 1
                    continue

                face, bbox = extractor.extract_with_bbox(image)
                if face is None:
                    stats["no_face"] += 1
                    continue

                if face.size == 0:
                    stats["empty_crop"] += 1
                    continue

                panel = visualize_pair(image, face, bbox)
                info = f"{dataset_dir.name} | {image_path.name}"
                cv2.putText(panel, info, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(args.window, panel)

                while True:
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("q"):
                        save_path = build_save_path(image_path, dataset_dir, args.output_subdir)
                        save_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(save_path), face)
                        stats["saved"] += 1
                        break
                    if key == ord("e"):
                        stats["skipped"] += 1
                        break
                    if key == 27:
                        cv2.destroyAllWindows()
                        print("\nStopped early by user (ESC).")
                        print(stats)
                        return

    cv2.destroyAllWindows()
    print("\nReview completed.")
    print(stats)


if __name__ == "__main__":
    run()
