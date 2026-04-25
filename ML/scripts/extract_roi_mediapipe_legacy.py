#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="data/raw")
    parser.add_argument("--output-subdir", default="crop")
    parser.add_argument("--margin-ratio", type=float, default=0.2)
    parser.add_argument("--min-det-conf", type=float, default=0.5)
    parser.add_argument("--model-selection", type=int, default=0, choices=[0, 1])
    parser.add_argument("--window", default="ROI Screening (Legacy MP)")
    return parser.parse_args()


def list_images(dataset_root: Path, output_subdir: str) -> List[Path]:
    image_paths: List[Path] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        if output_subdir in path.parts:
            continue
        image_paths.append(path)
    image_paths.sort()
    return image_paths


def build_save_path(image_path: Path, dataset_root: Path, output_subdir: str) -> Path:
    rel = image_path.relative_to(dataset_root)
    return dataset_root / output_subdir / rel


def extract_face_legacy(
    image_bgr,
    detector,
    margin_ratio: float,
) -> Tuple[Optional[object], Optional[Tuple[int, int, int, int]]]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = detector.process(rgb)

    if not result.detections:
        return None, None

    detection = max(result.detections, key=lambda d: d.score[0])
    box = detection.location_data.relative_bounding_box
    h, w = image_bgr.shape[:2]

    x1 = int(box.xmin * w)
    y1 = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)

    if bw <= 0 or bh <= 0:
        return None, None

    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    x1 = max(0, x1 - mx)
    y1 = max(0, y1 - my)
    x2 = min(w, x1 + bw + 2 * mx)
    y2 = min(h, y1 + bh + 2 * my)

    if x2 <= x1 or y2 <= y1:
        return None, None

    face = image_bgr[y1:y2, x1:x2]
    if face is None or face.size == 0:
        return None, None

    return face, (x1, y1, x2, y2)


def build_preview(original, face, bbox, dataset_name: str, image_name: str):
    display = original.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    face_resized = cv2.resize(face, (display.shape[1], display.shape[0]))
    combined = cv2.hconcat([display, face_resized])

    cv2.putText(combined, "Q: save | E: skip | ESC: exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(combined, f"{dataset_name} | {image_name}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return combined


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

    with mp.solutions.face_detection.FaceDetection(
        model_selection=args.model_selection,
        min_detection_confidence=args.min_det_conf,
    ) as detector:
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

                face, bbox = extract_face_legacy(image, detector, args.margin_ratio)
                if face is None:
                    stats["no_face"] += 1
                    continue

                if face.size == 0:
                    stats["empty_crop"] += 1
                    continue

                panel = build_preview(image, face, bbox, dataset_dir.name, image_path.name)
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
