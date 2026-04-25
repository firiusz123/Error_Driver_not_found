# Driver Drowsiness Detection (PyTorch + MediaPipe + MobileNetV2)

This project implements a lightweight drowsiness detector for:
- `n7i5x9/driver-drowsiness-dataset`
- `akahana/Driver-Drowsiness-Dataset`

Design target: deployable CNN pipeline with a practical RAM budget (8GB).

## 1) Map of Thoughts and Steps

1. Define objective and constraints
- Binary classification: `alert` vs `drowsy`
- CPU/GPU friendly and <=8GB RAM during training/inference

2. Build a unified data layer
- Download both datasets from Hugging Face
- Convert to one manifest CSV (`image_path,label,source`)
- Auto-infer labels from folder/file naming patterns

3. ROI preprocessing with MediaPipe
- Detect face bounding boxes
- Crop + margin to focus on eyes/mouth/face
- Drop images where no face is detected
- Save filtered ROI manifest

4. Train/val/test split
- Stratified split to preserve class balance

5. Model candidates
- Ready model: `MobileNetV2` (pretrained, lightweight, fast)
- Custom model: small `CustomDrowsinessCNN`

6. Train and compare
- Same augmentations, optimizer, and split
- Track val accuracy and keep best checkpoint

7. Deployable inference
- Single-image inference script
- Optional runtime ROI extraction with MediaPipe

## 2) Is ROI Extraction Worth It?

Short answer: yes, usually worth it for this task.

Why it helps:
- Reduces background noise (dashboard/window/lighting clutter)
- Forces model attention to face cues (eye closure, yawning)
- Can improve generalization across cars/cameras

Tradeoff:
- Adds one extra model stage (face detector)
- Slight latency overhead

Practical recommendation:
- Use MediaPipe ROI extraction offline as preprocessing for training (recommended)
- For real-time deployment, test both:
  - end-to-end on full frame
  - ROI-first pipeline
- Keep ROI if accuracy gain is meaningful for your target FPS/latency budget

## 3) Project Structure

- `scripts/download_hf_datasets.py` download dataset repos
- `scripts/build_manifest.py` build merged labeled manifest
- `scripts/extract_roi_mediapipe.py` face crop + filter no-face images
- `scripts/split_manifest.py` stratified train/val/test split
- `scripts/train.py` train MobileNetV2 or custom CNN
- `scripts/evaluate.py` evaluate saved checkpoint on a manifest
- `scripts/infer_image.py` single image inference

Core modules:
- `src/drowsiness/data.py` dataset loader + transforms
- `src/drowsiness/roi.py` MediaPipe face ROI extractor
- `src/drowsiness/models.py` model factory (`mobilenet_v2`, `custom_cnn`)
- `src/drowsiness/train_utils.py` train/eval loops

## 4) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5) End-to-End Commands

1. Download datasets
```bash
python scripts/download_hf_datasets.py
```

2. Build manifest from both datasets
```bash
python scripts/build_manifest.py \
  --roots data/raw/n7i5x9__driver-drowsiness-dataset data/raw/akahana__Driver-Drowsiness-Dataset \
  --output data/manifests/raw_manifest.csv
```

3. Extract MediaPipe face ROI
```bash
python scripts/extract_roi_mediapipe.py \
  --manifest data/manifests/raw_manifest.csv \
  --out-dir data/roi_images \
  --out-manifest data/manifests/roi_manifest.csv
```

4. Split
```bash
python scripts/split_manifest.py \
  --manifest data/manifests/roi_manifest.csv \
  --out-dir data/manifests
```

5. Train MobileNetV2 (recommended baseline)
```bash
python scripts/train.py \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --test-manifest data/manifests/test.csv \
  --model-name mobilenet_v2 \
  --pretrained \
  --batch-size 32 \
  --image-size 160 \
  --epochs 20 \
  --amp
```

6. Train custom CNN
```bash
python scripts/train.py \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --test-manifest data/manifests/test.csv \
  --model-name custom_cnn \
  --batch-size 32 \
  --image-size 160 \
  --epochs 20 \
  --amp
```

7. Inference
```bash
python scripts/infer_image.py \
  --image /path/to/image.jpg \
  --checkpoint artifacts/best_mobilenet_v2.pt \
  --use-roi
```

## 6) 8GB RAM/GPU Safety Tips

- Start with `image_size=160`, `batch_size=16` if memory is tight
- Use `--amp` for mixed precision on CUDA
- Set `num_workers=2` if host RAM pressure appears
- Keep ROI crops to reduce input variance and speed up training

