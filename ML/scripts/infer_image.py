#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drowsiness.models import build_model
from drowsiness.roi import FaceRoiExtractor
from drowsiness.utils import ID_TO_LABEL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-roi", action="store_true")
    parser.add_argument("--mp-model", default=None, help="Path to MediaPipe face detector model file.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get("model_name", "mobilenet_v2")
    model = build_model(model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    if args.use_roi:
        with FaceRoiExtractor(model_path=args.mp_model) as extractor:
            roi = extractor.extract(image)
        if roi is not None:
            image = roi

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(image_rgb)

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    x = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())

    print(f"prediction={ID_TO_LABEL[pred]} alert_prob={probs[0].item():.4f} drowsy_prob={probs[1].item():.4f}")


if __name__ == "__main__":
    main()
