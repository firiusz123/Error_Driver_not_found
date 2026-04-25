import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class FaceRoiExtractor:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        margin_ratio: float = 0.2,
        model_path: Optional[str] = None,
    ):
        self.margin_ratio = margin_ratio
        resolved_model_path = self._resolve_model_path(model_path)
        if not resolved_model_path:
            raise ValueError(
                "MediaPipe Tasks model is required. Set --mp-model or MEDIAPIPE_FACE_MODEL "
                "to a local face detector model path (for example blaze_face_short_range.tflite)."
            )

        base_options = mp.tasks.BaseOptions(model_asset_path=resolved_model_path)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=min_detection_confidence,
        )
        self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Optional[str]:
        candidates = [
            model_path,
            os.getenv("MEDIAPIPE_FACE_MODEL"),
            "models/blaze_face_short_range.tflite",
            "models/face_detector.task",
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if path.exists():
                return str(path.resolve())
        return None

    def extract_with_bbox(self, bgr_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if not result or not result.detections:
            return None, None

        detection = max(result.detections, key=self._detection_score)
        box = detection.bounding_box
        h, w = bgr_image.shape[:2]

        x1 = int(box.origin_x)
        y1 = int(box.origin_y)
        bw = int(box.width)
        bh = int(box.height)

        # Guard against malformed outputs.
        if bw <= 0 or bh <= 0:
            return None, None

        mx = int(bw * self.margin_ratio)
        my = int(bh * self.margin_ratio)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x1 + bw + 2 * mx)
        y2 = min(h, y1 + bh + 2 * my)

        if x2 <= x1 or y2 <= y1:
            return None, None

        face = bgr_image[y1:y2, x1:x2]
        if face.size == 0:
            return None, None

        return face, (x1, y1, x2, y2)

    @staticmethod
    def _detection_score(detection) -> float:
        categories = getattr(detection, "categories", None)
        if categories:
            return float(categories[0].score)
        return 0.0

    def extract(self, bgr_image: np.ndarray) -> Optional[np.ndarray]:
        face, _ = self.extract_with_bbox(bgr_image)
        return face

    def close(self) -> None:
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def extract_face(image: np.ndarray, margin: float = 0.2, model_path: Optional[str] = None):
    with FaceRoiExtractor(margin_ratio=margin, model_path=model_path) as extractor:
        return extractor.extract_with_bbox(image)


def visualize_pair(original: np.ndarray, face: np.ndarray, bbox=None):
    display = original.copy()
    if bbox:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

    face_resized = cv2.resize(face, (display.shape[1], display.shape[0]))
    combined = cv2.hconcat([display, face_resized])
    cv2.putText(
        combined,
        "Q: save | E: skip | ESC: exit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return combined
