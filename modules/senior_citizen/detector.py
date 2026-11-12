from pathlib import Path

import cv2
import numpy as np

from modules.common.face import detect_faces

YUNET_PATH = Path(__file__).resolve().parents[2] / "assets" / "models" / "face_detection_yunet_2023mar.onnx"

MIN_FACE_SIZE = 40
YUNET_SCORE_THRESHOLD = 0.7


class HaarDetector:
    name = "haar"

    def detect(self, image_rgb):
        return [(x, y, w, h, 1.0) for x, y, w, h in detect_faces(image_rgb, MIN_FACE_SIZE)]


class YuNetDetector:
    name = "yunet"

    def __init__(self, model_path=YUNET_PATH, score_threshold=YUNET_SCORE_THRESHOLD):
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold, 0.3, 5000
        )

    def detect(self, image_rgb):
        height, width = image_rgb.shape[:2]
        self._detector.setInputSize((width, height))
        _, faces = self._detector.detect(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if faces is None:
            return []

        boxes = []
        for row in faces:
            x, y, w, h = (int(round(v)) for v in row[:4])
            if min(w, h) >= MIN_FACE_SIZE:
                boxes.append((x, y, w, h, float(row[-1])))
        return boxes


def load_detector(model_path=YUNET_PATH):
    if Path(model_path).is_file() and Path(model_path).stat().st_size > 10_000:
        try:
            return YuNetDetector(model_path)
        except Exception:
            pass
    return HaarDetector()
