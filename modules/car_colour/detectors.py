import json
from pathlib import Path

import cv2
import numpy as np

from modules.common.model_download import get_model_root

MODEL_ROOT = get_model_root()
YOLO_WEIGHTS = MODEL_ROOT / "assets" / "models" / "yolov8n.pt"
ANCHOR_FILE = MODEL_ROOT / "Car-colour" / "anchors_v2.json"

COCO_PERSON, COCO_CAR = 0, 2
YOLO_CONFIDENCE = 0.25

CUSTOM_SIZE = 416
CUSTOM_GRID = 13
CUSTOM_CONFIDENCE = 0.45
CUSTOM_NMS_IOU = 0.30
CUSTOM_LABELS = ("car", "person")

_yolo = {}


def _detection(label, box, score):
    return {"label": label, "box": tuple(int(round(v)) for v in box), "score": round(float(score), 3)}


class YoloDetector:
    name = "yolo"

    def __init__(self, weights=YOLO_WEIGHTS, confidence=YOLO_CONFIDENCE):
        self.weights = Path(weights)
        self.confidence = confidence

    @staticmethod
    def available(weights=YOLO_WEIGHTS):
        try:
            import ultralytics  # noqa: F401
        except ImportError:
            return False
        return Path(weights).is_file()

    def _model(self):
        if "model" not in _yolo:
            from ultralytics import YOLO
            _yolo["model"] = YOLO(str(self.weights))
        return _yolo["model"]

    def detect(self, image_rgb):
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        result = self._model().predict(
            bgr, classes=[COCO_PERSON, COCO_CAR], conf=self.confidence, verbose=False
        )[0]
        found = []
        for box, score, cls in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy()
        ):
            found.append(_detection("car" if int(cls) == COCO_CAR else "person", box, score))
        return found


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def decode_custom(prediction, anchors, confidence=CUSTOM_CONFIDENCE, nms_iou=CUSTOM_NMS_IOU):
    """Decode one (13, 13, 3, 7) output of the custom KITTI detector into normalised boxes."""
    grid_y, grid_x = np.mgrid[0:CUSTOM_GRID, 0:CUSTOM_GRID]
    objectness = _sigmoid(prediction[..., 4])
    logits = prediction[..., 5:7]
    probabilities = np.exp(logits - logits.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    class_ids = probabilities.argmax(-1)
    scores = objectness * probabilities.max(-1)

    cx = (grid_x[..., None] + _sigmoid(prediction[..., 0])) / CUSTOM_GRID
    cy = (grid_y[..., None] + _sigmoid(prediction[..., 1])) / CUSTOM_GRID
    width = np.exp(np.clip(prediction[..., 2], -10, 10)) * anchors[:, 0]
    height = np.exp(np.clip(prediction[..., 3], -10, 10)) * anchors[:, 1]

    keep = scores >= confidence
    boxes = np.stack(
        [
            np.clip(cx - width / 2, 0, 1), np.clip(cy - height / 2, 0, 1),
            np.clip(cx + width / 2, 0, 1), np.clip(cy + height / 2, 0, 1),
        ],
        axis=-1,
    )[keep]
    scores, class_ids = scores[keep], class_ids[keep]

    selected = []
    for class_id in range(len(CUSTOM_LABELS)):
        order = [i for i in np.argsort(-scores) if class_ids[i] == class_id]
        while order:
            best = order.pop(0)
            selected.append(best)
            if order:
                overlaps = _iou(boxes[best], boxes[order])
                order = [i for i, o in zip(order, overlaps) if o <= nms_iou]
    return [(boxes[i], float(scores[i]), int(class_ids[i])) for i in selected]


class CustomDetector:
    name = "custom"

    def __init__(self, runner=None, anchor_file=ANCHOR_FILE, confidence=CUSTOM_CONFIDENCE):
        self.runner = runner
        self.confidence = confidence
        self.anchors = np.asarray(
            json.loads(Path(anchor_file).read_text())["anchors"], dtype=np.float32
        )

    def detect(self, image_rgb):
        runner = self.runner
        if runner is None:
            from modules.common.worker_client import predict as runner

        height, width = image_rgb.shape[:2]
        resized = cv2.resize(image_rgb, (CUSTOM_SIZE, CUSTOM_SIZE), interpolation=cv2.INTER_LINEAR)
        batch = (resized.astype(np.float32) / 255.0)[None]
        prediction = np.asarray(runner("car_detector", batch))[0]

        found = []
        for box, score, class_id in decode_custom(prediction, self.anchors, self.confidence):
            x1, y1, x2, y2 = box * np.array([width, height, width, height])
            found.append(_detection(CUSTOM_LABELS[class_id], (x1, y1, x2, y2), score))
        return found


def load_detector(kind="auto", runner=None):
    if kind == "yolo" or (kind == "auto" and YoloDetector.available()):
        return YoloDetector()
    return CustomDetector(runner=runner)

