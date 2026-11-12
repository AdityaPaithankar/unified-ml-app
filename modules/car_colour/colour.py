import json

import cv2
import numpy as np

from model_registry import MODELS

COLOUR_SIZE = MODELS["car_colour"]["input_size"][0]
COLOUR_NAMES = json.loads(MODELS["car_colour"]["class_names"].read_text(encoding="utf-8"))

CROP_MARGIN = 0.05
MIN_CROP_SIDE = 32
RELIABLE_CONFIDENCE = 0.40


def _default_runner(model_key, batch):
    from modules.common.worker_client import predict

    return predict(model_key, batch)


def crop_box(image_rgb, box, margin=CROP_MARGIN):
    height, width = image_rgb.shape[:2]
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    x2, y2 = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image_rgb[y1:y2, x1:x2]


def classify_colours(image_rgb, boxes, runner=None):
    """Return one dict per box: colour, confidence, reliable. Input is raw 0-255 pixels."""
    crops = [crop_box(image_rgb, box) for box in boxes]
    valid = [i for i, c in enumerate(crops) if c is not None]
    results = [{"colour": None, "confidence": 0.0, "reliable": False} for _ in boxes]
    if not valid:
        return results

    batch = np.stack([
        cv2.resize(crops[i], (COLOUR_SIZE, COLOUR_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        for i in valid
    ])
    probabilities = np.asarray((runner or _default_runner)("car_colour", batch))

    for index, row in zip(valid, probabilities):
        best = int(np.argmax(row))
        side = min(crops[index].shape[:2])
        confidence = float(row[best])
        results[index] = {
            "colour": COLOUR_NAMES[best],
            "confidence": round(confidence, 3),
            "reliable": confidence >= RELIABLE_CONFIDENCE and side >= MIN_CROP_SIDE,
        }
    return results
