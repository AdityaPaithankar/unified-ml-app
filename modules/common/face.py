import cv2
import numpy as np

_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
        if _cascade.empty():
            raise RuntimeError(f"Could not load face cascade: {path}")
    return _cascade


def detect_faces(image_rgb, min_size=40):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    boxes = [tuple(int(v) for v in f) for f in faces]
    return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)


def square_crop(image_rgb, center_x, center_y, side):
    side = max(int(round(side)), 8)
    half = side // 2
    x1, y1 = int(round(center_x)) - half, int(round(center_y)) - half
    x2, y2 = x1 + side, y1 + side

    height, width = image_rgb.shape[:2]
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)

    crop = image_rgb[
        max(0, y1):min(height, y2),
        max(0, x1):min(width, x2),
    ]
    if pad_left or pad_top or pad_right or pad_bottom:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE
        )
    return crop


def resize_rgb(image_rgb, size):
    return cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(
        np.float32
    )
