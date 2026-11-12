import cv2
import numpy as np

MIN_TORSO_PIXELS = 1500
SKIN_FALLBACK_RATIO = 0.4
CLUSTERS = 3

_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
        if _cascade.empty():
            raise RuntimeError(f"Could not load face cascade: {path}")
    return _cascade


def detect_face(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def torso_region(image_shape, face_box):
    height, width = image_shape[:2]
    x, y, w, h = face_box

    x1 = int(max(0, x - 0.4 * w))
    x2 = int(min(width, x + 1.4 * w))
    y1 = int(min(height, y + 1.3 * h))
    y2 = int(min(height, y + 3.0 * h))

    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) * (y2 - y1) < MIN_TORSO_PIXELS:
        return None
    return x1, y1, x2, y2


def _skin_mask(region_rgb):
    ycrcb = cv2.cvtColor(region_rgb, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    return (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)


def name_colour(rgb):
    pixel = np.uint8([[list(rgb)]])
    hue, sat, val = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0]
    hue = int(hue) * 2
    sat = int(sat)
    val = int(val)

    if val < 50:
        return "black"
    if sat < 40:
        if val > 200:
            return "white"
        if val > 120:
            return "grey"
        return "black"
    if hue < 15 or hue >= 345:
        return "brown" if val < 130 else "red"
    if hue < 50 and sat < 110 and val >= 150:
        return "beige"
    if hue < 40:
        return "brown" if val < 140 else "orange"
    if hue < 70:
        return "yellow"
    if hue < 170:
        return "green"
    if hue < 200:
        return "cyan"
    if hue < 260:
        return "blue"
    if hue < 290:
        return "purple"
    if hue < 345:
        return "pink"
    return "red"


def predict_dress_colour(image_rgb, face_box=None):
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image array of shape (H, W, 3).")

    face_box = face_box or detect_face(image_rgb)
    if face_box is None:
        return {"colour": None, "reason": "No face detected", "region": None}

    region_box = torso_region(image_rgb.shape, face_box)
    if region_box is None:
        return {
            "colour": None,
            "reason": "Clothing not visible below the face",
            "region": None,
        }

    x1, y1, x2, y2 = region_box
    region = image_rgb[y1:y2, x1:x2]

    keep = ~_skin_mask(region)
    skin_like_clothing = float(keep.mean()) < SKIN_FALLBACK_RATIO

    if skin_like_clothing:
        pixels = region.reshape(-1, 3).astype(np.float32)
    else:
        pixels = region[keep].astype(np.float32)

    if len(pixels) < MIN_TORSO_PIXELS // 3:
        return {
            "colour": None,
            "reason": "Not enough clothing pixels",
            "region": region_box,
        }

    k = min(CLUSTERS, len(pixels))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    cv2.setRNGSeed(42)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(labels.flatten(), minlength=k)

    share_by_name = {}
    rgb_by_name = {}
    for index in range(k):
        rgb = tuple(int(c) for c in np.clip(centers[index], 0, 255))
        name = name_colour(rgb)
        share_by_name[name] = share_by_name.get(name, 0) + int(counts[index])
        if name not in rgb_by_name or counts[index] > counts[rgb_by_name[name][1]]:
            rgb_by_name[name] = (rgb, index)

    best = max(share_by_name, key=share_by_name.get)

    return {
        "colour": best,
        "rgb": rgb_by_name[best][0],
        "pixel_share": float(share_by_name[best] / counts.sum()),
        "skin_like_region": skin_like_clothing,
        "region": region_box,
        "face": face_box,
    }
