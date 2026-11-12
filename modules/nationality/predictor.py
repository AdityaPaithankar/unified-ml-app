import json

import cv2
import numpy as np

from model_registry import MODELS
from modules.common.face import square_crop
from modules.nationality import dress_colour
from modules.senior_citizen.detector import load_detector

CATEGORY_NAMES = json.loads(MODELS["nationality_category"]["class_names"].read_text(encoding="utf-8"))
EMOTION_NAMES = json.loads(MODELS["emotion"]["class_names"].read_text(encoding="utf-8"))

CATEGORY_SIZE = MODELS["nationality_category"]["input_size"][0]
AGE_SIZE = MODELS["nationality_age"]["input_size"][0]
EMOTION_SIZE = MODELS["emotion"]["input_size"][0]
EMOTION_SOURCE_SIZE = 48

CATEGORY_CROP_SCALE = 1.3
AGE_CROP_SCALE = 1.1
EMOTION_CROP_SCALE = 1.0

UNCERTAIN_BELOW = 0.50

RULES = {
    "Indian": {"label": "Indian", "outputs": ["age", "dress_colour", "emotion"]},
    "White": {"label": "United States", "outputs": ["age", "emotion"]},
    "Black": {"label": "African", "outputs": ["emotion", "dress_colour"]},
    "Asian": {"label": "Others", "outputs": ["nationality", "emotion"]},
    "Others": {"label": "Others", "outputs": ["nationality", "emotion"]},
}

DISCLAIMER = (
    "Categories come from the UTKFace dataset labels used for training. They are not a "
    "determination of anyone's real nationality or ethnicity."
)

NO_FACE_MESSAGE = "No face detected. Upload a clear photo with a visible face."


def _default_runner(model_key, batch):
    from modules.common.worker_client import predict

    return predict(model_key, batch)


def _scaled_crop(image_rgb, box, scale, size):
    x, y, w, h = box
    crop = square_crop(image_rgb, x + w / 2, y + h / 2, max(w, h) * scale)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def mobilenet_input(crop):
    return crop / 127.5 - 1.0


def emotion_input(image_rgb, box):
    crop = _scaled_crop(image_rgb, box, EMOTION_CROP_SCALE, EMOTION_SOURCE_SIZE * 2)
    grey = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (EMOTION_SOURCE_SIZE, EMOTION_SOURCE_SIZE), interpolation=cv2.INTER_AREA)
    large = cv2.resize(small.astype(np.float32), (EMOTION_SIZE, EMOTION_SIZE), interpolation=cv2.INTER_LINEAR)
    return mobilenet_input(np.repeat(large[..., None], 3, axis=-1))


def _top(probabilities, names, count=2):
    order = np.argsort(probabilities)[::-1][:count]
    return [{"label": names[i], "confidence": round(float(probabilities[i]), 3)} for i in order]


def classify_category(image_rgb, box, runner):
    batch = mobilenet_input(_scaled_crop(image_rgb, box, CATEGORY_CROP_SCALE, CATEGORY_SIZE))[None]
    return np.asarray(runner("nationality_category", batch))[0]


def estimate_age(image_rgb, box, runner):
    batch = mobilenet_input(_scaled_crop(image_rgb, box, AGE_CROP_SCALE, AGE_SIZE))[None]
    value = float(np.asarray(runner("nationality_age", batch)).ravel()[0])
    return int(np.clip(np.floor(value + 0.5), 1, 100))


def classify_emotion(image_rgb, box, runner):
    probabilities = np.asarray(runner("emotion", emotion_input(image_rgb, box)[None]))[0]
    return _top(probabilities, EMOTION_NAMES, 1)[0]


def build_result(category_probabilities, compute_age, compute_emotion, compute_dress):
    top = _top(category_probabilities, CATEGORY_NAMES, 2)
    best = top[0]
    result = {"status": "ok", "category": best["label"], "category_confidence": best["confidence"],
              "top_categories": top, "disclaimer": DISCLAIMER}

    if best["confidence"] < UNCERTAIN_BELOW:
        result.update(status="uncertain", category="Uncertain", rule=None,
                      message="The category could not be determined with enough confidence.",
                      emotion=compute_emotion())
        return result

    rule = RULES[best["label"]]
    result["rule"] = rule["label"]
    for name in rule["outputs"]:
        if name == "age":
            result["age"] = compute_age()
        elif name == "emotion":
            result["emotion"] = compute_emotion()
        elif name == "dress_colour":
            result["dress_colour"] = compute_dress()
        elif name == "nationality":
            result["nationality"] = best["label"]
    return result


def predict(image_rgb, detector=None, runner=None):
    runner = runner or _default_runner
    detector = detector or load_detector()

    faces = [tuple(d[:4]) for d in detector.detect(image_rgb)]
    if not faces:
        return {"status": "no_face", "message": NO_FACE_MESSAGE}
    box = max(faces, key=lambda f: f[2] * f[3])

    result = build_result(
        classify_category(image_rgb, box, runner),
        lambda: estimate_age(image_rgb, box, runner),
        lambda: classify_emotion(image_rgb, box, runner),
        lambda: dress_colour.predict_dress_colour(image_rgb, face_box=box),
    )
    result["faces_found"] = len(faces)
    result["face_box"] = box
    return result
