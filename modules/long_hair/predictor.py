import cv2
import numpy as np

from model_registry import MODELS
from modules.common.face import detect_faces, resize_rgb, square_crop

HAIR_RULE_MIN_AGE = 20
HAIR_RULE_MAX_AGE = 30

GENDER_AGE_SIZE = MODELS["gender_age"]["input_size"][0]
HAIR_SIZE = MODELS["long_hair"]["input_size"][0]
HAIR_THRESHOLD = MODELS["long_hair"]["threshold"]

HAIR_SCORE_IS_SHORT_PROB = True
FEMALE_SCORE_IS_HIGH = True

FACE_CROP_SCALE = 1.0
HAIR_EXPAND_SIDE = 0.45
HAIR_EXPAND_TOP = 0.75
HAIR_EXPAND_BOTTOM = 1.8

_models = {}


def _load(name):
    if name not in _models:
        import keras

        _models[name] = keras.models.load_model(
            str(MODELS[name]["path"]), compile=False, safe_mode=True
        )
    return _models[name]


def decide_gender(age, base_female_prob, hair_score):
    rounded_age = int(np.floor(float(age) + 0.5))
    base_gender = "Female" if base_female_prob >= 0.5 else "Male"

    if not HAIR_RULE_MIN_AGE <= rounded_age <= HAIR_RULE_MAX_AGE:
        return {
            "gender": base_gender,
            "rule": "base_model",
            "hair": None,
        }

    short_score = hair_score if HAIR_SCORE_IS_SHORT_PROB else 1.0 - hair_score
    is_long = short_score < HAIR_THRESHOLD
    return {
        "gender": "Female" if is_long else "Male",
        "rule": "hair_rule",
        "hair": "Long" if is_long else "Short",
    }


def _hair_region(image_rgb, box):
    x, y, w, h = box
    height, width = image_rgb.shape[:2]
    x1 = max(0, x - int(w * HAIR_EXPAND_SIDE))
    y1 = max(0, y - int(h * HAIR_EXPAND_TOP))
    x2 = min(width, x + w + int(w * HAIR_EXPAND_SIDE))
    y2 = min(height, y + h + int(h * HAIR_EXPAND_BOTTOM))
    return image_rgb[y1:y2, x1:x2]


def _crops(image_rgb):
    faces = detect_faces(image_rgb)
    if not faces:
        return image_rgb, image_rgb, None, 0

    box = faces[0]
    x, y, w, h = box
    face_crop = square_crop(image_rgb, x + w / 2, y + h / 2, max(w, h) * FACE_CROP_SCALE)
    return face_crop, _hair_region(image_rgb, box), box, len(faces)


def predict(image_rgb):
    if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image array of shape (H, W, 3).")

    face_crop, head_crop, face_box, faces_found = _crops(image_rgb)

    gender_age = _load("gender_age")
    face_input = resize_rgb(face_crop, GENDER_AGE_SIZE)[None] / 255.0
    output = gender_age.predict(face_input, verbose=0)
    if not isinstance(output, dict):
        output = dict(zip(gender_age.output_names, output))
    gender_score = float(np.asarray(output["gender"]).flatten()[0])
    age = float(np.asarray(output["age"]).flatten()[0])
    base_female_prob = gender_score if FEMALE_SCORE_IS_HIGH else 1.0 - gender_score

    hair_model = _load("long_hair")
    hair_input = cv2.resize(head_crop, (HAIR_SIZE, HAIR_SIZE)).astype(np.float32)[None]
    hair_score = float(np.asarray(hair_model.predict(hair_input, verbose=0)).flatten()[0])

    decision = decide_gender(age, base_female_prob, hair_score)

    warnings = []
    if face_box is None:
        warnings.append(
            "No face detected, so the whole image was used. Treat this result as unreliable."
        )
    if faces_found > 1:
        warnings.append(f"{faces_found} faces found. Only the largest face was used.")

    return {
        "gender": decision["gender"],
        "rule": decision["rule"],
        "hair": decision["hair"],
        "predicted_age": round(age, 1),
        "base_model_gender": "Female" if base_female_prob >= 0.5 else "Male",
        "base_model_female_prob": round(base_female_prob, 3),
        "hair_score": round(hair_score, 3),
        "face_detected": face_box is not None,
        "faces_found": faces_found,
        "face_box": face_box,
        "warnings": warnings,
    }
