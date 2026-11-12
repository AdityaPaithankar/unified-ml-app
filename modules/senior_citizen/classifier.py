import numpy as np

from model_registry import MODELS
from modules.common.face import resize_rgb, square_crop

INPUT_SIZE = MODELS["gender_age"]["input_size"][0]
FACE_CROP_SCALE = 1.0
FEMALE_SCORE_IS_HIGH = True

_model = {}


def _load():
    if "model" not in _model:
        import keras

        _model["model"] = keras.models.load_model(
            str(MODELS["gender_age"]["path"]), compile=False, safe_mode=True
        )
    return _model["model"]


class AgeGenderClassifier:
    def classify(self, image_rgb, boxes):
        """Return one (age, female_probability) per box using the base model."""
        if not boxes:
            return []

        batch = np.stack([
            resize_rgb(
                square_crop(image_rgb, x + w / 2, y + h / 2, max(w, h) * FACE_CROP_SCALE),
                INPUT_SIZE,
            ) / 255.0
            for x, y, w, h in boxes
        ])
        model = _load()
        output = model.predict(batch, verbose=0)
        if not isinstance(output, dict):
            output = dict(zip(model.output_names, output))

        genders = np.asarray(output["gender"]).reshape(len(boxes), -1)[:, 0]
        ages = np.asarray(output["age"]).reshape(len(boxes), -1)[:, 0]
        female = genders if FEMALE_SCORE_IS_HIGH else 1 - genders
        return [(float(a), float(f)) for a, f in zip(ages, female)]
