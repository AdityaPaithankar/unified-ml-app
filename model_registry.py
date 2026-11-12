from modules.common.model_download import get_model_root

ROOT = get_model_root()

MODELS = {
    "gender_age": {
        "runtime": "keras3",
        "path": ROOT / "GEnder -Age" / "final_model.keras",
        "input_size": (128, 128),
        "outputs": ["gender", "age"],
    },
    "long_hair": {
        "runtime": "keras3",
        "path": ROOT / "Long-Hair" / "hair_model_best.keras",
        "input_size": (160, 160),
        "threshold": 0.70,
        "outputs": ["binary_hair_class"],
    },
    "senior_citizen": {
        "runtime": "keras3",
        "path": ROOT / "Senior-citizen" / "best_age_gender_model.keras",
        "input_size": (128, 128),
        "outputs": ["age", "gender"],
    },
    "car_detector": {
        "runtime": "tf210",
        "path": ROOT / "Car-colour" / "best_detector.keras",
        "input_size": (416, 416),
        "outputs": ["detections"],
    },
    "car_colour": {
        "runtime": "tf210",
        "path": ROOT / "Car-colour" / "colour_classifier" / "best_colour_model.keras",
        "class_names": ROOT / "Car-colour" / "colour_classifier" / "class_names.json",
        "input_size": (128, 128),
        "outputs": ["colour_probabilities"],
    },
    "nationality_age": {
        "runtime": "tf210",
        "path": ROOT / "Nationality" / "age" / "best_age_model.keras",
        "input_size": (160, 160),
        "outputs": ["predicted_age"],
    },
    "nationality_category": {
        "runtime": "tf210",
        "path": ROOT / "Nationality" / "category" / "best_category_model.keras",
        "class_names": ROOT / "Nationality" / "category" / "class_names.json",
        "input_size": (160, 160),
        "outputs": ["category_probabilities"],
    },
    "emotion": {
        "runtime": "tf210",
        "path": ROOT / "Nationality" / "emotion" / "best_emotion_model.keras",
        "class_names": ROOT / "Nationality" / "emotion" / "class_names.json",
        "input_size": (96, 96),
        "outputs": ["emotion_probabilities"],
    },
    "sign_language": {
        "runtime": "tf210",
        "path": ROOT / "Sign Language" / "isl_mobilenetv2.keras",
        "class_names": ROOT / "Sign Language" / "class_names.json",
        "input_size": (160, 160),
        "outputs": ["class_probabilities"],
    },
}
