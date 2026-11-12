import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from modules.common.model_download import get_model_root

VOICE_DIR = get_model_root() / "VOICE"

SAMPLE_RATE = 16000
MIN_SECONDS = 1.0
MAX_SECONDS = 30.0
SILENCE_PEAK = 1e-4

SENIOR_AGE = 60
REJECT_MESSAGE = "Upload male voice."

GENDER_LABELS = ["Female", "Male", "Child"]
EMOTION_LABELS = {"neu": "Neutral", "ang": "Angry", "hap": "Happy", "sad": "Sad"}

_models = {}


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_models():
    if _models:
        return _models["processor"], _models["age_gender"], _models["emotion"]

    from speechbrain.inference.interfaces import foreign_class
    from transformers import Wav2Vec2Processor

    from modules.voice_detection.age_gender_model import AgeGenderModel

    with _working_directory(VOICE_DIR):
        processor = Wav2Vec2Processor.from_pretrained(
            str(VOICE_DIR), local_files_only=True
        )
        age_gender = AgeGenderModel.from_pretrained(
            str(VOICE_DIR), local_files_only=True, use_safetensors=True
        )
        age_gender.to("cpu")
        age_gender.eval()

        emotion = foreign_class(
            source=str(VOICE_DIR),
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": "cpu"},
        )

    _models.update(processor=processor, age_gender=age_gender, emotion=emotion)
    return processor, age_gender, emotion


def prepare_audio(audio):
    warnings = []
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("The audio file is empty or could not be read.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("The audio file contains invalid samples.")

    if float(np.max(np.abs(audio))) < SILENCE_PEAK:
        raise ValueError("The audio is silent. Upload a recording that contains speech.")

    if audio.size < int(MIN_SECONDS * SAMPLE_RATE):
        raise ValueError(
            f"The recording is too short. Upload at least {MIN_SECONDS:g} second of speech."
        )

    max_samples = int(MAX_SECONDS * SAMPLE_RATE)
    if audio.size > max_samples:
        audio = audio[:max_samples]
        warnings.append(f"Only the first {MAX_SECONDS:g} seconds were analysed.")

    return audio, warnings


def load_audio(source):
    import librosa

    try:
        audio, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True)
    except Exception as error:
        raise ValueError(
            "Could not read the audio file. Upload a valid WAV, MP3, FLAC or OGG recording."
        ) from error
    return prepare_audio(audio)


def predict_age_gender(audio):
    import torch

    processor, age_gender, _ = _load_models()
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")

    with torch.inference_mode():
        _, age_output, gender_output = age_gender(inputs.input_values)

    age = float(age_output.squeeze().item()) * 100
    probabilities = gender_output.squeeze().cpu().numpy()
    gender = GENDER_LABELS[int(np.argmax(probabilities))]
    return age, gender, probabilities


def predict_emotion(audio):
    import torch

    _, _, emotion = _load_models()
    waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    with torch.inference_mode():
        _, score, _, label = emotion.classify_batch(waveform, torch.tensor([1.0]))

    code = str(label[0]).lower()
    return EMOTION_LABELS.get(code, code), float(score[0])


def decide(gender, age, emotion_fn):
    if gender != "Male":
        return {"status": "rejected", "message": REJECT_MESSAGE}

    age = round(float(age), 1)
    result = {"status": "ok", "age": age, "senior_citizen": age > SENIOR_AGE}

    if result["senior_citizen"]:
        emotion, confidence = emotion_fn()
        result["emotion"] = emotion
        result["emotion_confidence"] = round(confidence, 3)

    return result


def predict_voice(source):
    audio, warnings = load_audio(source)
    age, gender, probabilities = predict_age_gender(audio)

    result = decide(gender, age, lambda: predict_emotion(audio))
    result["detected_gender"] = gender
    result["gender_probabilities"] = {
        label: round(float(p), 3) for label, p in zip(GENDER_LABELS, probabilities)
    }
    result["duration_seconds"] = round(audio.size / SAMPLE_RATE, 1)
    result["warnings"] = warnings
    return result
