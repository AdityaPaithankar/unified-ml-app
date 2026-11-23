"""Shared test setup.

The real model weights live in a private Hugging Face repository, so by default the
tests replace `model_registry` with a tiny stand-in. This lets every logic test run on
any machine, offline, in seconds, and never loads a model.

Set USE_REAL_MODELS=1 (with HF_TOKEN available) to test against the real registry.
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

USE_REAL_MODELS = os.environ.get("USE_REAL_MODELS") == "1"

SIGN_WORDS = ["book", "careful", "hungry", "i need help", "thirsty", "understand", "wait", "write"]
CAR_COLOURS = ["black", "blue", "red", "white"]


def _install_fake_registry():
    folder = Path(tempfile.mkdtemp(prefix="fake_models_"))
    sign_names = folder / "sign_names.json"
    colour_names = folder / "colour_names.json"
    sign_names.write_text(json.dumps(SIGN_WORDS), encoding="utf-8")
    colour_names.write_text(json.dumps(CAR_COLOURS), encoding="utf-8")

    fake = types.ModuleType("model_registry")
    fake.ROOT = folder
    fake.MODELS = {
        "sign_language": {"runtime": "tf210", "input_size": (160, 160), "class_names": sign_names},
        "car_colour": {"runtime": "tf210", "input_size": (128, 128), "class_names": colour_names},
    }
    sys.modules["model_registry"] = fake


if not USE_REAL_MODELS:
    _install_fake_registry()
