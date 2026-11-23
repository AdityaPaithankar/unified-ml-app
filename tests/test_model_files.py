"""Optional: confirm every registered model file really exists.

Needs the real registry, so it only runs with USE_REAL_MODELS=1 (and HF_TOKEN set, because
the weights are stored in a private Hugging Face repository).
    PowerShell:  $env:USE_REAL_MODELS="1"; pytest tests/test_model_files.py
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("USE_REAL_MODELS") != "1",
    reason="Set USE_REAL_MODELS=1 to check the real model files.",
)


def test_every_registered_model_file_exists():
    from model_registry import MODELS

    missing = []
    for name, spec in MODELS.items():
        for key in ("path", "class_names"):
            if key in spec and not spec[key].exists():
                missing.append(f"{name}.{key}: {spec[key]}")
    assert not missing, "Missing model files:\n" + "\n".join(missing)


def test_every_model_declares_a_known_runtime():
    from model_registry import MODELS

    for name, spec in MODELS.items():
        assert spec["runtime"] in {"keras3", "tf210"}, name
