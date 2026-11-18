import importlib
import os
import tempfile
import traceback
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "webm"]


def to_rgb(file):
    image = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.asarray(image)


def image_input(key, camera=True, label="Image"):
    """Upload or webcam snapshot. Returns an RGB array or None."""
    if not camera:
        file = st.file_uploader(label, type=IMAGE_TYPES, key=f"{key}_upload")
        return to_rgb(file) if file else None

    upload_tab, camera_tab = st.tabs(["Upload", "Webcam snapshot"])
    with upload_tab:
        file = st.file_uploader(label, type=IMAGE_TYPES, key=f"{key}_upload")
    with camera_tab:
        snapshot = st.camera_input("Take a photo", key=f"{key}_camera")
    chosen = snapshot or file
    return to_rgb(chosen) if chosen else None


def save_upload(uploaded):
    suffix = Path(uploaded.name).suffix or ".mp4"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    handle.write(uploaded.getbuffer())
    handle.close()
    return handle.name


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def configure_worker(python_path=None):
    python_path = (
        python_path
        or os.environ.get("TF210_PYTHON")
        or ""
    ).strip()

    if python_path:
        os.environ["TF210_PYTHON"] = python_path
        from modules.common import worker_client

        worker_client._client.python_path = python_path
def worker_is_configured():
    path = os.environ.get("TF210_PYTHON", "")
    return bool(path) and Path(path).is_file()


def require_worker():
    if worker_is_configured():
        return True
    st.warning(
        "This task needs the TensorFlow 2.10 environment. Enter the path to its python.exe "
        "in the sidebar (Settings), or set the TF210_PYTHON environment variable."
    )
    return False


def show_error(error):
    from modules.common.worker_client import WorkerError

    if isinstance(error, WorkerError):
        st.error(f"Model worker problem: {error}")
    elif isinstance(error, ValueError):
        st.error(str(error))
    elif isinstance(error, (ImportError, FileNotFoundError)):
        st.error(f"A required component is missing: {error}")
    else:
        st.error(f"This task could not be completed ({type(error).__name__}).")
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(error)))


def run_page(module_name):
    """Import and render one page. A failure here never affects the other pages."""
    try:
        importlib.import_module(module_name).render()
    except Exception as error:
        show_error(error)
