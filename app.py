import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from ui.common import configure_worker, run_page  # noqa: E402

PAGES = {
    "Home": "ui.home_page",
    "Base: Gender & Age": "ui.base_page",
    "Task 1: Long Hair": "ui.hair_page",
    "Task 2: Senior Citizen": "ui.senior_page",
    "Task 3: Voice Age & Emotion": "ui.voice_page",
    "Task 4: Sign Language": "ui.sign_page",
    "Task 5: Car Colour & Counts": "ui.car_page",
    "Task 6: Nationality": "ui.nationality_page",
}

st.set_page_config(page_title="Unified AI Detection Suite", page_icon="🧠", layout="wide")

with st.sidebar:
    st.title("AI Detection Suite")
    choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
    with st.expander("Settings"):
        path = st.text_input(
            "TensorFlow 2.10 python.exe",
            value=os.environ.get("TF210_PYTHON", ""),
            help="Needed for Tasks 4, 5 and 6.",
        )
        configure_worker(path.strip().strip('"'))

run_page(PAGES[choice])
