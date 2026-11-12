import streamlit as st

from ui.common import worker_is_configured

TASKS = [
    ("Base model", "Gender & age from a face photo", "Keras 3"),
    ("Task 1", "Long-hair rule for ages 20–30, base model otherwise", "Keras 3"),
    ("Task 2", "Senior-citizen detection in video or webcam, with a visit log", "Keras 3"),
    ("Task 3", "Voice age & emotion (male voices only)", "PyTorch"),
    ("Task 4", "Sign language word recognition (time-window restricted)", "TensorFlow 2.10"),
    ("Task 5", "Car detection, colour, and people count", "TensorFlow 2.10 + YOLO"),
    ("Task 6", "Nationality category with age, emotion and dress colour", "TensorFlow 2.10"),
]


def render():
    st.title("Unified AI Detection Suite")
    st.caption("Seven trained models in one application. Choose a page in the sidebar.")

    rows = [{"Page": name, "What it does": what, "Runtime": runtime} for name, what, runtime in TASKS]
    st.table(rows)

    if worker_is_configured():
        st.success("TensorFlow 2.10 environment: configured.")
    else:
        st.warning(
            "TensorFlow 2.10 environment: not configured. Tasks 4, 5 and 6 need it. "
            "Set it in the sidebar under Settings."
        )
    st.caption(
        "Each task is independent. A failure in one task shows an error on its own page "
        "and does not affect the others. Models load only when a page first needs them."
    )
