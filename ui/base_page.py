import cv2
import streamlit as st

from ui.common import image_input


def _label(face):
    return f"{face['gender']}, {face['age']}"


def render():
    st.header("Base model: Gender & age")
    image = image_input("base")
    if image is None:
        st.info("Upload a photo or take a webcam snapshot to begin.")
        return

    from modules.senior_citizen.classifier import AgeGenderClassifier
    from modules.senior_citizen.detector import load_detector

    with st.spinner("Analysing..."):
        boxes = [tuple(d[:4]) for d in load_detector().detect(image)]
        results = AgeGenderClassifier().classify(image, boxes) if boxes else []

    if not boxes:
        st.warning("No face detected. Upload a clear photo with a visible face.")
        st.image(image)
        return

    canvas, rows = image.copy(), []
    for index, ((x, y, w, h), (age, female)) in enumerate(zip(boxes, results), start=1):
        gender = "Female" if female > 0.5 else "Male"
        age_years = int(age + 0.5)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (40, 160, 70), 2)
        cv2.putText(canvas, f"{gender}, {age_years}", (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 160, 70), 2)
        rows.append({"Face": index, "Gender": gender, "Female probability": round(female, 3), "Age": age_years})

    st.image(canvas)
    st.table(rows)
    st.caption("Age estimates from this model tend to run low for older people.")
