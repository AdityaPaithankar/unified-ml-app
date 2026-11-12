import streamlit as st

from ui.common import image_input, require_worker


def render():
    st.header("Task 6: Nationality category, age & emotion")
    st.warning(
        "Categories come from the UTKFace dataset labels used for training. They are not a "
        "determination of anyone's real nationality or ethnicity."
    )
    if not require_worker():
        return

    image = image_input("nationality")
    if image is None:
        st.info("Upload a photo or take a webcam snapshot of one person.")
        return

    from modules.nationality import predictor

    with st.spinner("Analysing..."):
        result = predictor.predict(image)

    st.image(image, width=360)
    if result["status"] == "no_face":
        st.warning(result["message"])
        return

    left, right = st.columns(2)
    left.metric("Dataset category", result["category"], f"confidence {result['category_confidence']}")
    right.metric("Rule applied", result["rule"] or "none")
    first, second = result["top_categories"]
    st.write(f"Top-2 categories: **{first['label']}** ({first['confidence']}), {second['label']} ({second['confidence']})")
    if result["status"] == "uncertain":
        st.info(result["message"])
    if result.get("faces_found", 1) > 1:
        st.caption("More than one face found. The largest face was used.")

    columns = st.columns(3)
    if "nationality" in result:
        columns[0].metric("Nationality category", result["nationality"])
    if "age" in result:
        columns[0].metric("Age", result["age"])
    if "emotion" in result:
        columns[1].metric("Emotion", result["emotion"]["label"], f"confidence {result['emotion']['confidence']}")
    if "dress_colour" in result:
        dress = result["dress_colour"]
        if dress.get("colour"):
            columns[2].metric("Dress colour", dress["colour"].title())
        else:
            columns[2].metric("Dress colour", "Not available")
            columns[2].caption(dress.get("reason", ""))
    st.caption(result["disclaimer"])
