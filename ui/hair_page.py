import streamlit as st

from ui.common import image_input


def render():
    st.header("Task 1: Long hair & gender")
    st.caption(
        "For ages 20–30, long hair means Female and short hair means Male. "
        "For all other ages the base gender model decides."
    )
    image = image_input("hair")
    if image is None:
        st.info("Upload a photo or take a webcam snapshot to begin.")
        return

    from modules.long_hair import predictor

    with st.spinner("Analysing..."):
        result = predictor.predict(image)

    st.image(image)
    left, middle, right = st.columns(3)
    left.metric("Gender", result["gender"])
    middle.metric("Estimated age", result["predicted_age"])
    right.metric("Rule applied", "Hair rule" if result["rule"] == "hair_rule" else "Base model")
    if result["hair"]:
        st.write(f"Hair: **{result['hair']}** (short-hair probability {result['hair_score']})")
    for warning in result["warnings"]:
        st.warning(warning)
    with st.expander("Details"):
        st.json({k: result[k] for k in ("base_model_gender", "base_model_female_prob", "hair_score", "faces_found")})
