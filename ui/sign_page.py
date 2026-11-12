from datetime import datetime

import streamlit as st

from ui.common import VIDEO_TYPES, image_input, remove_quietly, require_worker, save_upload


def _result_view(result):
    if result["status"] == "closed":
        st.warning(result["message"])
        return
    if result["status"] == "uncertain":
        st.warning("Low confidence. The closest match is shown below.")
    left, right = st.columns(2)
    left.metric("Word", result["word"].title())
    right.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
    st.bar_chart({row["word"]: row["confidence"] for row in result["top"]}, horizontal=True)
    if "frames_used" in result:
        st.caption(f"Averaged over {result['frames_used']} frames.")


def _live_mode():
    from modules.sign_language import predictor
    from modules.sign_language.live import LiveSignStream

    try:
        from ui.live import live_camera
    except ImportError:
        st.error("Live camera needs the streamlit-webrtc package. Use a snapshot or video file instead.")
        return
    st.caption("Press Start, allow the camera, and hold a sign in front of it. The prediction appears on the video.")
    live_camera("sign_live", lambda: LiveSignStream(predictor.predict))


def render():
    from modules.sign_language import predictor

    st.header("Task 4: Sign language recognition")
    first, last = predictor.window_label()
    zone = predictor.WINDOW_ZONE
    now = datetime.now(zone)
    st.caption(f"Recognises 8 words: {', '.join(predictor.CLASS_NAMES)}.")
    st.info(f"Available daily from {first} to {last} ({zone.tzname(None)}). Current time: {now:%I:%M %p}.")

    if not require_worker():
        return

    source = st.radio("Source", ["Live camera", "Image / webcam snapshot", "Video file"], horizontal=True)
    if source == "Live camera":
        _live_mode()
        return
    if source == "Image / webcam snapshot":
        image = image_input("sign")
        if image is None:
            st.info("Upload a photo or take a webcam snapshot of the sign.")
            return
        st.image(image, width=360)
        with st.spinner("Recognising..."):
            _result_view(predictor.predict(image))
        return

    uploaded = st.file_uploader("Video file", type=VIDEO_TYPES, key="sign_video")
    if uploaded is None:
        return
    path = save_upload(uploaded)
    try:
        frames = predictor.sample_video_frames(path)
        with st.spinner("Recognising..."):
            _result_view(predictor.predict_frames(frames))
    finally:
        remove_quietly(path)
