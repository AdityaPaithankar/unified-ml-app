import streamlit as st

from ui.common import VIDEO_TYPES, image_input, remove_quietly, require_worker, save_upload

VIDEO_STATE = "car_video_result"
DETECTORS = {
    "Automatic (YOLOv8 if available)": "auto",
    "YOLOv8 (pretrained)": "yolo",
    "Custom KITTI detector (own model)": "custom",
}


def _car_table(result):
    return [
        {
            "Car": index,
            "Colour": car["colour"] or "unknown",
            "Colour confidence": car["confidence"],
            "Rectangle": car["rectangle"].upper(),
            "Reliable": "yes" if car["reliable"] else "low",
        }
        for index, car in enumerate(result["cars"], start=1)
    ]


def _show_result(result, annotated):
    st.image(annotated)
    left, middle, right = st.columns(3)
    left.metric("Cars", result["car_count"])
    middle.metric("People", result["people_count"])
    right.metric("Detector", result["detector"])
    if result["cars"]:
        st.table(_car_table(result))
    st.caption("Blue cars are outlined in RED. All other cars are outlined in BLUE. People are green.")


def _image_mode(detector):
    from modules.car_colour import predictor

    image = image_input("car")
    if image is None:
        st.info("Upload a photo or take a webcam snapshot to begin.")
        return
    with st.spinner("Detecting cars..."):
        result = predictor.analyse(image, detector)
    _show_result(result, predictor.draw(image, result))


def _video_mode(detector):
    from modules.car_colour import predictor

    uploaded = st.file_uploader("Video file", type=VIDEO_TYPES, key="car_video")
    if uploaded and st.button("Analyse video", type="primary"):
        path = save_upload(uploaded)
        progress = st.progress(0.0, text="Analysing video...")
        try:
            st.session_state[VIDEO_STATE] = {
                "name": uploaded.name,
                "summary": predictor.analyse_video(path, detector, progress=progress.progress),
            }
        finally:
            remove_quietly(path)
            progress.empty()

    saved = st.session_state.get(VIDEO_STATE)
    if saved:
        summary = saved["summary"]
        st.subheader(f"Results for {saved['name']}")
        st.caption(f"{summary['frames_analysed']} frames sampled evenly across the video.")
        left, right = st.columns(2)
        left.metric("Most cars in one frame", summary["max_cars_in_a_frame"])
        right.metric("Most people in one frame", summary["max_people_in_a_frame"])
        st.markdown("**Busiest frame**")
        _show_result(summary["busiest_frame_result"], summary["busiest_frame"])
        st.dataframe(summary["per_frame"], hide_index=True)


def render():
    st.header("Task 5: Car colour, car count & people count")
    st.caption("Detects cars and people, names each car's colour, and counts both.")
    if not require_worker():
        return

    from modules.car_colour.detectors import load_detector

    choice = st.selectbox("Detector", list(DETECTORS))
    detector = load_detector(DETECTORS[choice])
    mode = st.radio("Source", ["Image / webcam snapshot", "Video file"], horizontal=True)
    _image_mode(detector) if mode == "Image / webcam snapshot" else _video_mode(detector)
