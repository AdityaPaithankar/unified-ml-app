import hashlib
import io
from datetime import datetime

import pandas as pd
import streamlit as st

from ui.common import OUTPUTS, VIDEO_TYPES, image_input, remove_quietly, save_upload

LOG_PATH = OUTPUTS / "senior_visits.csv"
PREVIEW_FRAMES = 4
VIDEO_STATE = "senior_video_result"
SNAPSHOT_STATE = "senior_snapshot_saved"


def _excel_bytes(frame):
    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return buffer.getvalue()


def _show_log(frame, key):
    if frame.empty:
        st.info("No visits were recorded.")
        return
    seniors = int((frame["senior_citizen"] == "Yes").sum())
    left, right = st.columns(2)
    left.metric("People recorded", len(frame))
    right.metric("Senior citizens (age above 60)", seniors)
    st.dataframe(frame, hide_index=True)
    csv_col, xlsx_col = st.columns(2)
    csv_col.download_button("Download CSV", frame.to_csv(index=False), "senior_visits.csv", "text/csv", key=f"{key}_csv")
    xlsx_col.download_button("Download Excel", _excel_bytes(frame), "senior_visits.xlsx", key=f"{key}_xlsx")


def _rows_since(first_id):
    frame = pd.read_csv(LOG_PATH)
    return frame[frame["visit_id"] >= first_id].reset_index(drop=True)


def _analyse_video(uploaded):
    from modules.senior_citizen.classifier import AgeGenderClassifier
    from modules.senior_citizen.detector import load_detector
    from modules.senior_citizen.recorder import VisitRecorder
    from modules.senior_citizen.session import SeniorCitizenSession, process_video

    path = save_upload(uploaded)
    previews, progress = [], st.progress(0.0, text="Analysing video...")
    detector = load_detector()
    recorder = VisitRecorder(LOG_PATH, source=uploaded.name)
    first_id = recorder.next_id
    session = SeniorCitizenSession(detector, AgeGenderClassifier(), recorder)

    def keep_preview(frame, seconds):
        if len(previews) < PREVIEW_FRAMES and int(seconds) >= len(previews) * 2:
            previews.append(frame)

    try:
        summary = process_video(path, session, progress=progress.progress, on_frame=keep_preview)
    finally:
        remove_quietly(path)
    progress.empty()

    st.session_state[VIDEO_STATE] = {
        "name": uploaded.name,
        "caption": f"Face detector: {detector.name}. Frames analysed: {summary['frames_analysed']}.",
        "previews": previews,
        "log": _rows_since(first_id),
    }


def _video_mode():
    uploaded = st.file_uploader("Video file", type=VIDEO_TYPES, key="senior_video")
    if uploaded and st.button("Analyse video", type="primary"):
        _analyse_video(uploaded)

    result = st.session_state.get(VIDEO_STATE)
    if result:
        st.subheader(f"Results for {result['name']}")
        st.caption(result["caption"])
        if result["previews"]:
            st.image(result["previews"], width=240)
        _show_log(result["log"], "video")


def _snapshot_mode():
    from modules.senior_citizen.classifier import AgeGenderClassifier
    from modules.senior_citizen.detector import load_detector
    from modules.senior_citizen.recorder import VisitRecorder
    from modules.senior_citizen.session import annotate

    image = image_input("senior_snap")
    if image is None:
        st.info("Take a webcam snapshot or upload a photo to begin.")
        return

    boxes = [tuple(d[:4]) for d in load_detector().detect(image)]
    if not boxes:
        st.warning("No face detected.")
        st.image(image)
        return

    people, rows = [], []
    for index, (box, (age, female)) in enumerate(zip(boxes, AgeGenderClassifier().classify(image, boxes)), start=1):
        senior = round(age, 1) > 60
        gender = "Female" if female > 0.5 else "Male"
        people.append({"track_id": index, "box": box, "age": age, "gender": gender, "senior_citizen": senior})
        rows.append({"Person": index, "Age": int(age + 0.5), "Gender": gender, "Senior citizen": "Yes" if senior else "No"})

    st.image(annotate(image, people))
    st.table(rows)

    digest = hashlib.sha1(image.tobytes()).hexdigest()
    saved = st.session_state.setdefault(SNAPSHOT_STATE, {})
    if digest in saved:
        st.success("This snapshot is already in the visit log.")
        _show_log(saved[digest], "snap")
    elif st.button("Save to visit log"):
        recorder = VisitRecorder(LOG_PATH, source="webcam snapshot")
        first_id, now = recorder.next_id, datetime.now()
        for person in people:
            recorder.write(now, now, int(person["age"] + 0.5), person["gender"], person["senior_citizen"], 1)
        saved[digest] = _rows_since(first_id)
        st.rerun()


def _live_mode():
    import pandas as pd

    from modules.senior_citizen.classifier import AgeGenderClassifier
    from modules.senior_citizen.detector import load_detector
    from modules.senior_citizen.live import LIVE_SOURCE, LiveSeniorStream
    from modules.senior_citizen.recorder import VisitRecorder
    from modules.senior_citizen.session import SeniorCitizenSession

    try:
        from ui.live import live_camera
    except ImportError:
        st.error("Live camera needs the streamlit-webrtc package. Use a snapshot or video file instead.")
        return

    def make_stream():
        session = SeniorCitizenSession(load_detector(), AgeGenderClassifier(), VisitRecorder(LOG_PATH, source=LIVE_SOURCE))
        return LiveSeniorStream(session)

    st.caption(
        "Press Start and allow the camera. Each person is logged once when they leave the view "
        "or when you press Stop. Use the button below to save the people currently visible."
    )
    context = live_camera("senior_live", make_stream)
    stream = context.video_processor.stream if context.video_processor else None
    if stream is not None and st.button("Save visits now"):
        stream.finish()
        st.success("Current visits were saved to the log.")

    if st.button("Show live-camera visit log"):
        if not LOG_PATH.exists():
            st.info("No visits were recorded yet.")
            return
        frame = pd.read_csv(LOG_PATH)
        _show_log(frame[frame["source"] == LIVE_SOURCE].reset_index(drop=True), "live")


def render():
    st.header("Task 2: Senior citizen detection")
    st.caption(
        "Finds every face, estimates age and gender, and flags anyone above 60 as a senior citizen. "
        "In videos each person is tracked and logged once, not once per frame."
    )
    mode = st.radio("Source", ["Live camera", "Video file", "Webcam snapshot / photo"], horizontal=True)
    if mode == "Live camera":
        _live_mode()
    elif mode == "Video file":
        _video_mode()
    else:
        _snapshot_mode()
    st.caption(f"Visit log file: {LOG_PATH}")
