import streamlit as st

AUDIO_TYPES = ["wav", "mp3", "flac", "ogg", "m4a"]


def render():
    st.header("Task 3: Voice age & emotion")
    st.caption(
        "Accepts male voices only. For a male voice aged above 60 the app reports Senior Citizen "
        "and the speaker's emotion; for 60 or younger it reports age only."
    )
    uploaded = st.file_uploader("Audio file", type=AUDIO_TYPES, key="voice_upload")
    recorded = st.audio_input("Or record from the microphone", key="voice_record")
    audio = recorded or uploaded
    if audio is None:
        st.info("Upload or record a short speech clip to begin.")
        return

    st.audio(audio)
    from modules.voice_detection import predictor

    with st.spinner("Analysing voice (the first run loads the models and takes longer)..."):
        result = predictor.predict_voice(audio)

    if result["status"] == "rejected":
        st.error(result["message"])
        return

    left, middle, right = st.columns(3)
    left.metric("Estimated age", result["age"])
    middle.metric("Senior citizen", "Yes" if result["senior_citizen"] else "No")
    if result["senior_citizen"]:
        right.metric("Emotion", result["emotion"], f"confidence {result['emotion_confidence']}")
    for warning in result["warnings"]:
        st.warning(warning)
    with st.expander("Details"):
        st.json({k: result[k] for k in ("detected_gender", "gender_probabilities", "duration_seconds")})
