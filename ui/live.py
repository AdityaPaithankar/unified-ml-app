import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


class _Processor(VideoProcessorBase):
    def __init__(self, make_stream):
        self.error = None
        self.stream = None
        self._make_stream = make_stream

    def recv(self, frame):
        image = frame.to_ndarray(format="rgb24")
        try:
            if self.stream is None:
                self.stream = self._make_stream()
            image = self.stream.process(image)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        return av.VideoFrame.from_ndarray(image, format="rgb24")

    def on_ended(self):
        finish = getattr(self.stream, "finish", None)
        if finish:
            finish()


def live_camera(key, make_stream):
    context = webrtc_streamer(
        key=key,
        video_processor_factory=lambda: _Processor(make_stream),
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
        async_processing=True,
    )
    processor = context.video_processor
    if processor is not None and processor.error:
        st.error(f"Live analysis stopped: {processor.error}")
    return context
