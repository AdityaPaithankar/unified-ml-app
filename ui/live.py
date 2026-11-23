import os

import av
import requests
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

_STUN_ONLY = [{"urls": ["stun:stun.l.google.com:19302"]}]


@st.cache_data(ttl=3600, show_spinner=False)
def _ice_servers():
    """STUN plus a TURN relay from Metered. Keys come from environment variables,
    never from the repository. Falls back to STUN only when they are not set."""
    app = os.environ.get("METERED_APP")
    key = os.environ.get("METERED_API_KEY")
    if not app or not key:
        return _STUN_ONLY
    try:
        reply = requests.get(
            f"https://{app}.metered.live/api/v1/turn/credentials",
            params={"apiKey": key},
            timeout=10,
        )
        reply.raise_for_status()
        servers = reply.json()
        return servers or _STUN_ONLY
    except Exception:
        return _STUN_ONLY


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
        rtc_configuration={"iceServers": _ice_servers()},
        async_processing=True,
    )
    processor = context.video_processor
    if processor is not None and processor.error:
        st.error(f"Live analysis stopped: {processor.error}")
    return context
