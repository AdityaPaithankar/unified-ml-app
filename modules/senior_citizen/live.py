import threading
import time

from modules.common.throttle import Throttle
from modules.senior_citizen.session import annotate

LIVE_SOURCE = "live webcam"
ANALYSIS_INTERVAL = 0.2


class LiveSeniorStream:
    def __init__(self, session, interval=ANALYSIS_INTERVAL, clock=time.monotonic):
        self.session = session
        self.clock = clock
        self.started = clock()
        self.throttle = Throttle(interval, clock)
        self.people = []
        self._lock = threading.Lock()

    def process(self, image_rgb):
        if self.throttle.ready():
            with self._lock:
                self.people = self.session.process_frame(image_rgb, self.clock() - self.started)
        return annotate(image_rgb, self.people)

    def finish(self):
        with self._lock:
            self.people = []
            return self.session.finish()
