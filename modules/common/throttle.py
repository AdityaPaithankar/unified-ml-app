import threading
import time


class Throttle:
    def __init__(self, interval, clock=time.monotonic):
        self.interval = interval
        self.clock = clock
        self._last = None
        self._lock = threading.Lock()

    def ready(self):
        now = self.clock()
        with self._lock:
            if self._last is None or now - self._last >= self.interval:
                self._last = now
                return True
        return False
