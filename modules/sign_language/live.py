import time

import cv2

from modules.common.throttle import Throttle

PREDICT_INTERVAL = 0.6


def overlay_text(result):
    if result["status"] == "closed":
        return "Outside the allowed time window"
    text = f"{result['word'].title()}  {result['confidence'] * 100:.0f}%"
    return text if result["status"] == "ok" else f"Uncertain: {text}"


class LiveSignStream:
    def __init__(self, predict, interval=PREDICT_INTERVAL, clock=time.monotonic):
        self.predict = predict
        self.throttle = Throttle(interval, clock)
        self.text = "Show a sign to the camera"

    def process(self, image_rgb):
        if self.throttle.ready():
            self.text = overlay_text(self.predict(image_rgb))
        canvas = image_rgb.copy()
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(canvas, self.text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return canvas
