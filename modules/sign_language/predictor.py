import json
from datetime import datetime, time, timedelta, timezone

import cv2
import numpy as np

from model_registry import MODELS

INPUT_SIZE = MODELS["sign_language"]["input_size"][0]
CLASS_NAMES = json.loads(MODELS["sign_language"]["class_names"].read_text(encoding="utf-8"))

WINDOW_START = time(18, 0)
WINDOW_END = time(22, 0)
INDIA_TIME = timezone(timedelta(hours=5, minutes=30), "IST")
WINDOW_ZONE = INDIA_TIME

MIN_CONFIDENCE = 0.50
TOP_K = 3
MAX_VIDEO_FRAMES = 15

CLOSED_MESSAGE = "Sign language detection is available between {start} and {end} ({zone})."


def _default_runner(model_key, batch):
    from modules.common.worker_client import predict

    return predict(model_key, batch)


def window_label(start=None, end=None):
    start, end = start or WINDOW_START, end or WINDOW_END
    return start.strftime("%I:%M %p").lstrip("0"), end.strftime("%I:%M %p").lstrip("0")


def is_within_window(now=None, start=None, end=None, zone=None):
    start, end, zone = start or WINDOW_START, end or WINDOW_END, zone or WINDOW_ZONE
    current = (now or datetime.now(zone)).astimezone(zone).time()
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def preprocess(image_rgb):
    resized = cv2.resize(image_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
    return resized.astype(np.float32) / 127.5 - 1.0


def _summarise(probabilities):
    order = np.argsort(probabilities)[::-1]
    top = [
        {"word": CLASS_NAMES[i], "confidence": round(float(probabilities[i]), 4)}
        for i in order[:TOP_K]
    ]
    best = top[0]
    status = "ok" if best["confidence"] >= MIN_CONFIDENCE else "uncertain"
    return {"status": status, "word": best["word"], "confidence": best["confidence"], "top": top}


def classify(images_rgb, runner=None):
    runner = runner or _default_runner
    batch = np.stack([preprocess(image) for image in images_rgb])
    probabilities = np.asarray(runner("sign_language", batch))
    return probabilities.reshape(len(images_rgb), -1)


def predict(image_rgb, now=None, runner=None, start=None, end=None, zone=None):
    if not is_within_window(now, start, end, zone):
        first, last = window_label(start, end)
        zone_name = (zone or WINDOW_ZONE).tzname(None) or "local time"
        return {"status": "closed", "message": CLOSED_MESSAGE.format(start=first, end=last, zone=zone_name)}
    return _summarise(classify([image_rgb], runner)[0])


def predict_frames(frames_rgb, now=None, runner=None, start=None, end=None, zone=None):
    if not frames_rgb:
        raise ValueError("No frames to analyse.")
    if not is_within_window(now, start, end, zone):
        return predict(frames_rgb[0], now, runner, start, end, zone)
    result = _summarise(classify(frames_rgb, runner).mean(axis=0))
    result["frames_used"] = len(frames_rgb)
    return result


def sample_video_frames(path, max_frames=MAX_VIDEO_FRAMES):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open the video file.")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    indices = set(np.linspace(0, max(total - 1, 0), max_frames).astype(int)) if total else None
    frames, index = [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if indices is None or index in indices:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
            if len(frames) >= max_frames:
                break
    finally:
        capture.release()
    if not frames:
        raise ValueError("The video contains no readable frames.")
    return frames
