import cv2
import numpy as np

from modules.car_colour.colour import classify_colours
from modules.car_colour.detectors import load_detector

BLUE_CAR_RECTANGLE = "red"
OTHER_CAR_RECTANGLE = "blue"
RECTANGLE_RGB = {"red": (255, 0, 0), "blue": (0, 0, 255)}
PERSON_RGB = (0, 200, 0)
VIDEO_MAX_FRAMES = 20


def rectangle_for(colour):
    return BLUE_CAR_RECTANGLE if colour == "blue" else OTHER_CAR_RECTANGLE


def analyse(image_rgb, detector=None, runner=None):
    detector = detector or load_detector()
    detections = detector.detect(image_rgb)
    cars = [d for d in detections if d["label"] == "car"]
    people = [d for d in detections if d["label"] == "person"]

    colours = classify_colours(image_rgb, [c["box"] for c in cars], runner) if cars else []
    car_rows = []
    for car, colour in zip(cars, colours):
        car_rows.append({
            "box": car["box"], "detection_score": car["score"], **colour,
            "rectangle": rectangle_for(colour["colour"]),
        })

    return {
        "detector": detector.name,
        "car_count": len(car_rows),
        "people_count": len(people),
        "cars": car_rows,
        "people": [{"box": p["box"], "detection_score": p["score"]} for p in people],
    }


def draw(image_rgb, result):
    canvas = image_rgb.copy()
    thickness = max(2, round(min(canvas.shape[:2]) / 250))
    for car in result["cars"]:
        x1, y1, x2, y2 = car["box"]
        rgb = RECTANGLE_RGB[car["rectangle"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), rgb, thickness)
        label = f"{car['colour'] or '?'} {car['confidence']:.2f}"
        cv2.putText(canvas, label, (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb, 2)
    for person in result["people"]:
        x1, y1, x2, y2 = person["box"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), PERSON_RGB, max(1, thickness - 1))
    return canvas


def analyse_video(path, detector=None, runner=None, max_frames=VIDEO_MAX_FRAMES, progress=None):
    detector = detector or load_detector()
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open the video file.")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    wanted = set(np.linspace(0, max(total - 1, 0), max_frames).astype(int)) if total else None
    frames, best, index = [], None, 0
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if wanted is None or index in wanted:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = analyse(rgb, detector, runner)
                result["frame_index"] = index
                frames.append(result)
                if best is None or result["car_count"] > best[1]["car_count"]:
                    best = (rgb, result)
                if progress:
                    progress(len(frames) / max_frames)
            index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError("The video contains no readable frames.")
    return {
        "frames_analysed": len(frames),
        "max_cars_in_a_frame": max(f["car_count"] for f in frames),
        "max_people_in_a_frame": max(f["people_count"] for f in frames),
        "per_frame": [
            {"frame_index": f["frame_index"], "cars": f["car_count"], "people": f["people_count"]}
            for f in frames
        ],
        "busiest_frame": draw(best[0], best[1]),
        "busiest_frame_result": best[1],
    }
