from datetime import datetime, timedelta

import cv2

from modules.senior_citizen.tracker import Tracker

MAX_SAMPLES = 12
SAMPLE_EVERY_FRAMES = 3
PROCESS_FPS = 5


class SeniorCitizenSession:
    def __init__(self, detector, classifier, recorder, start_time=None):
        self.detector = detector
        self.classifier = classifier
        self.recorder = recorder
        self.tracker = Tracker()
        self.start_time = start_time or datetime.now()
        self.visits_written = 0

    def _clock(self, seconds):
        return self.start_time + timedelta(seconds=seconds)

    def _record(self, tracks):
        for track in tracks:
            if not track.confirmed or not track.sample_count:
                continue
            self.recorder.write(
                self._clock(track.first_seen), self._clock(track.last_seen),
                int(track.age + 0.5), track.gender, track.senior, track.sample_count,
            )
            self.visits_written += 1

    def process_frame(self, image_rgb, timestamp):
        boxes = [tuple(d[:4]) for d in self.detector.detect(image_rgb)]
        matched, closed = self.tracker.update(boxes, timestamp)
        self._record(closed)

        due = [
            (i, t) for i, t in matched.items()
            if t.sample_count < MAX_SAMPLES and t.frames_since_sample >= SAMPLE_EVERY_FRAMES
        ]
        if due:
            results = self.classifier.classify(image_rgb, [boxes[i] for i, _ in due])
            for (_, track), (age, female) in zip(due, results):
                track.add_sample(age, female)

        return [
            {
                "track_id": t.track_id, "box": boxes[i], "age": t.age,
                "gender": t.gender, "senior_citizen": t.senior,
            }
            for i, t in sorted(matched.items()) if t.confirmed and t.sample_count
        ]

    def finish(self):
        self._record(self.tracker.close_all())
        return self.visits_written


def annotate(image_rgb, people):
    canvas = image_rgb.copy()
    for person in people:
        x, y, w, h = person["box"]
        senior = person["senior_citizen"]
        colour = (220, 40, 40) if senior else (40, 160, 70)
        label = f"#{person['track_id']} {person['gender']} {int(person['age'] + 0.5)}"
        if senior:
            label += " Senior"
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(canvas, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return canvas


def process_video(path, session, progress=None, on_frame=None):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open the video file.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps / PROCESS_FPS)))
    index = analysed = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                people = session.process_frame(rgb, index / fps)
                analysed += 1
                if on_frame:
                    on_frame(annotate(rgb, people), index / fps)
            index += 1
            if progress and total:
                progress(min(index / total, 1.0))
    finally:
        capture.release()

    if index == 0:
        raise ValueError("The video contains no readable frames.")
    return {"frames_read": index, "frames_analysed": analysed, "visits_recorded": session.finish()}
