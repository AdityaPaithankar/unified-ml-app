import statistics
from dataclasses import dataclass, field

MATCH_IOU = 0.2
MAX_GAP_SECONDS = 2.0
MIN_HITS = 3
SENIOR_AGE = 60


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter_w = max(0, min(ax2, bx2) - max(a[0], b[0]))
    inter_h = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def centre_inside(a, b):
    cx, cy = a[0] + a[2] / 2, a[1] + a[3] / 2
    return b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]


@dataclass
class Track:
    track_id: int
    box: tuple
    first_seen: float
    last_seen: float
    hits: int = 1
    ages: list = field(default_factory=list)
    female_probs: list = field(default_factory=list)
    frames_since_sample: int = 10**6

    @property
    def confirmed(self):
        return self.hits >= MIN_HITS

    @property
    def sample_count(self):
        return len(self.ages)

    @property
    def age(self):
        return round(statistics.median(self.ages), 1) if self.ages else None

    @property
    def gender(self):
        if not self.female_probs:
            return None
        return "Female" if sum(self.female_probs) / len(self.female_probs) > 0.5 else "Male"

    @property
    def senior(self):
        return self.age is not None and self.age > SENIOR_AGE

    def add_sample(self, age, female_prob):
        self.ages.append(float(age))
        self.female_probs.append(float(female_prob))
        self.frames_since_sample = 0


class Tracker:
    def __init__(self, max_gap=MAX_GAP_SECONDS):
        self.max_gap = max_gap
        self._tracks = {}
        self._next_id = 1

    def update(self, boxes, timestamp):
        """Match boxes (x, y, w, h) to tracks. Returns (matched, closed)."""
        pairs = []
        for index, box in enumerate(boxes):
            for track in self._tracks.values():
                score = iou(box, track.box)
                if score < MATCH_IOU and centre_inside(box, track.box):
                    score = MATCH_IOU
                if score >= MATCH_IOU:
                    pairs.append((score, index, track.track_id))
        pairs.sort(reverse=True)

        used_boxes, used_tracks, matched = set(), set(), {}
        for _, index, track_id in pairs:
            if index in used_boxes or track_id in used_tracks:
                continue
            used_boxes.add(index)
            used_tracks.add(track_id)
            track = self._tracks[track_id]
            track.box = tuple(boxes[index])
            track.last_seen = timestamp
            track.hits += 1
            track.frames_since_sample += 1
            matched[index] = track

        for index, box in enumerate(boxes):
            if index in used_boxes:
                continue
            track = Track(self._next_id, tuple(box), timestamp, timestamp)
            self._tracks[track.track_id] = track
            self._next_id += 1
            matched[index] = track

        closed = [t for t in self._tracks.values() if timestamp - t.last_seen > self.max_gap]
        for track in closed:
            del self._tracks[track.track_id]
        return matched, closed

    def close_all(self):
        closed = list(self._tracks.values())
        self._tracks.clear()
        return closed
