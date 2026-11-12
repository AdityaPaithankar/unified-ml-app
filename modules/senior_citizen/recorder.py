import csv
from pathlib import Path

COLUMNS = [
    "visit_id", "first_seen", "last_seen", "duration_seconds",
    "age", "gender", "senior_citizen", "samples", "source",
]


class VisitRecorder:
    def __init__(self, csv_path, source=""):
        self.path = Path(csv_path)
        self.source = source
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._next_id = self._read_next_id()
        if not self.path.exists() or self.path.stat().st_size == 0:
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(COLUMNS)

    @property
    def next_id(self):
        return self._next_id

    def _read_next_id(self):
        if not self.path.exists():
            return 1
        with self.path.open(newline="", encoding="utf-8") as handle:
            ids = [int(r["visit_id"]) for r in csv.DictReader(handle) if r.get("visit_id", "").isdigit()]
        return max(ids, default=0) + 1

    def write(self, first_seen, last_seen, age, gender, senior, samples):
        row = [
            self._next_id,
            first_seen.strftime("%Y-%m-%d %H:%M:%S"),
            last_seen.strftime("%Y-%m-%d %H:%M:%S"),
            round((last_seen - first_seen).total_seconds(), 1),
            age, gender, "Yes" if senior else "No", samples, self.source,
        ]
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(row)
        self._next_id += 1
        return row
