"""Task 2: the visit log CSV (timestamps, no duplicate rows)."""
import csv
from datetime import datetime, timedelta

from modules.senior_citizen.recorder import COLUMNS, VisitRecorder


def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_new_file_gets_header_row(tmp_path):
    path = tmp_path / "visits.csv"
    VisitRecorder(path)
    assert path.read_text(encoding="utf-8").splitlines()[0].split(",") == COLUMNS


def test_one_visit_is_one_row_with_timestamps_and_duration(tmp_path):
    path = tmp_path / "visits.csv"
    start = datetime(2026, 11, 23, 10, 0, 0)
    recorder = VisitRecorder(path, source="camera")
    recorder.write(start, start + timedelta(seconds=90), 67.4, "Male", True, 12)

    rows = _rows(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["first_seen"] == "2026-11-23 10:00:00"
    assert row["last_seen"] == "2026-11-23 10:01:30"
    assert row["duration_seconds"] == "90.0"
    assert row["senior_citizen"] == "Yes"
    assert row["source"] == "camera"


def test_non_senior_is_marked_no(tmp_path):
    path = tmp_path / "visits.csv"
    now = datetime(2026, 11, 23, 10, 0, 0)
    VisitRecorder(path).write(now, now, 30, "Female", False, 3)
    assert _rows(path)[0]["senior_citizen"] == "No"


def test_visit_ids_increase_and_continue_after_restart(tmp_path):
    path = tmp_path / "visits.csv"
    now = datetime(2026, 11, 23, 10, 0, 0)
    first = VisitRecorder(path)
    first.write(now, now, 30, "Male", False, 3)
    first.write(now, now, 70, "Female", True, 5)

    second = VisitRecorder(path)  # simulates restarting the app
    assert second.next_id == 3
    second.write(now, now, 45, "Male", False, 4)

    assert [r["visit_id"] for r in _rows(path)] == ["1", "2", "3"]


def test_header_is_not_duplicated_on_reopen(tmp_path):
    path = tmp_path / "visits.csv"
    VisitRecorder(path)
    VisitRecorder(path)
    assert path.read_text(encoding="utf-8").count("visit_id") == 1


def test_creates_missing_folders(tmp_path):
    path = tmp_path / "deep" / "nested" / "visits.csv"
    VisitRecorder(path)
    assert path.exists()
