"""Task 2: tracking, de-duplication and the senior-citizen rule."""
import pytest

from modules.senior_citizen.tracker import MIN_HITS, SENIOR_AGE, Tracker, Track, iou


def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_half_overlap():
    # intersection 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_iou_zero_area_boxes_do_not_divide_by_zero():
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_same_person_keeps_one_track_id_across_frames():
    tracker = Tracker()
    ids = set()
    for step in range(5):
        matched, _ = tracker.update([(100 + step * 3, 100, 50, 50)], timestamp=step * 0.1)
        ids.add(matched[0].track_id)
    assert len(ids) == 1


def test_two_people_get_two_tracks():
    tracker = Tracker()
    matched, _ = tracker.update([(0, 0, 40, 40), (300, 300, 40, 40)], timestamp=0.0)
    assert matched[0].track_id != matched[1].track_id


def test_track_confirmed_only_after_min_hits():
    tracker = Tracker()
    track = None
    for step in range(MIN_HITS):
        matched, _ = tracker.update([(10, 10, 40, 40)], timestamp=step * 0.1)
        track = matched[0]
        assert track.confirmed == (step + 1 >= MIN_HITS)
    assert track.confirmed


def test_track_closes_after_gap_and_not_before():
    tracker = Tracker(max_gap=2.0)
    tracker.update([(10, 10, 40, 40)], timestamp=0.0)
    _, closed = tracker.update([], timestamp=1.0)
    assert closed == []
    _, closed = tracker.update([], timestamp=3.0)
    assert len(closed) == 1


def test_returning_person_after_gap_is_a_new_visit():
    tracker = Tracker(max_gap=2.0)
    first, _ = tracker.update([(10, 10, 40, 40)], timestamp=0.0)
    tracker.update([], timestamp=5.0)  # first track closes
    second, _ = tracker.update([(10, 10, 40, 40)], timestamp=6.0)
    assert second[0].track_id != first[0].track_id


def test_close_all_returns_open_tracks_and_empties_tracker():
    tracker = Tracker()
    tracker.update([(0, 0, 40, 40), (300, 300, 40, 40)], timestamp=0.0)
    assert len(tracker.close_all()) == 2
    assert tracker.close_all() == []


def _track():
    return Track(track_id=1, box=(0, 0, 10, 10), first_seen=0.0, last_seen=0.0)


def test_senior_requires_age_strictly_above_60():
    assert SENIOR_AGE == 60
    over, exact = _track(), _track()
    for age in (61, 62, 63):
        over.add_sample(age, 0.2)
    for age in (60, 60, 60):
        exact.add_sample(age, 0.2)
    assert over.senior is True
    assert exact.senior is False


def test_age_uses_median_so_one_bad_frame_does_not_matter():
    track = _track()
    for age in (30, 31, 90):
        track.add_sample(age, 0.5)
    assert track.age == 31


def test_gender_is_average_of_samples():
    track = _track()
    for prob in (0.9, 0.8, 0.1):
        track.add_sample(40, prob)
    assert track.gender == "Female"
    male = _track()
    for prob in (0.1, 0.2, 0.9):
        male.add_sample(40, prob)
    assert male.gender == "Male"


def test_empty_track_has_no_age_gender_or_senior_flag():
    track = _track()
    assert track.age is None and track.gender is None and track.senior is False
