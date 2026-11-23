"""Task 4: the 6 PM - 10 PM IST window, word + confidence output, video sampling."""
from datetime import datetime, time, timezone

import numpy as np
import pytest

from modules.sign_language import predictor

UTC = timezone.utc
FRAME = np.zeros((240, 320, 3), dtype=np.uint8)


def ist(hour, minute=0, second=0):
    """A UTC timestamp that equals the given clock time in IST (UTC+05:30)."""
    total = hour * 3600 + minute * 60 + second - (5 * 3600 + 30 * 60)
    total %= 24 * 3600
    return datetime(2026, 11, 23, total // 3600, (total % 3600) // 60, total % 60, tzinfo=UTC)


def never_called(*_args, **_kwargs):
    raise AssertionError("The model must not run outside the time window.")


def probs(index, value=0.9, size=8):
    row = np.full(size, (1 - value) / (size - 1))
    row[index] = value
    return row


# ---- time window -----------------------------------------------------------------

@pytest.mark.parametrize(
    "now, expected",
    [
        (ist(17, 59, 59), False),
        (ist(18, 0, 0), True),
        (ist(20, 0), True),
        (ist(21, 59, 59), True),
        (ist(22, 0, 0), False),
        (ist(3, 0), False),
    ],
)
def test_window_is_6pm_inclusive_to_10pm_exclusive_ist(now, expected):
    assert predictor.is_within_window(now) is expected


def test_window_uses_ist_not_the_servers_timezone():
    # 18:00 IST is 12:30 UTC. A server clock in UTC must still get the IST answer.
    assert predictor.is_within_window(datetime(2026, 11, 23, 12, 30, tzinfo=UTC)) is True
    assert predictor.is_within_window(datetime(2026, 11, 23, 18, 0, tzinfo=UTC)) is False


def test_overnight_window_wraps_past_midnight():
    start, end = time(22, 0), time(6, 0)
    assert predictor.is_within_window(ist(23, 0), start, end) is True
    assert predictor.is_within_window(ist(5, 0), start, end) is True
    assert predictor.is_within_window(ist(12, 0), start, end) is False


def test_window_label_is_human_readable():
    assert predictor.window_label() == ("6:00 PM", "10:00 PM")


def test_closed_window_returns_message_and_never_runs_the_model():
    result = predictor.predict(FRAME, now=ist(12, 0), runner=never_called)
    assert result["status"] == "closed"
    assert "6:00 PM" in result["message"] and "10:00 PM" in result["message"]


def test_closed_window_for_video_frames_also_skips_the_model():
    result = predictor.predict_frames([FRAME, FRAME], now=ist(9, 0), runner=never_called)
    assert result["status"] == "closed"


# ---- predictions -----------------------------------------------------------------

def test_confident_prediction_returns_word_and_confidence():
    result = predictor.predict(FRAME, now=ist(19, 0), runner=lambda key, batch: [probs(2, 0.9)])
    assert result["status"] == "ok"
    assert result["word"] == "hungry"
    assert result["confidence"] == pytest.approx(0.9, abs=1e-3)
    assert len(result["top"]) == predictor.TOP_K


def test_low_confidence_is_flagged_uncertain_not_ok():
    uniform = np.full(8, 1 / 8)
    result = predictor.predict(FRAME, now=ist(19, 0), runner=lambda key, batch: [uniform])
    assert result["status"] == "uncertain"


def test_runner_is_asked_for_the_sign_language_model():
    seen = {}

    def runner(key, batch):
        seen["key"], seen["shape"] = key, batch.shape
        return [probs(0)]

    predictor.predict(FRAME, now=ist(19, 0), runner=runner)
    size = predictor.INPUT_SIZE
    assert seen["key"] == "sign_language"
    assert seen["shape"] == (1, size, size, 3)


def test_video_frames_are_averaged():
    runner = lambda key, batch: [probs(1, 0.8), probs(1, 0.6)]  # both frames say "careful"
    result = predictor.predict_frames([FRAME, FRAME], now=ist(19, 0), runner=runner)
    assert result["word"] == "careful"
    assert result["frames_used"] == 2
    assert result["confidence"] == pytest.approx(0.7, abs=1e-3)


def test_no_frames_raises_a_clear_error():
    with pytest.raises(ValueError):
        predictor.predict_frames([], now=ist(19, 0), runner=never_called)


def test_class_names_are_the_eight_words():
    assert predictor.CLASS_NAMES == [
        "book", "careful", "hungry", "i need help", "thirsty", "understand", "wait", "write"
    ]


# ---- preprocessing and video -----------------------------------------------------

def test_preprocess_scales_pixels_to_minus_one_to_one():
    size = predictor.INPUT_SIZE
    black = predictor.preprocess(np.zeros((50, 50, 3), dtype=np.uint8))
    white = predictor.preprocess(np.full((50, 50, 3), 255, dtype=np.uint8))
    assert black.shape == (size, size, 3)
    assert black.min() == pytest.approx(-1.0) and white.max() == pytest.approx(1.0)


def test_missing_video_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        predictor.sample_video_frames(tmp_path / "missing.mp4")


def test_sample_video_frames_returns_rgb_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot write mp4 files.")
    for _ in range(30):
        writer.write(np.full((48, 64, 3), 128, dtype=np.uint8))
    writer.release()
    try:
        frames = predictor.sample_video_frames(path, max_frames=5)
    except ValueError:
        pytest.skip("This OpenCV build cannot read mp4 files.")
    assert 1 <= len(frames) <= 5
    assert frames[0].shape == (48, 64, 3)
