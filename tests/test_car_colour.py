"""Task 5: cropping car boxes and turning model output into a colour name."""
import numpy as np
import pytest

from modules.car_colour import colour

IMAGE = np.full((200, 200, 3), 120, dtype=np.uint8)
NAMES = colour.COLOUR_NAMES  # ["black", "blue", "red", "white"] in the test registry


def row(index, value):
    probs = np.full(len(NAMES), (1 - value) / (len(NAMES) - 1))
    probs[index] = value
    return probs


def test_crop_is_clamped_to_the_image():
    crop = colour.crop_box(IMAGE, (-20, -20, 60, 60))
    assert crop is not None
    assert crop.shape[0] <= 200 and crop.shape[1] <= 200


def test_zero_size_box_gives_no_crop():
    assert colour.crop_box(IMAGE, (10, 10, 10, 10)) is None


def test_confident_large_car_is_reliable_and_named():
    results = colour.classify_colours(
        IMAGE, [(10, 10, 110, 110)], runner=lambda key, batch: [row(NAMES.index("blue"), 0.8)]
    )
    assert results[0]["colour"] == "blue"
    assert results[0]["reliable"] is True


def test_runner_gets_the_colour_model_and_a_correctly_sized_batch():
    seen = {}

    def runner(key, batch):
        seen["key"], seen["shape"] = key, batch.shape
        return [row(0, 0.9), row(1, 0.9)]

    colour.classify_colours(IMAGE, [(0, 0, 100, 100), (100, 100, 190, 190)], runner=runner)
    size = colour.COLOUR_SIZE
    assert seen["key"] == "car_colour"
    assert seen["shape"] == (2, size, size, 3)


def test_low_confidence_is_not_reliable():
    results = colour.classify_colours(
        IMAGE, [(10, 10, 110, 110)], runner=lambda key, batch: [row(2, 0.3)]
    )
    assert results[0]["reliable"] is False


def test_tiny_car_is_not_reliable_even_when_model_is_confident():
    results = colour.classify_colours(
        IMAGE, [(10, 10, 30, 30)], runner=lambda key, batch: [row(0, 0.95)]
    )
    assert results[0]["colour"] == "black"
    assert results[0]["reliable"] is False


def test_invalid_box_is_skipped_and_order_is_kept():
    results = colour.classify_colours(
        IMAGE,
        [(5, 5, 5, 5), (10, 10, 110, 110)],
        runner=lambda key, batch: [row(NAMES.index("white"), 0.9)],
    )
    assert results[0] == {"colour": None, "confidence": 0.0, "reliable": False}
    assert results[1]["colour"] == "white"


def test_no_boxes_never_calls_the_model():
    def never(*_):
        raise AssertionError("runner should not be called")

    assert colour.classify_colours(IMAGE, [], runner=never) == []
