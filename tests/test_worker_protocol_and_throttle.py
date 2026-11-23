"""The TensorFlow worker's wire format, and the camera frame throttle."""
import numpy as np
import pytest

from modules.common.throttle import Throttle
from worker.protocol import decode, decode_output, encode, encode_output


@pytest.mark.parametrize("dtype", ["float32", "float64", "uint8", "int64"])
def test_array_round_trip_keeps_values_dtype_and_shape(dtype):
    original = (np.arange(24).reshape(2, 3, 4)).astype(dtype)
    restored = decode(encode(original))
    assert restored.dtype == original.dtype
    assert restored.shape == original.shape
    assert np.array_equal(restored, original)


def test_non_contiguous_arrays_survive():
    original = np.arange(12, dtype="float32").reshape(3, 4).T
    assert np.array_equal(decode(encode(original)), original)


def test_decoded_array_is_writable():
    restored = decode(encode(np.zeros(3, dtype="float32")))
    restored[0] = 1.0  # would raise if the array were a read-only view


def test_dict_output_round_trip():
    output = {"age": np.array([[41.5]], dtype="float32"), "gender": np.array([[0.2]], dtype="float32")}
    restored = decode_output(encode_output(output))
    assert set(restored) == {"age", "gender"}
    assert np.array_equal(restored["age"], output["age"])


def test_list_output_round_trip():
    output = [np.ones((2, 2), dtype="float32"), np.zeros(3, dtype="float32")]
    restored = decode_output(encode_output(output))
    assert len(restored) == 2 and np.array_equal(restored[0], output[0])


def test_single_array_output_round_trip():
    output = np.random.default_rng(0).random((4, 8)).astype("float32")
    assert np.array_equal(decode_output(encode_output(output)), output)


def test_throttle_allows_first_call_then_blocks_until_interval():
    now = [0.0]
    throttle = Throttle(interval=1.0, clock=lambda: now[0])
    assert throttle.ready() is True
    now[0] = 0.5
    assert throttle.ready() is False
    now[0] = 1.0
    assert throttle.ready() is True
    now[0] = 1.2
    assert throttle.ready() is False
