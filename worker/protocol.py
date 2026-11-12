import numpy as np


def encode(array):
    array = np.ascontiguousarray(array)
    return {"dtype": str(array.dtype), "shape": list(array.shape), "data": array.tobytes()}


def decode(payload):
    return np.frombuffer(payload["data"], dtype=payload["dtype"]).reshape(payload["shape"]).copy()


def encode_output(output):
    if isinstance(output, dict):
        return {"kind": "dict", "value": {k: encode(v) for k, v in output.items()}}
    if isinstance(output, (list, tuple)):
        return {"kind": "list", "value": [encode(v) for v in output]}
    return {"kind": "array", "value": encode(output)}


def decode_output(payload):
    kind, value = payload["kind"], payload["value"]
    if kind == "dict":
        return {k: decode(v) for k, v in value.items()}
    if kind == "list":
        return [decode(v) for v in value]
    return decode(value)
