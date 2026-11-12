import sys
from multiprocessing.connection import Listener
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.protocol import decode, encode_output  # noqa: E402

HOST = "127.0.0.1"


def tensorflow_loader(model_key):
    import tensorflow as tf

    from model_registry import MODELS

    return tf.keras.models.load_model(str(MODELS[model_key]["path"]), compile=False)


def handle(request, models, loader):
    operation = request.get("op")
    if operation == "ping":
        return {"ok": True}
    if operation != "predict":
        return {"ok": False, "error": f"Unknown operation: {operation}"}

    key = request["model"]
    if key not in models:
        models[key] = loader(key)
    output = models[key].predict(decode(request["batch"]), verbose=0)
    return {"ok": True, "output": encode_output(output)}


def serve(port, authkey, loader=tensorflow_loader):
    models = {}
    with Listener((HOST, port), authkey=authkey) as listener:
        while True:
            with listener.accept() as connection:
                while True:
                    try:
                        request = connection.recv()
                    except (EOFError, ConnectionResetError):
                        break
                    if request.get("op") == "shutdown":
                        return
                    try:
                        response = handle(request, models, loader)
                    except Exception as error:
                        response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                    connection.send(response)


if __name__ == "__main__":
    serve(int(sys.argv[1]), sys.argv[2].encode())
