import os
import secrets
import socket
import subprocess
import threading
import time
from multiprocessing.connection import Client
from pathlib import Path

from worker.protocol import decode_output, encode

ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPT = ROOT / "worker" / "server.py"
PYTHON_ENV_VAR = "TF210_PYTHON"
GPU_ENV_VAR = "WORKER_USE_GPU"
STARTUP_TIMEOUT = 120
REQUEST_TIMEOUT = 120


class WorkerError(RuntimeError):
    pass


class WorkerClient:
    def __init__(self, python_path=None):
        self.python_path = python_path or os.environ.get(PYTHON_ENV_VAR)
        self._process = None
        self._connection = None
        self._lock = threading.Lock()

    def _start(self):
        if not self.python_path or not Path(self.python_path).is_file():
            raise WorkerError(
                f"The TensorFlow 2.10 environment was not found. Set the {PYTHON_ENV_VAR} "
                "environment variable to the Python interpreter path."
            )

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        key = secrets.token_hex(16)

        env = dict(os.environ)
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        if os.environ.get(GPU_ENV_VAR) != "1":
            env["CUDA_VISIBLE_DEVICES"] = "-1"

        self._process = subprocess.Popen(
            [self.python_path, str(SERVER_SCRIPT), str(port), key],
            cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline:
            if self._process.poll() is not None:
                raise WorkerError("The model worker exited during start-up.")
            try:
                self._connection = Client(("127.0.0.1", port), authkey=key.encode())
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(0.3)
        self.close()
        raise WorkerError("The model worker did not start in time.")

    def _alive(self):
        return self._process is not None and self._process.poll() is None and self._connection

    def predict(self, model_key, batch):
        with self._lock:
            if not self._alive():
                self.close()
                self._start()
            try:
                self._connection.send({"op": "predict", "model": model_key, "batch": encode(batch)})
                if not self._connection.poll(REQUEST_TIMEOUT):
                    raise WorkerError("The model worker did not respond in time.")
                response = self._connection.recv()
            except (EOFError, OSError, BrokenPipeError) as error:
                self.close()
                raise WorkerError("The model worker connection was lost.") from error

        if not response["ok"]:
            raise WorkerError(response["error"])
        return decode_output(response["output"])

    def close(self):
        if self._connection:
            try:
                self._connection.send({"op": "shutdown"})
            except Exception:
                pass
            try:
                self._connection.close()
            except Exception:
                pass
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._connection = self._process = None


_client = WorkerClient()


def predict(model_key, batch):
    return _client.predict(model_key, batch)

