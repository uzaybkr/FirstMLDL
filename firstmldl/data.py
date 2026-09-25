import hashlib
import urllib.request
from pathlib import Path

import numpy as np

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
CACHE = Path.home() / ".cache" / "firstmldl" / "mnist.npz"


def load_mnist(path=CACHE):
    """Return (X_train, y_train, X_test, y_test); images flattened to 784 floats in [0, 1]."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading MNIST to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MNIST_URL, tmp)
        if hashlib.sha256(tmp.read_bytes()).hexdigest() != MNIST_SHA256:
            tmp.unlink()
            raise RuntimeError("MNIST download failed checksum")
        tmp.replace(path)
    with np.load(path) as d:
        return _flat(d["x_train"]), d["y_train"], _flat(d["x_test"]), d["y_test"]


def _flat(images):
    return images.reshape(len(images), -1).astype(np.float32) / 255
