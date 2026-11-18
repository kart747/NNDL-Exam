"""Drop-in replacement for keras.datasets.cifar10.load_data() that reads the
local cifar10_full.npz (made by get_cifar_full.py) instead of downloading
from the slow Toronto server.

    from cifar_local import load_data
    (X_train, y_train), (X_test, y_test) = load_data()
"""
import os
import numpy as np

_PATH = os.path.expanduser("~/.keras/datasets/cifar10_full.npz")


def load_data():
    if not os.path.exists(_PATH):
        raise FileNotFoundError(f"{_PATH} not found - run scripts/get_cifar_full.py first")
    d = np.load(_PATH)
    return (d["X_train"], d["y_train"]), (d["X_test"], d["y_test"])
