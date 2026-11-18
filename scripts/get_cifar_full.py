"""Download the full CIFAR-10 dataset from Hugging Face (fast mirror of the
original Toronto release; images stored as lossless PNG) and save it as
cifar10_full.npz next to this script.

Arrays match keras.datasets.cifar10.load_data():
X uint8 (N,32,32,3), y uint8 (N,1).  Load with cifar_local.load_data().
"""
import io, os, urllib.request
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/"
HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(split):
    path = os.path.join(HERE, f".{split}.parquet")
    if not os.path.exists(path):
        print(f"downloading {split} ...", flush=True)
        urllib.request.urlretrieve(BASE + f"{split}-00000-of-00001.parquet", path + ".part")
        os.replace(path + ".part", path)
    t = pq.read_table(path).to_pydict()
    X = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in t["img"]])
    y = np.array(t["label"], dtype=np.uint8).reshape(-1, 1)
    os.remove(path)
    print(f"{split}: {X.shape}", flush=True)
    return X.astype(np.uint8), y


if __name__ == "__main__":
    X_train, y_train = fetch("train")
    X_test, y_test = fetch("test")
    out = os.path.expanduser("~/.keras/datasets/cifar10_full.npz")
    np.savez(out, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("saved", out)
