"""Download only a small CIFAR-10 subset (default 2000 train + 500 test)
from the Hugging Face dataset viewer API, instead of the full 170 MB archive.
Saves cifar10_subset.npz next to this script with the same arrays/shapes as
keras.datasets.cifar10.load_data(): X uint8 (N,32,32,3), y uint8 (N,1).
Note: HF serves the images as JPEG, so pixels differ very slightly from the original PNGs.
"""
import io, json, sys, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

API = "https://datasets-server.huggingface.co/rows?dataset=uoft-cs/cifar10&config=plain_text&split={}&offset={}&length=100"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar10_subset.npz")


def get(url, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:
            if t == tries - 1:
                raise
            print(f"  retry ({e})", file=sys.stderr)


def fetch(split, n):
    rows = []
    for off in range(0, n, 100):
        page = json.loads(get(API.format(split, off)))["rows"]
        rows += page[: n - off]
        print(f"{split}: listed {len(rows)}/{n}", flush=True)
    with ThreadPoolExecutor(16) as ex:
        imgs = list(ex.map(lambda r: np.array(Image.open(io.BytesIO(get(r["row"]["img"]["src"]))).convert("RGB")), rows))
    X = np.stack(imgs).astype(np.uint8)
    y = np.array([[r["row"]["label"]] for r in rows], dtype=np.uint8)
    print(f"{split}: downloaded {len(X)} images", flush=True)
    return X, y


if __name__ == "__main__":
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    X_train, y_train = fetch("train", n_train)
    X_test, y_test = fetch("test", n_test)
    np.savez_compressed(OUT, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("saved", OUT, X_train.shape, X_test.shape)
