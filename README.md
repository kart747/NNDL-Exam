# NNDL-Exam

Neural Networks & Deep Learning lab experiments (TensorFlow / Keras): MLPs, CNNs, RNNs,
regularisation and transfer learning.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open the notebooks from inside `notebooks/` (they read data from `../data`).

### GPU memory

Every TensorFlow notebook starts with a cell that turns on GPU memory growth. Without it
TensorFlow reserves the whole GPU, and a second open notebook fails with an out-of-memory
error on a 4 GB card. Keep that cell first.

### CIFAR-10

`keras.datasets.cifar10.load_data()` downloads from a slow server. Two helpers in `scripts/`
fetch it from a faster mirror:

```bash
python scripts/get_cifar_full.py      # full dataset -> ~/.keras/datasets/cifar10_full.npz
python scripts/get_cifar_subset.py    # small subset (2000 train / 500 test), saved next to the script
```

then `from cifar_local import load_data` (add `scripts/` to `sys.path`).

## Notebooks

| # | Notebook | Topic |
|---|----------|-------|
| 01 | `01_breast_cancer_mlp` | Binary classification MLP, training curves |
| 02 | `02_mnist_mlp` | MNIST with a 2-hidden-layer MLP, sample predictions |
| 03 | `03_mnist_mlp_confusion_matrix` | Smaller MNIST MLP with a confusion matrix |
| 04 | `04_digits_early_stopping` | Early stopping vs. fixed epochs on digits |
| 05 | `05_digits_early_stopping_deep` | Same, deeper network, with timing |
| 06 | `06_moons_dropout_rates` | Dropout rates 0 / 0.2 / 0.5 on make_moons (loop) |
| 07 | `07_moons_dropout_comparison` | Same, three models compared side by side |
| 08 | `08_olivetti_faces_cnn` | Small CNN on Olivetti faces |
| 09 | `09_olivetti_faces_deeper_cnn` | Deeper CNN on Olivetti faces |
| 10 | `10_cifar10_augmentation` | CNN with and without data augmentation |
| 11 | `11_cifar10_transfer_learning` | VGG16 vs VGG19 vs ResNet50 (frozen ImageNet features) |
| 12 | `12_imdb_rnn` | SimpleRNN sentiment model on IMDB |
| 13 | `13_sentiment_csv_rnn` | SimpleRNN on `data/Sentimental Analysis.csv` |
| 14 | `14_imdb_and_sentiment_rnn` | IMDB at sequence length 200 plus the CSV model |
| 15 | `15_rice_leaf_cnn` | CNN on a rice leaf disease image folder |

### Rice leaf dataset

Notebook 15 needs an image folder with one sub-folder per class. It is not included. Put it in
`rice_leaf_diseases/` next to where you run the notebook, or set `RICE_DATA_DIR`.

## Fixes made when this was turned into a repo

- GPU out-of-memory (`ResourceExhaustedError`, `cudaSetDevice() ... out of memory`): memory growth enabled.
- Sentiment notebooks: `Tokenizer()` could emit word ids above the `Embedding(10000, ...)` size; now `Tokenizer(num_words=10000)`.
- Rice leaf notebook: crashed on a missing folder with a bare `FileNotFoundError`; path is now configurable and the class count is read from the data.
- CIFAR-10 scripts: create `~/.keras/datasets` if missing, fixed a wrong docstring.
- Transfer learning: earlier drafts divided pixels by 255, which is wrong for VGG/ResNet. Only the corrected notebook (own `preprocess_input` per model) is kept.
