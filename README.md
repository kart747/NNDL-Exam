# NNDL-Exam

Hands-on **Neural Networks & Deep Learning** experiments in TensorFlow / Keras: multilayer perceptrons,
convolutional networks, recurrent networks, regularisation (dropout, L1/L2, early stopping), data
augmentation and transfer learning.

Each notebook is a short, self-contained experiment on a classic dataset. They are saved without
outputs, so run them to reproduce the results.

## Contents

| # | Notebook | What it covers |
|---|----------|----------------|
| 01 | [`01_breast_cancer_mlp`](notebooks/01_breast_cancer_mlp.ipynb) | MLP classifier (breast cancer) and MLP regressor (California housing) |
| 02 | [`02_mnist_mlp`](notebooks/02_mnist_mlp.ipynb) | MNIST with a two-hidden-layer MLP, predictions, confusion matrix |
| 03 | [`03_mnist_mlp_confusion_matrix`](notebooks/03_mnist_mlp_confusion_matrix.ipynb) | Smaller MNIST MLP with a confusion matrix |
| 04 | [`04_digits_early_stopping`](notebooks/04_digits_early_stopping.ipynb) | Early stopping vs. fixed epochs: time, epochs, overfitting gap |
| 05 | [`05_digits_early_stopping_deep`](notebooks/05_digits_early_stopping_deep.ipynb) | The same comparison with a deeper network |
| 06 | [`06_moons_dropout_rates`](notebooks/06_moons_dropout_rates.ipynb) | Dropout rates 0 / 0.2 / 0.5 on `make_moons` |
| 07 | [`07_moons_dropout_comparison`](notebooks/07_moons_dropout_comparison.ipynb) | Dropout vs. L1 vs. L2 regularisation |
| 08 | [`08_olivetti_faces_cnn`](notebooks/08_olivetti_faces_cnn.ipynb) | Small CNN on the Olivetti faces |
| 09 | [`09_olivetti_faces_deeper_cnn`](notebooks/09_olivetti_faces_deeper_cnn.ipynb) | Deeper CNN on the Olivetti faces |
| 10 | [`10_cifar10_augmentation`](notebooks/10_cifar10_augmentation.ipynb) | CIFAR-10 CNN with and without data augmentation |
| 11 | [`11_cifar10_transfer_learning`](notebooks/11_cifar10_transfer_learning.ipynb) | VGG16 vs. VGG19 vs. ResNet50 with frozen ImageNet features |
| 12 | [`12_imdb_rnn`](notebooks/12_imdb_rnn.ipynb) | SimpleRNN sentiment model on IMDB |
| 13 | [`13_sentiment_csv_rnn`](notebooks/13_sentiment_csv_rnn.ipynb) | SimpleRNN on the included review-sentiment CSV |
| 14 | [`14_imdb_and_sentiment_rnn`](notebooks/14_imdb_and_sentiment_rnn.ipynb) | IMDB at sequence length 200, plus the CSV model |
| 15 | [`15_rice_leaf_cnn`](notebooks/15_rice_leaf_cnn.ipynb) | Image-folder CNN for rice leaf diseases (dataset not included) |

## Repository layout

```
notebooks/   the experiments (run them from inside this folder)
data/        Sentimental Analysis.csv, used by notebooks 13 and 14
scripts/     CIFAR-10 download helpers
```

## Getting started

```bash
git clone https://github.com/kart747/NNDL-Exam.git
cd NNDL-Exam
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd notebooks && jupyter lab
```

Standard datasets (MNIST, IMDB, CIFAR-10, Olivetti faces, digits, California housing) are downloaded
automatically by Keras or scikit-learn on first use.

### GPU memory

Each TensorFlow notebook begins with a cell that enables GPU memory growth. Without it TensorFlow
reserves the whole GPU, and a second open notebook fails with an out-of-memory error on small cards
(this repo was developed on a 4 GB laptop GPU). Keep that cell first.

### Faster CIFAR-10

`keras.datasets.cifar10.load_data()` downloads from a slow server. Two helpers fetch it from a faster mirror:

```bash
python scripts/get_cifar_full.py     # full dataset -> ~/.keras/datasets/cifar10_full.npz
python scripts/get_cifar_subset.py   # 2000 train / 500 test images -> scripts/cifar10_subset.npz
```

Then use `from cifar_local import load_data` (add `scripts/` to `sys.path`) in place of `keras.datasets.cifar10.load_data`.

### Rice leaf dataset (notebook 15)

The notebook expects a folder with one sub-folder of images per class. It is not included. Put it in
`rice_leaf_diseases/` next to where you run the notebook, or set the `RICE_DATA_DIR` environment variable.

## Data credits

`data/Sentimental Analysis.csv` is a sample of the Rotten Tomatoes movie-review polarity data
(Pang & Lee, 2005), included only so the notebooks run out of the box. It remains subject to its
original terms.
