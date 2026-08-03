# make_metro_sequences.py
import math
from pathlib import Path

import numpy as np


DATA_DIR = Path("../../data/processed/metro/")
OUT_DIR = Path("../../data/diffusion_ready/metro/168")

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 168
TARGET_WINDOWS = 20000


def make_sequences(arr, seq_len, stride):

    X = []

    for i in range(0, len(arr) - seq_len + 1, stride):
        X.append(arr[i:i + seq_len])

    return np.stack(X)


print("Loading processed Metro dataset...")

train = np.load(DATA_DIR / "train.npy")
test = np.load(DATA_DIR / "test.npy")

print("Train:", train.shape)
print("Test :", test.shape)

train_stride = max(
    1,
    round((len(train) - SEQ_LEN) / TARGET_WINDOWS),
    )

test_stride = train_stride

print("Train stride:", train_stride)
print("Test stride :", test_stride)

train_seq = make_sequences(
    train,
    SEQ_LEN,
    train_stride,
)

test_seq = make_sequences(
    test,
    SEQ_LEN,
    test_stride,
)

print("Train sequences:", train_seq.shape)
print("Test sequences :", test_seq.shape)

np.save(OUT_DIR / "train_seq.npy", train_seq)
np.save(OUT_DIR / "test_seq.npy", test_seq)

print("\nSaved.")