# make_tep_sequences.py

from pathlib import Path

import numpy as np


DATA_DIR = Path("../../data/processed/tep/")
OUT_DIR = Path("../../data/diffusion_ready/tep/64")

OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 64
TARGET_WINDOWS = 30000
MAX_TEST_WINDOWS = 30000


def make_sequences(arr, seq_len, stride):

    X = []

    for i in range(0, len(arr) - seq_len, stride):
        X.append(arr[i:i + seq_len])

    if len(X) == 0:
        return None

    return np.stack(X)


print("Loading processed TEP data...")

train = np.load(DATA_DIR / "train.npy")
test = np.load(DATA_DIR / "test.npy")

train_runs = np.load(DATA_DIR / "train_runs.npy")
test_runs = np.load(DATA_DIR / "test_runs.npy")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("Unique train runs:", len(np.unique(train_runs)))
print("Unique test runs:", len(np.unique(test_runs)))

# --------------------------------
# adaptive stride
# --------------------------------

estimated_stride = max(
    1,
    (len(train) - SEQ_LEN) // TARGET_WINDOWS
)

print("\nUsing stride (train):", estimated_stride)

# --------------------------------
# TRAIN SEQUENCES
# --------------------------------

all_train_seq = []

for run_id in np.unique(train_runs):

    arr = train[train_runs == run_id]

    seq = make_sequences(
        arr,
        SEQ_LEN,
        estimated_stride
    )

    if seq is not None:
        all_train_seq.append(seq)

train_seq = np.concatenate(all_train_seq)

print("\nTrain sequences before subsample:", train_seq.shape)

# --------------------------------
# TEST SEQUENCES
# --------------------------------

test_stride = max(1, estimated_stride * 2)

print("\nUsing stride (test):", test_stride)

all_test_seq = []

for run_id in np.unique(test_runs):

    arr = test[test_runs == run_id]

    seq = make_sequences(
        arr,
        SEQ_LEN,
        test_stride
    )

    if seq is not None:
        all_test_seq.append(seq)

test_seq = np.concatenate(all_test_seq)

print("\nTest sequences before subsample:", test_seq.shape)

# --------------------------------
# save
# --------------------------------

np.save(OUT_DIR / "train_seq.npy", train_seq)
np.save(OUT_DIR / "test_seq.npy", test_seq)

print("\nSaved:")
print("train_seq:", train_seq.shape)
print("test_seq:", test_seq.shape)