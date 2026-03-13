import numpy as np
from pathlib import Path

DATA_DIR = Path("../../data/processed/store/")
OUT_DIR = Path("../../data/diffusion_ready/store/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 168
TARGET_WINDOWS = 15000


def make_sequences(arr, seq_len, stride):
    X = []
    for i in range(0, len(arr) - seq_len, stride):
        X.append(arr[i:i+seq_len])
    return np.stack(X)


print("Loading processed data...")
train = np.load(DATA_DIR / "train.npy")
test = np.load(DATA_DIR / "test.npy")

# Auto-compute stride for ~15k windows
estimated_stride = max(1, (len(train) - SEQ_LEN) // TARGET_WINDOWS)
print("Using stride:", estimated_stride)

train_seq = make_sequences(train, SEQ_LEN, estimated_stride)
test_seq = make_sequences(test, SEQ_LEN, estimated_stride)

np.save(OUT_DIR / "train_seq_small.npy", train_seq)
np.save(OUT_DIR / "test_seq_small.npy", test_seq)

print("Saved:")
print(" train_seq:", train_seq.shape)
print(" test_seq:", test_seq.shape)