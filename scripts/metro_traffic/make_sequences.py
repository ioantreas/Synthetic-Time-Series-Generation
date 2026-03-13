import numpy as np
from pathlib import Path

DATA_DIR = Path("../../data/processed/metro_traffic/")
OUT_DIR = Path("../../data/diffusion_ready/metro_traffic/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 168

def make_sequences(arr, seq_len):
    X = []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
    return np.stack(X)

print("Loading processed data...")
train = np.load(DATA_DIR / "train.npy")
test = np.load(DATA_DIR / "test.npy")

train_seq = make_sequences(train, SEQ_LEN)
test_seq = make_sequences(test, SEQ_LEN)

np.save(OUT_DIR / "train_seq.npy", train_seq)
np.save(OUT_DIR / "test_seq.npy", test_seq)

print("Saved:")
print(" train_seq:", train_seq.shape)
print(" test_seq:", test_seq.shape)