import numpy as np
from pathlib import Path

DATA_DIR = Path("../../data/processed/appliances/")
OUT_DIR = Path("../../data/diffusion_ready/appliances/")
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

print("Original train shape:", train.shape)

train_seq = make_sequences(train, SEQ_LEN)
test_seq = make_sequences(test, SEQ_LEN)

np.save(OUT_DIR / "train_seq.npy", train_seq)
np.save(OUT_DIR / "test_seq.npy", test_seq)

print("\nSaved diffusion-ready data:")
print(" train_seq shape:", train_seq.shape)
print(" test_seq shape:", test_seq.shape)
