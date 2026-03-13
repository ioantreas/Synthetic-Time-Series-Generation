import numpy as np
from pathlib import Path

DATA_DIR = Path("../../data/processed/tourism/")
OUT_DIR = Path("../../data/diffusion_ready/tourism/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 24 # 6 years (quarterly)


def make_sequences(arr, seq_len):
    X = []
    for i in range(0, len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
    return np.stack(X)


print("Loading processed data...")
# IMPORTANT: load full dataset, not train/test split
data = np.load(DATA_DIR / "train.npy")  # this should contain ALL 76 timesteps

print("Data shape:", data.shape)

seq = make_sequences(data, SEQ_LEN)

np.save(OUT_DIR / "train_seq.npy", seq)

print("Saved:")
print(" train_seq:", seq.shape)