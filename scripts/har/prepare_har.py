import json
from pathlib import Path

import numpy as np


RAW_PATH = Path("../../data/raw/har/")
OUT_DIR = Path("../../data/diffusion_ready/har/")
OUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURE_FILES = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def load_split(split):

    split_dir = RAW_PATH / split

    signals = []

    for feat in FEATURE_FILES:

        path = split_dir / f"{feat}_{split}.txt"

        print("Loading:", path)

        data = np.loadtxt(path)

        # shape = (N, 128)
        signals.append(data)

    # stack → (N, 128, C)
    X = np.stack(signals, axis=-1)

    return X.astype("float32")


def main():

    print("Loading HAR dataset...")

    train = load_split("train")
    test = load_split("test")

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    # -----------------------------
    # Normalize using train stats
    # -----------------------------

    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + 1e-8

    train = (train - mean) / std
    test = (test - mean) / std

    # -----------------------------
    # Save arrays
    # -----------------------------

    np.save(OUT_DIR / "train_seq.npy", train)
    np.save(OUT_DIR / "test_seq.npy", test)

    np.save(OUT_DIR / "mean.npy", mean.squeeze())
    np.save(OUT_DIR / "std.npy", std.squeeze())

    with open(OUT_DIR / "features.json", "w") as f:
        json.dump(FEATURE_FILES, f, indent=2)

    print("\nSaved:")
    print(" train_seq shape:", train.shape)
    print(" test_seq shape:", test.shape)
    print(" mean.npy")
    print(" std.npy")
    print(" features.json")

    print("\nReady for autoencoder + diffusion ✔")


if __name__ == "__main__":
    main()