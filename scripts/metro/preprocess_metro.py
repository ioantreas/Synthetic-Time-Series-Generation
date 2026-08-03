# preprocess_metro.py

import json
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path("../../data/raw/metro/Metro_Interstate_Traffic_Volume.csv")

OUT_DIR = Path("../../data/processed/metro/")
OUT_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "traffic_volume",
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
]


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Original shape:", df.shape)

    # keep only desired features
    df = df[FEATURES]

    # numeric conversion
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)

    df = df.dropna()

    print("Removed rows:", before - len(df))
    print("Final shape:", df.shape)

    # chronological split
    split = int(0.8 * len(df))

    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    print("Train:", train_df.shape)
    print("Test :", test_df.shape)

    # normalize using training statistics
    mean = train_df.mean()
    std = train_df.std() + 1e-8

    train_norm = ((train_df - mean) / std).clip(-5, 5)
    test_norm = ((test_df - mean) / std).clip(-5, 5)

    train = train_norm.values.astype(np.float32)
    test = test_norm.values.astype(np.float32)

    np.save(OUT_DIR / "train.npy", train)
    np.save(OUT_DIR / "test.npy", test)

    np.save(OUT_DIR / "mean.npy", mean.values)
    np.save(OUT_DIR / "std.npy", std.values)

    with open(OUT_DIR / "features.json", "w") as f:
        json.dump(FEATURES, f, indent=2)

    print("\nSaved:")
    print("train:", train.shape)
    print("test :", test.shape)
    print("features:", len(FEATURES))


if __name__ == "__main__":
    main()