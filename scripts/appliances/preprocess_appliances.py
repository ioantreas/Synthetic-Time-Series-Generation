import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_PATH = Path("../../data/raw/appliances/energydata_complete.csv")

OUT_DIR = Path("../../data/processed/appliances/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.8


def main():
    print("Loading CSV...")
    df = pd.read_csv(RAW_PATH)

    print("Columns:", df.columns.tolist())

    # ---------------------------------
    # Handle datetime
    # ---------------------------------
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Create cyclic time features
    df["hour"] = df["date"].dt.hour
    df["dow"] = df["date"].dt.dayofweek

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)

    # ---------------------------------
    # Drop useless columns
    # ---------------------------------
    drop_cols = ["date", "hour", "dow", "rv1", "rv2"]

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=c)

    # ---------------------------------
    # Final feature list
    # ---------------------------------
    features = df.columns.tolist()

    print("Final features:", features)
    print("Timesteps:", len(df))

    # ---------------------------------
    # Handle missing values
    # ---------------------------------
    df = df.ffill().bfill()

    # ---------------------------------
    # Normalize
    # ---------------------------------
    mean = df.mean()
    std = df.std() + 1e-8

    df_norm = (df - mean) / std

    # ---------------------------------
    # Train/test split
    # ---------------------------------
    split = int(len(df_norm) * TRAIN_FRAC)

    train = df_norm.iloc[:split].values.astype("float32")
    test = df_norm.iloc[split:].values.astype("float32")

    # ---------------------------------
    # Save arrays
    # ---------------------------------
    np.save(OUT_DIR / "train.npy", train)
    np.save(OUT_DIR / "test.npy", test)

    # Save normalization stats (needed for denormalization)
    np.save(OUT_DIR / "mean.npy", mean.values)
    np.save(OUT_DIR / "std.npy", std.values)

    # Save feature names
    with open(OUT_DIR / "features.json", "w") as f:
        json.dump(features, f, indent=2)

    print("\nSaved:")
    print(" train.npy shape:", train.shape)
    print(" test.npy shape:", test.shape)
    print(" mean.npy")
    print(" std.npy")
    print(" features.json")

    print("\nReady for autoencoder + diffusion ✔")


if __name__ == "__main__":
    main()