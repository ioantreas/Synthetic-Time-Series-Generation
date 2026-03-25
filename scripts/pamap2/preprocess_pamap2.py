import json
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("../../data/raw/pamap2/")
OUT_DIR = Path("../../data/processed/pamap2/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Subject split
# -----------------------------
TRAIN_SUBJECTS = [101, 102, 103, 104, 105, 106, 107]
TEST_SUBJECTS = [108, 109]


# -----------------------------
# Feature selection
# -----------------------------
def select_features(df):
    """
    Select clean, reliable features:
    - heart rate
    - per IMU: temperature, accel (±16g), gyro, magnetometer
    """

    keep = []

    # heart rate
    keep.append(2)

    # IMU blocks
    imu_starts = [3, 20, 37]

    for start in imu_starts:
        # temperature
        keep.append(start)

        # accel (±16g)
        keep += [start + 1, start + 2, start + 3]

        # skip accel (±6g) → start+4..6

        # gyroscope
        keep += [start + 7, start + 8, start + 9]

        # magnetometer
        keep += [start + 10, start + 11, start + 12]

        # skip orientation → start+13..16

    return df[keep]


# -----------------------------
# Load one subject
# -----------------------------
def load_subject(path):
    print("Loading:", path)

    data = np.loadtxt(path)
    df = pd.DataFrame(data)

    # ---------------------------------
    # Remove transient activity
    # ---------------------------------
    df = df[df[1] != 0]

    # ---------------------------------
    # Select features (important!)
    # ---------------------------------
    df = select_features(df)

    # ---------------------------------
    # Handle NaNs
    # ---------------------------------
    df = df.ffill().bfill()

    # Drop any remaining NaNs (safety)
    df = df.dropna()

    return df


# -----------------------------
# Load multiple subjects
# -----------------------------
def load_split(subjects):
    dfs = []

    for s in subjects:
        path = RAW_DIR / f"subject{s}.dat"
        df = load_subject(path)
        dfs.append(df)

    return pd.concat(dfs, axis=0)


# -----------------------------
# Main
# -----------------------------
def main():

    print("Loading PAMAP2 dataset...")

    train_df = load_split(TRAIN_SUBJECTS)
    test_df = load_split(TEST_SUBJECTS)

    print("Train shape (before norm):", train_df.shape)
    print("Test shape  (before norm):", test_df.shape)

    # ---------------------------------
    # Normalize (train stats only)
    # ---------------------------------
    mean = train_df.mean()
    std = train_df.std() + 1e-8

    train = ((train_df - mean) / std).values.astype("float32")
    test = ((test_df - mean) / std).values.astype("float32")

    # ---------------------------------
    # Save arrays
    # ---------------------------------
    np.save(OUT_DIR / "train.npy", train)
    np.save(OUT_DIR / "test.npy", test)

    np.save(OUT_DIR / "mean.npy", mean.values)
    np.save(OUT_DIR / "std.npy", std.values)

    # Save feature indices (for reference)
    with open(OUT_DIR / "features.json", "w") as f:
        json.dump(list(range(train.shape[1])), f, indent=2)

    print("\nSaved:")
    print(" train.npy shape:", train.shape)
    print(" test.npy shape:", test.shape)
    print(" features:", train.shape[1])
    print(" mean.npy")
    print(" std.npy")
    print(" features.json")

    print("\nReady for windowing ✔")


if __name__ == "__main__":
    main()