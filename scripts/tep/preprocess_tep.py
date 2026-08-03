# preprocess_tep.py

import json
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_PATH = Path("../../data/raw/tep/TEP_FaultFree_Training.csv")
TEST_PATH = Path("../../data/raw/tep/TEP_FaultFree_Testing.csv")

OUT_DIR = Path("../../data/processed/tep/")
OUT_DIR.mkdir(parents=True, exist_ok=True)


META_COLS = [
    "faultNumber",
    "simulationRun",
    "sample"
]


def load_dataset(path):

    print(f"\nLoading: {path.name}")

    df = pd.read_csv(path)

    print("Original shape:", df.shape)

    # --------------------------------
    # store run ids BEFORE dropping
    # --------------------------------

    run_ids = df["simulationRun"].values.astype(np.int32)

    # --------------------------------
    # remove metadata columns
    # --------------------------------

    df = df.drop(columns=META_COLS)

    print("Feature shape:", df.shape)

    # --------------------------------
    # numeric conversion
    # --------------------------------

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --------------------------------
    # remove missing rows
    # --------------------------------

    before = len(df)

    valid_mask = ~df.isna().any(axis=1)

    df = df[valid_mask]

    run_ids = run_ids[valid_mask]

    print("Removed rows:", before - len(df))

    return df, run_ids


def main():

    # --------------------------------
    # load train/test
    # --------------------------------

    train_df, train_runs = load_dataset(TRAIN_PATH)
    test_df, test_runs = load_dataset(TEST_PATH)

    print("\nFinal train shape:", train_df.shape)
    print("Final test shape:", test_df.shape)

    # --------------------------------
    # normalize using TRAIN stats
    # --------------------------------

    mean = train_df.mean()
    std = train_df.std() + 1e-8

    train_norm = (train_df - mean) / std
    test_norm = (test_df - mean) / std

    # optional clipping
    train_norm = train_norm.clip(-5, 5)
    test_norm = test_norm.clip(-5, 5)

    # --------------------------------
    # convert
    # --------------------------------

    train = train_norm.values.astype(np.float32)
    test = test_norm.values.astype(np.float32)

    # --------------------------------
    # save arrays
    # --------------------------------

    np.save(OUT_DIR / "train.npy", train)
    np.save(OUT_DIR / "test.npy", test)

    np.save(OUT_DIR / "train_runs.npy", train_runs)
    np.save(OUT_DIR / "test_runs.npy", test_runs)

    np.save(OUT_DIR / "mean.npy", mean.values)
    np.save(OUT_DIR / "std.npy", std.values)

    # --------------------------------
    # save features
    # --------------------------------

    with open(OUT_DIR / "features.json", "w") as f:
        json.dump(train_df.columns.tolist(), f, indent=2)

    print("\nSaved:")
    print("train:", train.shape)
    print("test:", test.shape)
    print("train_runs:", train_runs.shape)
    print("test_runs:", test_runs.shape)
    print("features:", len(train_df.columns))


if __name__ == "__main__":
    main()