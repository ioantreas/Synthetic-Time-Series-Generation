import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW_PATH = Path("../../data/raw/tourism/tourism.csv")

OUT_DIR = Path("../../data/processed/tourism/")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading Tourism CSV...")
    df = pd.read_csv(RAW_PATH)

    print("Original shape:", df.shape)

    # Drop useless index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Convert Quarter to datetime
    df["Quarter"] = pd.to_datetime(df["Quarter"])

    # Create channel identifier: Region_Purpose
    df["Channel"] = df["Region"] + "_" + df["Purpose"]

    # Pivot: time × (region-purpose)
    df_wide = df.pivot(index="Quarter", columns="Channel", values="Trips")
    df_wide = df_wide.sort_index()

    print("After pivot shape:", df_wide.shape)

    # Fill missing values
    df_wide = df_wide.ffill().bfill()

    # Z-score normalization (global, since no split)
    mean = df_wide.mean()
    std = df_wide.std() + 1e-8
    df_norm = (df_wide - mean) / std

    data = df_norm.values.astype("float32")

    # Save full dataset (NO SPLIT)
    np.save(OUT_DIR / "train.npy", data)

    # Save metadata
    with open(OUT_DIR / "features.json", "w") as f:
        json.dump(df_wide.columns.tolist(), f, indent=2)

    with open(OUT_DIR / "norm_stats.json", "w") as f:
        json.dump(
            {
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            f,
            indent=2,
        )

    print("Saved:")
    print(" data:", data.shape)


if __name__ == "__main__":
    main()