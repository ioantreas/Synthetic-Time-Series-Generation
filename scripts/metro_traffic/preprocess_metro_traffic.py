import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("../../data/raw/metro_traffic/")
OUT_DIR = Path("../../data/processed/metro_traffic/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.8


def main():
    print("Loading CSV...")
    df = pd.read_csv(RAW_DIR / "data.csv")

    print("Original shape:", df.shape)

    # Parse datetime
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time")
    df = df.set_index("date_time")

    # --------------------------------
    # Time features
    # --------------------------------
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)

    # --------------------------------
    # Holiday flag
    # --------------------------------
    if "holiday" in df.columns:
        df["is_holiday"] = (df["holiday"] != "None").astype(float)

    # --------------------------------
    # Select useful features
    # --------------------------------
    FEATURES = [
        "traffic_volume",
        "temp",
        "rain_1h",
        "snow_1h",
        "clouds_all",
        "is_holiday",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos"
    ]

    df = df[FEATURES]

    print("Final features:", df.columns.tolist())
    print("Timesteps:", len(df))

    # --------------------------------
    # Fill missing
    # --------------------------------
    df = df.ffill().bfill()

    # --------------------------------
    # Remove constant channels
    # --------------------------------
    stds = df.std()
    constant_cols = stds[stds < 1e-8].index.tolist()

    if constant_cols:
        print("Removing constant columns:", constant_cols)
        df = df.drop(columns=constant_cols)

    # --------------------------------
    # Normalize
    # --------------------------------
    mean = df.mean()
    std = df.std() + 1e-8
    df_norm = (df - mean) / std

    # --------------------------------
    # Train/test split
    # --------------------------------
    split = int(len(df_norm) * TRAIN_FRAC)

    train = df_norm.iloc[:split].values.astype("float32")
    test = df_norm.iloc[split:].values.astype("float32")

    # --------------------------------
    # Save
    # --------------------------------
    np.save(OUT_DIR / "train.npy", train)
    np.save(OUT_DIR / "test.npy", test)

    np.save(OUT_DIR / "mean.npy", mean.values)
    np.save(OUT_DIR / "std.npy", std.values)

    with open(OUT_DIR / "features.json", "w") as f:
        json.dump(df.columns.tolist(), f, indent=2)

    print("\nSaved:")
    print(" train:", train.shape)
    print(" test:", test.shape)
    print(" features:", len(df.columns))


if __name__ == "__main__":
    main()