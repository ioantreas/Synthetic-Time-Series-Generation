import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("../../data/raw/air_quality/PRSA_Data_Aotizhongxin_20130301-20170228.csv")
OUT_DIR = Path("../../data/processed/air_quality/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.8
STATION = "Aotizhongxin"

# Environmental sensor features
FEATURES = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM"
]


def main():

    print("Loading CSV...")
    df = pd.read_csv(RAW_DIR)

    print("Original shape:", df.shape)

    # --------------------------------
    # Filter station
    # --------------------------------
    if "station" in df.columns:
        df = df[df["station"] == STATION]
        print("Filtered station:", STATION)

    # --------------------------------
    # Create datetime index
    # --------------------------------
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("datetime")
    df = df.set_index("datetime")

    # --------------------------------
    # Cyclic time features
    # --------------------------------
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    TIME_FEATURES = [
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos"
    ]

    # --------------------------------
    # Select final features
    # --------------------------------
    df = df[FEATURES + TIME_FEATURES]

    print("Initial features:", df.columns.tolist())
    print("Timesteps:", len(df))

    # --------------------------------
    # Handle missing values
    # --------------------------------
    df = df.ffill().bfill()

    # --------------------------------
    # Remove constant / near-constant channels
    # --------------------------------
    stds = df.std()
    constant_cols = stds[stds < 1e-8].index.tolist()

    if constant_cols:
        print("Removing constant columns:", constant_cols)
        df = df.drop(columns=constant_cols)

    print("Final features:", df.columns.tolist())

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
    # Save arrays
    # --------------------------------
    np.save(OUT_DIR / "train.npy", train)
    np.save(OUT_DIR / "test.npy", test)

    # Save normalization stats for later denormalization
    np.save(OUT_DIR / "mean.npy", mean.values)
    np.save(OUT_DIR / "std.npy", std.values)

    # Save feature names
    with open(OUT_DIR / "features.json", "w") as f:
        json.dump(df.columns.tolist(), f, indent=2)

    print("\nSaved:")
    print(" train:", train.shape)
    print(" test:", test.shape)
    print(" features:", len(df.columns))


if __name__ == "__main__":
    main()