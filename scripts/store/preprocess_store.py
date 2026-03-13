import json
from pathlib import Path
import numpy as np
import pandas as pd

RAW_TRAIN = Path("../../data/raw/store/train.csv")
RAW_STORE = Path("../../data/raw/store/store.csv")

OUT_DIR = Path("../../data/processed/store/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FRAC = 0.8


def main():

    print("Loading CSVs...")
    train_df = pd.read_csv(RAW_TRAIN)
    store_df = pd.read_csv(RAW_STORE)

    print("Merging store info...")
    df = train_df.merge(store_df, on="Store", how="left")

    print("Parsing dates...")
    df["Date"] = pd.to_datetime(df["Date"])

    # --------------------------------
    # Time features
    # --------------------------------
    df["dow"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)

    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    # --------------------------------
    # Drop unused columns
    # --------------------------------
    drop_cols = [
        "Customers",
        "Date",
        "dow",
        "month"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # --------------------------------
    # One-hot encode categoricals
    # --------------------------------
    cat_cols = ["StoreType", "Assortment", "StateHoliday"]

    df = pd.get_dummies(
        df,
        columns=[c for c in cat_cols if c in df.columns],
        drop_first=True
    )

    # --------------------------------
    # Keep numeric features
    # --------------------------------
    df = df.select_dtypes(include=["number"])

    print("Final features:", df.columns.tolist())
    print("Rows:", len(df))

    # --------------------------------
    # Fill missing
    # --------------------------------
    df = df.ffill().bfill()

    # --------------------------------
    # Remove constant columns
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
    # Chronological split
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