#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import pandas as pd


MSE_KEYS = ("mse_missing", "mse", "MSE")
CRPS_KEYS = ("crps_missing", "crps", "CRPS")


def find_metric(data: object, possible_keys: tuple[str, ...]) -> float:
    """Recursively find the first matching numeric metric."""

    if isinstance(data, dict):
        for key in possible_keys:
            if key in data:
                return float(data[key])

        for value in data.values():
            try:
                return find_metric(value, possible_keys)
            except KeyError:
                pass

    raise KeyError(f"Could not find any of {possible_keys}")


def read_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        "mse": find_metric(data, MSE_KEYS),
        "crps": find_metric(data, CRPS_KEYS),
    }


def parse_structure(metrics_path: Path, root: Path) -> dict[str, object]:
    relative = metrics_path.relative_to(root)
    parts = relative.parts

    # Expected:
    # dataset/scenario/seed_N/metrics.json
    if len(parts) != 4:
        raise ValueError(f"Unexpected path structure: {relative}")

    dataset, scenario, seed_folder, filename = parts

    if filename != "metrics.json":
        raise ValueError(f"Unexpected metric filename: {relative}")

    match = re.fullmatch(r"seed_(\d+)", seed_folder)
    if match is None:
        raise ValueError(f"Expected seed_N as third folder in {relative}")

    return {
        "dataset": dataset,
        "scenario": scenario,
        "seed": int(match.group(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("OUTPUT"), help="Root following OUTPUT/dataset/scenario/seed_N/metrics.json")
    args = parser.parse_args()

    root = args.root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    metric_files = sorted(root.rglob("metrics.json"))

    if not metric_files:
        raise FileNotFoundError(f"No metrics.json files found under {root}")

    rows = []

    for path in metric_files:
        try:
            metadata = parse_structure(path, root)
            metrics = read_metrics(path)
            rows.append({**metadata, **metrics, "metrics_path": str(path)})
        except (ValueError, KeyError, TypeError, json.JSONDecodeError, OSError) as error:
            print(f"Skipping {path}: {error}")

    if not rows:
        raise RuntimeError("No valid metric files were parsed.")

    raw_df = pd.DataFrame(rows).sort_values(["dataset", "scenario", "seed"])

    summary_df = raw_df.groupby(["dataset", "scenario"], as_index=False).agg(
        num_seeds=("seed", "nunique"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        crps_mean=("crps", "mean"),
        crps_std=("crps", "std"),
    ).sort_values(["dataset", "scenario"])

    raw_output = root / "diffusion_ts_seed_statistics_per_seed.csv"
    summary_output = root / "diffusion_ts_seed_statistics.csv"

    raw_df.to_csv(raw_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print(f"Found {len(metric_files)} metrics files.")
    print(f"Parsed {len(raw_df)} successful runs.")
    print(f"Saved per-seed results: {raw_output}")
    print(f"Saved aggregate results: {summary_output}")

    print("\nAggregate results:")
    print(summary_df.to_string(index=False))

    incomplete = summary_df[summary_df["num_seeds"] != 5]

    if not incomplete.empty:
        print("\nWarning: groups without exactly five seeds:")
        print(incomplete[["dataset", "scenario", "num_seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()