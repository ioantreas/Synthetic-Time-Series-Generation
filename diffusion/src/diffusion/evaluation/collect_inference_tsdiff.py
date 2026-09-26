import argparse
import json
import re
from pathlib import Path

import pandas as pd


def read_primary_metrics(path: Path) -> dict[str, float]:
    with path.open("r", encoding="utf-8") as file:
        values = json.load(file)

    required = {
        "mse_missing",
        "crps_missing",
        "inference_time_sec",
        "inference_time_per_imputation_sec",
    }
    missing = required - values.keys()

    if missing:
        raise ValueError(f"Missing {sorted(missing)} in {path}")

    return {
        "mse_missing": float(values["mse_missing"]),
        "crps_missing": float(values["crps_missing"]),
        "inference_time_sec": float(values["inference_time_sec"]),
        "inference_time_per_imputation_sec": float(
            values["inference_time_per_imputation_sec"]
        ),
    }


def parse_structure(metrics_path: Path, root: Path) -> dict[str, object]:
    relative = metrics_path.relative_to(root)
    parts = relative.parts

    # Expected: dataset/scenario/seed_N/metrics.json
    if len(parts) != 4:
        raise ValueError(f"Unexpected path structure: {relative}")

    dataset, scenario, seed_folder, filename = parts

    if filename != "metrics.json":
        raise ValueError(f"Unexpected filename: {relative}")

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
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("../../../results/inference/tsdiff"),
        help="Root containing dataset/scenario/seed_N/metrics.json",
    )
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
            metrics = read_primary_metrics(path)

            rows.append(
                {
                    **metadata,
                    "mse": metrics["mse_missing"],
                    "crps": metrics["crps_missing"],
                    "inference_time_sec": metrics["inference_time_sec"],
                    "inference_time_per_imputation_sec": metrics[
                        "inference_time_per_imputation_sec"
                    ],
                    "metrics_path": str(path),
                }
            )
        except (ValueError, OSError, json.JSONDecodeError) as error:
            print(f"Skipping {path}: {error}")

    if not rows:
        raise RuntimeError("No valid metric files were parsed.")

    raw_df = pd.DataFrame(rows).sort_values(["dataset", "scenario", "seed"])

    summary_df = (
        raw_df.groupby(["dataset", "scenario"], as_index=False)
        .agg(
            num_seeds=("seed", "nunique"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            crps_mean=("crps", "mean"),
            crps_std=("crps", "std"),
            inference_time_sec_mean=("inference_time_sec", "mean"),
            inference_time_sec_std=("inference_time_sec", "std"),
            inference_time_per_imputation_sec_mean=(
                "inference_time_per_imputation_sec",
                "mean",
            ),
            inference_time_per_imputation_sec_std=(
                "inference_time_per_imputation_sec",
                "std",
            ),
        )
        .sort_values(["dataset", "scenario"])
    )

    raw_output = root / "tsdiff_seed_statistics_per_seed.csv"
    summary_output = root / "tsdiff_seed_statistics.csv"

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
        print(
            incomplete[
                ["dataset", "scenario", "num_seeds"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()