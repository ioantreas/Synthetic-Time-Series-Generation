#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd


METRIC_PATTERN = re.compile(
    r"^(mse_missing|crps_missing):\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def read_primary_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")

    # Read only the first section:
    # TARGET/GUIDED SILO: DIFFUSION vs REAL
    next_section = text.find(
        "=== TARGET/GUIDED SILO: DIFFUSION vs AE ==="
    )
    if next_section != -1:
        text = text[:next_section]

    values = {
        name: float(value)
        for name, value in METRIC_PATTERN.findall(text)
    }

    required = {"mse_missing", "crps_missing"}
    missing = required - values.keys()

    if missing:
        raise ValueError(
            f"Missing {sorted(missing)} in {path}"
        )

    return values


def parse_structure(
    metrics_path: Path,
    root: Path,
) -> dict[str, object]:
    relative = metrics_path.relative_to(root)
    parts = relative.parts

    # Expected beginning:
    # dataset/scenario/guidance/seed_N/...
    if len(parts) < 5:
        raise ValueError(
            f"Unexpected path structure: {relative}"
        )

    dataset = parts[0]
    scenario = parts[1]
    guidance = parts[2]
    seed_folder = parts[3]

    match = re.fullmatch(r"seed_(\d+)", seed_folder)
    if match is None:
        raise ValueError(
            f"Expected seed_N as fourth folder in {relative}"
        )

    return {
        "dataset": dataset,
        "scenario": scenario,
        "guidance": guidance,
        "seed": int(match.group(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("../../../results/inference/spinning"),
        help="Root containing dataset/scenario/guidance/seed folders",
    )
    args = parser.parse_args()

    root = args.root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    metric_files = sorted(root.rglob("metrics.txt"))

    if not metric_files:
        raise FileNotFoundError(
            f"No metrics.txt files found under {root}"
        )

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
                    "metrics_path": str(path),
                }
            )

        except (ValueError, OSError) as error:
            print(f"Skipping {path}: {error}")

    if not rows:
        raise RuntimeError("No valid metric files were parsed.")

    raw_df = pd.DataFrame(rows).sort_values(
        ["dataset", "scenario", "guidance", "seed"]
    )

    # ddof=1 gives the usual sample standard deviation across seeds.
    summary_df = (
        raw_df
        .groupby(
            ["dataset", "scenario", "guidance"],
            as_index=False,
        )
        .agg(
            num_seeds=("seed", "nunique"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            crps_mean=("crps", "mean"),
            crps_std=("crps", "std"),
        )
        .sort_values(
            ["dataset", "scenario", "guidance"]
        )
    )

    summary_output = root / "guidance_seed_statistics.csv"
    raw_output = root / "guidance_seed_statistics_per_seed.csv"

    summary_df.to_csv(summary_output, index=False)
    raw_df.to_csv(raw_output, index=False)

    print(f"Found {len(metric_files)} metrics files.")
    print(f"Parsed {len(raw_df)} successful runs.")
    print(f"Saved summary:  {summary_output}")
    print(f"Saved per-seed: {raw_output}")

    incomplete = summary_df[summary_df["num_seeds"] != 5]
    if not incomplete.empty:
        print("\nWarning: groups without exactly five seeds:")
        print(
            incomplete[
                ["dataset", "scenario", "guidance", "num_seeds"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()