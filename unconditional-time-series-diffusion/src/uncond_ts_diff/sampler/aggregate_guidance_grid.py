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

TIMING_PATTERN = re.compile(
    r"^(anchor_time_sec|weight_time_sec|inference_time_sec|"
    r"inference_time_per_imputation_sec|total_method_time_sec):\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def read_primary_metrics(path: Path) -> dict[str, float]:
    full_text = path.read_text(encoding="utf-8")

    # Read MSE/CRPS only from:
    # TARGET/GUIDED SILO: DIFFUSION vs REAL
    primary_text = full_text

    next_section = primary_text.find(
        "=== TARGET/GUIDED SILO: DIFFUSION vs AE ==="
    )

    if next_section != -1:
        primary_text = primary_text[:next_section]

    values = {
        name: float(value)
        for name, value in METRIC_PATTERN.findall(primary_text)
    }

    required = {"mse_missing", "crps_missing"}
    missing = required - values.keys()

    if missing:
        raise ValueError(f"Missing {sorted(missing)} in {path}")

    timing = {
        name: float(value)
        for name, value in TIMING_PATTERN.findall(full_text)
    }

    values.update(timing)

    return values


def parse_structure(
    metrics_path: Path,
    root: Path,
) -> dict[str, object]:
    relative = metrics_path.relative_to(root)
    parts = relative.parts

    # Current anchor-ablation structure:
    #
    # dataset/seed_N/anchor_type/variance/scenario/metrics.txt
    #
    # Example:
    #
    # appliances/seed_1/interpolation/variance/random/metrics.txt

    if len(parts) != 6:
        raise ValueError(
            f"Unexpected path structure: {relative}"
        )

    dataset = parts[0]
    seed_folder = parts[1]
    anchor_type = parts[2]
    mask_type = parts[3]
    scenario = parts[4]

    seed_match = re.fullmatch(r"seed_(\d+)", seed_folder)

    if seed_match is None:
        raise ValueError(
            f"Expected seed_N as second folder in {relative}"
        )

    if anchor_type not in {
        "recovered",
        "interpolation",
        "oracle",
    }:
        raise ValueError(
            f"Unknown anchor type '{anchor_type}' in {relative}"
        )

    if mask_type != "variance":
        raise ValueError(
            f"Expected variance weighting in {relative}, "
            f"found '{mask_type}'"
        )

    seed = int(seed_match.group(1))

    return {
        "dataset": dataset,
        "scenario": scenario,
        "anchor_type": anchor_type,
        "mask_type": mask_type,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "../../../results/inference/name_100_1"
        ),
        help=(
            "Root containing "
            "dataset/seed_N/anchor_type/variance/scenario folders"
        ),
    )

    args = parser.parse_args()

    root = args.root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {root}"
        )

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
                    "anchor_time_sec": metrics.get(
                        "anchor_time_sec",
                        float("nan"),
                    ),
                    "weight_time_sec": metrics.get(
                        "weight_time_sec",
                        float("nan"),
                    ),
                    "inference_time_sec": metrics.get(
                        "inference_time_sec",
                        float("nan"),
                    ),
                    "inference_time_per_imputation_sec": metrics.get(
                        "inference_time_per_imputation_sec",
                        float("nan"),
                    ),
                    "total_method_time_sec": metrics.get(
                        "total_method_time_sec",
                        float("nan"),
                    ),
                    "metrics_path": str(path),
                }
            )

        except (ValueError, OSError) as error:
            print(f"Skipping {path}: {error}")

    if not rows:
        raise RuntimeError(
            "No valid metric files were parsed."
        )

    raw_df = (
        pd.DataFrame(rows)
        .sort_values(
            [
                "dataset",
                "scenario",
                "anchor_type",
                "seed",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 1. Per-case statistics
    #
    # One row for each:
    #
    # dataset + scenario + anchor type
    #
    # Mean/std are computed over all completed inference seeds.
    # ---------------------------------------------------------

    case_summary_df = (
        raw_df
        .groupby(
            [
                "dataset",
                "scenario",
                "anchor_type",
                "mask_type",
            ],
            as_index=False,
        )
        .agg(
            num_seeds=("seed", "nunique"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            crps_mean=("crps", "mean"),
            crps_std=("crps", "std"),
            anchor_time_mean=("anchor_time_sec", "mean"),
            anchor_time_std=("anchor_time_sec", "std"),
            weight_time_mean=("weight_time_sec", "mean"),
            weight_time_std=("weight_time_sec", "std"),
            inference_time_mean=("inference_time_sec", "mean"),
            inference_time_std=("inference_time_sec", "std"),
            inference_time_per_imputation_mean=(
                "inference_time_per_imputation_sec",
                "mean",
            ),
            inference_time_per_imputation_std=(
                "inference_time_per_imputation_sec",
                "std",
            ),
            total_method_time_mean=(
                "total_method_time_sec",
                "mean",
            ),
            total_method_time_std=(
                "total_method_time_sec",
                "std",
            ),
        )
        .sort_values(
            [
                "dataset",
                "scenario",
                "anchor_type",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 2. Overall summary for each anchor type
    #
    # A complete interpolation/oracle anchor has:
    #
    # 5 datasets x 4 scenarios = 20 cases.
    #
    # The recovered anchor may contain only the single sanity
    # check, which is fine and will be reported separately.
    # ---------------------------------------------------------

    anchor_summary_df = (
        case_summary_df
        .groupby(
            [
                "anchor_type",
                "mask_type",
            ],
            as_index=False,
        )
        .agg(
            num_cases=("crps_mean", "count"),
            mse_mean=("mse_mean", "mean"),
            mse_std_across_cases=("mse_mean", "std"),
            crps_mean=("crps_mean", "mean"),
            crps_std_across_cases=("crps_mean", "std"),
            anchor_time_mean=("anchor_time_mean", "mean"),
            weight_time_mean=("weight_time_mean", "mean"),
            inference_time_mean=("inference_time_mean", "mean"),
            inference_time_per_imputation_mean=(
                "inference_time_per_imputation_mean",
                "mean",
            ),
            total_method_time_mean=(
                "total_method_time_mean",
                "mean",
            ),
        )
        .sort_values("anchor_type")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 3. Scenario-level summary
    #
    # Average across the five datasets for each scenario and
    # anchor type.
    # ---------------------------------------------------------

    scenario_summary_df = (
        case_summary_df
        .groupby(
            [
                "scenario",
                "anchor_type",
                "mask_type",
            ],
            as_index=False,
        )
        .agg(
            num_datasets=("dataset", "nunique"),
            mse_mean=("mse_mean", "mean"),
            crps_mean=("crps_mean", "mean"),
            inference_time_mean=("inference_time_mean", "mean"),
            total_method_time_mean=(
                "total_method_time_mean",
                "mean",
            ),
        )
        .sort_values(
            [
                "scenario",
                "anchor_type",
            ]
        )
        .reset_index(drop=True)
    )

    raw_output = root / "anchor_ablation_per_seed.csv"
    case_output = root / "anchor_ablation_case_statistics.csv"
    anchor_output = root / "anchor_ablation_summary.csv"
    scenario_output = root / "anchor_ablation_scenario_summary.csv"

    raw_df.to_csv(
        raw_output,
        index=False,
    )

    case_summary_df.to_csv(
        case_output,
        index=False,
    )

    anchor_summary_df.to_csv(
        anchor_output,
        index=False,
    )

    scenario_summary_df.to_csv(
        scenario_output,
        index=False,
    )

    print()
    print(f"Found {len(metric_files)} metrics files.")
    print(f"Parsed {len(raw_df)} successful runs.")
    print()

    print(f"Saved per-seed:     {raw_output}")
    print(f"Saved per-case:     {case_output}")
    print(f"Saved per-anchor:   {anchor_output}")
    print(f"Saved per-scenario: {scenario_output}")

    # ---------------------------------------------------------
    # Progress information
    #
    # Only interpolation and oracle are expected to contain
    # the complete 20-case anchor ablation.
    #
    # Recovered is only a sanity-check run and is therefore
    # deliberately excluded from the completeness warning.
    # ---------------------------------------------------------

    expected_cases_per_anchor = 20

    ablation_anchors = anchor_summary_df[
        anchor_summary_df["anchor_type"].isin(
            ["interpolation", "oracle"]
        )
    ]

    incomplete = ablation_anchors[
        ablation_anchors["num_cases"]
        < expected_cases_per_anchor
    ]

    if not incomplete.empty:
        print()
        print(
            "Incomplete anchor ablations "
            "(this is allowed while runs are still running):"
        )

        print(
            incomplete[
                [
                    "anchor_type",
                    "num_cases",
                ]
            ].to_string(index=False)
        )

    print()
    print("Current overall CRPS by anchor type:")
    print()

    print(
        anchor_summary_df[
            [
                "anchor_type",
                "num_cases",
                "crps_mean",
                "anchor_time_mean",
                "inference_time_mean",
                "total_method_time_mean",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()