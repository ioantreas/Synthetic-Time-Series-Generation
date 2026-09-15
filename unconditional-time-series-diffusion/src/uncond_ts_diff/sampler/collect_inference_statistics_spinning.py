import argparse
import re
from pathlib import Path

import pandas as pd


PRIMARY_METRIC_PATTERN = re.compile(
    r"^(mse_missing|crps_missing):\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)

COMM_PATTERN = re.compile(
    r"^(communication_rounds|"
    r"communication_server_to_client_bytes|"
    r"communication_client_to_server_bytes|"
    r"communication_total_bytes|"
    r"communication_per_imputation_bytes):\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


def read_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")

    # Only parse the first evaluation block for MSE / CRPS.
    next_section = text.find("=== TARGET/GUIDED SILO: DIFFUSION vs AE ===")
    primary_text = text[:next_section] if next_section != -1 else text

    primary = {
        name: float(value)
        for name, value in PRIMARY_METRIC_PATTERN.findall(primary_text)
    }

    required_primary = {"mse_missing", "crps_missing"}
    missing_primary = required_primary - primary.keys()
    if missing_primary:
        raise ValueError(f"Missing {sorted(missing_primary)} in {path}")

    # Parse communication section from the full file.
    comm_start = text.find("=== COMMUNICATION ===")
    if comm_start == -1:
        raise ValueError(f"Missing COMMUNICATION section in {path}")

    comm_text = text[comm_start:]
    communication = {
        name: float(value)
        for name, value in COMM_PATTERN.findall(comm_text)
    }

    required_comm = {
        "communication_server_to_client_bytes",
        "communication_client_to_server_bytes",
        "communication_total_bytes",
        "communication_per_imputation_bytes",
    }
    missing_comm = required_comm - communication.keys()
    if missing_comm:
        raise ValueError(f"Missing {sorted(missing_comm)} in {path}")

    return {**primary, **communication}


def parse_structure(metrics_path: Path, root: Path) -> dict[str, object]:
    relative = metrics_path.relative_to(root)
    parts = relative.parts

    # Expected: dataset/seed_N/scenario/metrics.txt
    if len(parts) != 4:
        raise ValueError(f"Unexpected path structure: {relative}")

    dataset, seed_folder, scenario, filename = parts

    if filename != "metrics.txt":
        raise ValueError(f"Unexpected filename: {relative}")

    match = re.fullmatch(r"seed_(\d+)", seed_folder)
    if match is None:
        raise ValueError(f"Expected seed_N as second folder in {relative}")

    return {
        "dataset": dataset,
        "scenario": scenario,
        "seed": int(match.group(1)),
    }


def bytes_to_gib(value: float) -> float:
    return value / (1024 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("../../../results/inference/spinning"),
        help="Root containing dataset/seed_N/scenario/metrics.txt",
    )
    args = parser.parse_args()

    root = args.root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    metric_files = sorted(root.rglob("metrics.txt"))

    if not metric_files:
        raise FileNotFoundError(f"No metrics.txt files found under {root}")

    rows = []

    for path in metric_files:
        try:
            metadata = parse_structure(path, root)
            metrics = read_metrics(path)

            row = {
                **metadata,
                "mse": metrics["mse_missing"],
                "crps": metrics["crps_missing"],
                "client_to_server_bytes": metrics["communication_client_to_server_bytes"],
                "server_to_client_bytes": metrics["communication_server_to_client_bytes"],
                "total_bytes": metrics["communication_total_bytes"],
                "per_imputation_bytes": metrics["communication_per_imputation_bytes"],
                "metrics_path": str(path),
            }

            if "communication_rounds" in metrics:
                row["communication_rounds"] = metrics["communication_rounds"]

            row["client_to_server_gib"] = bytes_to_gib(row["client_to_server_bytes"])
            row["server_to_client_gib"] = bytes_to_gib(row["server_to_client_bytes"])
            row["total_gib"] = bytes_to_gib(row["total_bytes"])
            row["per_imputation_gib"] = bytes_to_gib(row["per_imputation_bytes"])

            rows.append(row)

        except (ValueError, OSError) as error:
            print(f"Skipping {path}: {error}")

    if not rows:
        raise RuntimeError("No valid metric files were parsed.")

    raw_df = pd.DataFrame(rows).sort_values(["dataset", "scenario", "seed"])

    agg_spec = {
        "num_seeds": ("seed", "nunique"),
        "mse_mean": ("mse", "mean"),
        "mse_std": ("mse", "std"),
        "crps_mean": ("crps", "mean"),
        "crps_std": ("crps", "std"),
        "client_to_server_bytes_mean": ("client_to_server_bytes", "mean"),
        "client_to_server_bytes_std": ("client_to_server_bytes", "std"),
        "server_to_client_bytes_mean": ("server_to_client_bytes", "mean"),
        "server_to_client_bytes_std": ("server_to_client_bytes", "std"),
        "total_bytes_mean": ("total_bytes", "mean"),
        "total_bytes_std": ("total_bytes", "std"),
        "per_imputation_bytes_mean": ("per_imputation_bytes", "mean"),
        "per_imputation_bytes_std": ("per_imputation_bytes", "std"),
        "client_to_server_gib_mean": ("client_to_server_gib", "mean"),
        "client_to_server_gib_std": ("client_to_server_gib", "std"),
        "server_to_client_gib_mean": ("server_to_client_gib", "mean"),
        "server_to_client_gib_std": ("server_to_client_gib", "std"),
        "total_gib_mean": ("total_gib", "mean"),
        "total_gib_std": ("total_gib", "std"),
        "per_imputation_gib_mean": ("per_imputation_gib", "mean"),
        "per_imputation_gib_std": ("per_imputation_gib", "std"),
    }

    if "communication_rounds" in raw_df.columns:
        agg_spec["communication_rounds_mean"] = ("communication_rounds", "mean")
        agg_spec["communication_rounds_std"] = ("communication_rounds", "std")

    summary_df = (
        raw_df.groupby(["dataset", "scenario"], as_index=False)
        .agg(**agg_spec)
        .sort_values(["dataset", "scenario"])
    )

    raw_output = root / "spinning_seed_statistics_per_seed.csv"
    summary_output = root / "spinning_seed_statistics.csv"

    raw_df.to_csv(raw_output, index=False)
    summary_df.to_csv(summary_output, index=False)

    print(f"Found {len(metric_files)} metrics files.")
    print(f"Parsed {len(raw_df)} successful runs.")
    print(f"Saved per-seed results: {raw_output}")
    print(f"Saved aggregate results: {summary_output}")

    display_cols = [
        "dataset",
        "scenario",
        "num_seeds",
        "mse_mean",
        "mse_std",
        "crps_mean",
        "crps_std",
        "total_gib_mean",
        "total_gib_std",
        "per_imputation_gib_mean",
        "per_imputation_gib_std",
    ]

    if "communication_rounds_mean" in summary_df.columns:
        display_cols += ["communication_rounds_mean", "communication_rounds_std"]

    print("\nAggregate results:")
    print(summary_df[display_cols].to_string(index=False))

    incomplete = summary_df[summary_df["num_seeds"] != 5]
    if not incomplete.empty:
        print("\nWarning: groups without exactly five seeds:")
        print(incomplete[["dataset", "scenario", "num_seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()
