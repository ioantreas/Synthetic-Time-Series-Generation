import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MSE_KEYS = ("mse_missing", "missing_mse", "mse")
CRPS_KEYS = ("crps_missing", "missing_crps", "crps")
TEXT_PATTERN = re.compile(r"(?im)^\s*(mse_missing|missing_mse|mse|crps_missing|missing_crps|crps)\s*(?:[:=]|is)\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$")


def scalar_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Metric value is not scalar")
    return float(array.reshape(-1)[0])


def recursive_metric(data: Any, keys: tuple[str, ...]) -> float:
    if isinstance(data, dict):
        lowered = {str(key).lower(): value for key, value in data.items()}
        for key in keys:
            if key.lower() in lowered:
                return scalar_float(lowered[key.lower()])
        for value in data.values():
            try:
                return recursive_metric(value, keys)
            except (KeyError, TypeError, ValueError):
                pass
    elif isinstance(data, (list, tuple)):
        for value in data:
            try:
                return recursive_metric(value, keys)
            except (KeyError, TypeError, ValueError):
                pass
    raise KeyError(keys)


def parse_object(data: Any, path: Path) -> dict[str, float]:
    try:
        return {"mse": recursive_metric(data, MSE_KEYS), "crps": recursive_metric(data, CRPS_KEYS)}
    except (KeyError, TypeError, ValueError):
        pass
    if isinstance(data, (list, tuple)) and len(data) >= 3:
        try:
            return {"mse": scalar_float(data[0]), "crps": scalar_float(data[2])}
        except (TypeError, ValueError):
            pass
    raise ValueError(f"Could not identify MSE and CRPS in {path}; expected named keys or [MSE, MAE, CRPS]")


def read_metrics(path: Path) -> dict[str, float]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as file:
            return parse_object(json.load(file), path)
    if suffix in {".txt", ".log"}:
        values: dict[str, float] = {}
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, value in TEXT_PATTERN.findall(text):
            values["crps" if "crps" in name.lower() else "mse"] = float(value)
        missing = {"mse", "crps"} - values.keys()
        if missing:
            raise ValueError(f"Missing {sorted(missing)}")
        return values
    if suffix in {".pk", ".pkl", ".pickle"}:
        with path.open("rb") as file:
            return parse_object(pickle.load(file), path)
    if suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return parse_object({key: data[key] for key in data.files}, path)
    if suffix == ".npy":
        return parse_object(np.load(path, allow_pickle=True), path)
    raise ValueError(f"Unsupported format: {suffix}")

def read_timing(path: Path) -> float:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return float(data["total_method_time_sec"])

def metric_candidates(seed_dir: Path) -> list[Path]:
    patterns = ("metrics.json", "metrics.txt", "metrics.log", "result_nsample*.pk", "result_nsample*.pkl", "result*.pk", "result*.pkl", "metrics.npz", "metrics.npy")
    output: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(seed_dir.rglob(pattern)):
            if path not in seen:
                output.append(path)
                seen.add(path)
    return output

def timing_candidates(seed_dir: Path) -> list[Path]:
    return sorted(seed_dir.rglob("timing.json"))

def discover_runs(root: Path, method: str) -> list[tuple[str, str, int, Path]]:
    pattern = "*/full/*/seed_*" if method == "csdi" else "*/*/seed_*"
    runs = []
    for seed_dir in sorted(root.glob(pattern)):
        if not seed_dir.is_dir():
            continue
        parts = seed_dir.relative_to(root).parts
        if method == "csdi":
            if len(parts) != 4 or parts[1] != "full":
                continue
            dataset, _, scenario, seed_folder = parts
        else:
            if len(parts) != 3:
                continue
            dataset, scenario, seed_folder = parts
        match = re.fullmatch(r"seed_(\d+)", seed_folder)
        if match:
            runs.append((dataset, scenario, int(match.group(1)), seed_dir))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CSDI or SSSD-S4 MSE/CRPS across inference seeds")
    parser.add_argument("--method", choices=("csdi", "sssd"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    runs = discover_runs(root, args.method)
    if not runs:
        expected = "dataset/full/scenario/seed_N" if args.method == "csdi" else "dataset/scenario/seed_N"
        raise FileNotFoundError(f"No {expected} directories found under {root}")

    rows = []
    for dataset, scenario, seed, seed_dir in runs:
        candidates = metric_candidates(seed_dir)
        if not candidates:
            print(f"Skipping {seed_dir}: no supported metric file found")
            continue
        errors = []
        for path in candidates:
            try:
                metrics = read_metrics(path)

                time = np.nan
                timing_path = ""
                if args.method == "csdi":
                    timings = timing_candidates(seed_dir)
                    if not timings:
                        raise FileNotFoundError(f"No timing.json found under {seed_dir}")
                    timing_path = str(timings[0])
                    time = read_timing(timings[0])

                rows.append({
                    "dataset": dataset,
                    "scenario": scenario,
                    "seed": seed,
                    "mse": metrics["mse"],
                    "crps": metrics["crps"],
                    "time": time,
                    "metrics_path": str(path),
                    "timing_path": timing_path,
                })
                break
            except Exception as error:
                errors.append(f"{path.name}: {error}")
        else:
            print(f"Skipping {seed_dir}: {' | '.join(errors)}")

    if not rows:
        raise RuntimeError("No valid metric files were parsed")

    raw_df = pd.DataFrame(rows).sort_values(["dataset", "scenario", "seed"])
    summary_df = raw_df.groupby(["dataset", "scenario"], as_index=False).agg(
        num_seeds=("seed", "nunique"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        crps_mean=("crps", "mean"),
        crps_std=("crps", "std"),
        time_mean=("time", "mean"),
        time_std=("time", "std"),
    ).sort_values(["dataset", "scenario"])

    prefix = "csdi" if args.method == "csdi" else "sssd_s4"
    summary_path = root / f"{prefix}_seed_statistics.csv"
    raw_path = root / f"{prefix}_seed_statistics_per_seed.csv"
    summary_df.to_csv(summary_path, index=False)
    raw_df.to_csv(raw_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"\nSaved summary:  {summary_path}")
    print(f"Saved per-seed: {raw_path}")

    incomplete = summary_df[summary_df["num_seeds"] != 5]
    if not incomplete.empty:
        print("\nWarning: groups without exactly five seeds:")
        print(incomplete[["dataset", "scenario", "num_seeds"]].to_string(index=False))


if __name__ == "__main__":
    main()