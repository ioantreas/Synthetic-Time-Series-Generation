import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def quantile_loss(target, forecast, q, eval_points):
    return 2.0 * np.sum(
        np.abs(
            (forecast - target)
            * eval_points
            * ((target <= forecast).astype(float) - q)
        )
    )


def compute_crps(target, samples, eval_points):
    # target:      [N,T,C]
    # samples:     [N,S,T,C]
    # eval_points: [N,T,C], 1 = missing/evaluate
    quantiles = np.arange(0.05, 1.0, 0.05)

    denom = np.sum(np.abs(target * eval_points))
    if denom == 0:
        return float("nan")

    crps = 0.0

    for q in quantiles:
        forecast = np.quantile(samples, q, axis=1)
        crps += quantile_loss(target, forecast, q, eval_points) / denom

    return float(crps / len(quantiles))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nsample", type=int, default=10)
    args = parser.parse_args()

    root = args.root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(root)

    seed_dirs = sorted(root.glob("*/full/*/seed_*"))

    if not seed_dirs:
        raise RuntimeError(
            f"No dataset/full/scenario/seed_N directories found under {root}"
        )

    updated = 0

    for seed_dir in seed_dirs:
        results_path = seed_dir / f"generated_outputs_nsample{args.nsample}.pk"

        if not results_path.exists():
            print(f"SKIP: {results_path} not found")
            continue

        with results_path.open("rb") as f:
            (
                all_generated_samples,
                all_target,
                all_evalpoint,
                _all_observed_point,
                _all_observed_time,
                _scaler,
                _mean_scaler,
            ) = pickle.load(f)

        all_generated_samples = to_numpy(all_generated_samples)
        all_target = to_numpy(all_target)
        all_evalpoint = to_numpy(all_evalpoint)

        if all_generated_samples.ndim != 4:
            raise ValueError(
                f"{results_path}: expected predictions [N,S,T,C], "
                f"got {all_generated_samples.shape}"
            )

        if all_target.shape != all_evalpoint.shape:
            raise ValueError(
                f"{results_path}: target/eval mask mismatch: "
                f"{all_target.shape} vs {all_evalpoint.shape}"
            )

        if all_generated_samples.shape[0] != all_target.shape[0]:
            raise ValueError(f"{results_path}: sample count mismatch")

        if all_generated_samples.shape[2:] != all_target.shape[1:]:
            raise ValueError(
                f"{results_path}: prediction/target shape mismatch: "
                f"{all_generated_samples.shape} vs {all_target.shape}"
            )

        # Same point estimator as all other methods.
        pred = np.median(all_generated_samples, axis=1)

        # CSDI eval_points: 1 = missing/evaluated.
        missing = all_evalpoint.astype(bool)

        if not np.any(missing):
            raise ValueError(f"{results_path}: no evaluation points")

        mse = float(
            np.mean(
                (pred[missing] - all_target[missing]) ** 2
            )
        )

        mae = float(
            np.mean(
                np.abs(pred[missing] - all_target[missing])
            )
        )

        # Same CSDI-style normalized quantile CRPS used by the other methods.
        crps = compute_crps(
            all_target,
            all_generated_samples,
            all_evalpoint,
        )

        metrics = {
            "mse": mse,
            "mae": mae,
            "crps": crps,
        }

        out_path = seed_dir / "metrics.json"

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)

        relative = seed_dir.relative_to(root)

        print(
            f"{relative}: "
            f"MSE={mse:.9f}  "
            f"MAE={mae:.9f}  "
            f"CRPS={crps:.9f}"
        )

        updated += 1

    print(f"\nUpdated {updated} runs.")


if __name__ == "__main__":
    main()