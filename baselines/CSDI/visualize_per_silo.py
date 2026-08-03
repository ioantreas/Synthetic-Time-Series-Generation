import argparse
import pickle
import json
import numpy as np
from pathlib import Path


def quantile_loss(target, forecast, q, mask):
    return 2.0 * np.sum(
        np.abs((forecast - target) * mask * (((target <= forecast).astype(float)) - q))
    )


def calc_quantile_crps(target, forecast, mask):
    quantiles = np.arange(0.05, 1.0, 0.05)

    denom = np.sum(np.abs(target * mask))
    if denom == 0:
        return np.nan

    crps = 0.0

    for q in quantiles:
        q_pred = np.quantile(
            forecast,
            q,
            axis=1,  # sample dimension
        )

        q_loss = quantile_loss(
            target,
            q_pred,
            q,
            mask,
        )

        crps += q_loss / denom

    return crps / len(quantiles)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="generated_outputs_nsampleX.pk",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.results, "rb") as f:
        (
            all_generated_samples,
            all_target,
            all_evalpoint,
            all_observed_point,
            all_observed_time,
            scaler,
            mean_scaler,
        ) = pickle.load(f)

    if hasattr(all_generated_samples, "cpu"):
        all_generated_samples = all_generated_samples.cpu().numpy()

    if hasattr(all_target, "cpu"):
        all_target = all_target.cpu().numpy()

    if hasattr(all_evalpoint, "cpu"):
        all_evalpoint = all_evalpoint.cpu().numpy()

    # Mean prediction for MSE / MAE
    pred = np.mean(all_generated_samples, axis=1)

    # =====================================================
    # EDIT THESE FOR YOUR DATASET
    # =====================================================
    silos = {
        "silo1": [0, 1, 2, 3, 41, 42],
        "silo2": [4, 5, 6, 7, 8, 9, 43, 44],
        "silo3": [10, 11, 12, 13, 14, 45],
        "silo4": [15, 16, 17, 18, 19, 46],
        "silo5": [
            20, 21, 22, 23, 24, 25, 26, 27, 28,
            29, 30, 31, 32, 33, 34, 35, 36, 37,
            38, 39, 40, 47, 48, 49, 50, 51,
        ],
    }
    # =====================================================

    results = {}

    for silo_name, feature_idx in silos.items():

        silo_dir = out_dir / silo_name
        silo_dir.mkdir(parents=True, exist_ok=True)

        target = all_target[:, :, feature_idx]
        forecast = all_generated_samples[:, :, :, feature_idx]
        point_pred = pred[:, :, feature_idx]
        mask = all_evalpoint[:, :, feature_idx]

        valid = mask.astype(bool)

        mse = np.mean(
            (point_pred[valid] - target[valid]) ** 2
        )

        mae = np.mean(
            np.abs(
                point_pred[valid] - target[valid]
            )
        )

        crps = calc_quantile_crps(
            target,
            forecast,
            mask,
        )

        # Save metrics for this silo
        metrics = {
            "num_features": len(feature_idx),
            "feature_indices": feature_idx,
            "mse": float(mse),
            "mae": float(mae),
            "crps": float(crps),
        }

        with open(silo_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # -------------------------------------------------
        # Plot first 3 features of this silo
        # -------------------------------------------------
        sample_idx = 0  # change if desired

        num_plot = min(3, len(feature_idx))

        for local_idx in range(num_plot):
            global_idx = feature_idx[local_idx]

            true = target[sample_idx, :, local_idx]
            prediction = point_pred[sample_idx, :, local_idx]
            eval_mask = mask[sample_idx, :, local_idx]

            prediction = prediction.copy()
            prediction[eval_mask == 0] = np.nan

            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 4))

            plt.plot(
                true,
                "--",
                linewidth=2,
                color="black",
                label="ground truth",
            )

            plt.plot(
                prediction,
                linewidth=3,
                color="red",
                label="prediction",
            )

            plt.title(
                f"{silo_name} - feature {global_idx}"
            )

            plt.legend()
            plt.tight_layout()

            plt.savefig(
                silo_dir / f"feature_{global_idx}.png"
            )

            plt.close()

        print(f"{silo_name}")
        print(f"  MSE  : {mse:.6f}")
        print(f"  MAE  : {mae:.6f}")
        print(f"  CRPS : {crps:.6f}")
        print()

    with open(out_dir / "metrics_per_silo.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results to {out_dir / 'metrics_per_silo.json'}")


if __name__ == "__main__":
    main()