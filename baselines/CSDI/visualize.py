import argparse
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def quantile_loss(target, forecast, q, mask):
    return 2.0 * np.sum(
        np.abs((forecast - target) * mask * (((target <= forecast).astype(float)) - q))
    )


def calc_quantile_crps(target, forecast, mask):
    """
    target   : (N, T, C)
    forecast : (N, S, T, C)
    mask     : (N, T, C)   (1 = evaluate, 0 = ignore)

    Matches the quantile-based CRPS approximation used by CSDI.
    """

    quantiles = np.arange(0.05, 1.0, 0.05)

    denom = np.sum(np.abs(target * mask))
    crps = 0.0

    for q in quantiles:
        q_pred = np.quantile(
            forecast,
            q,
            axis=1,  # sample dimension
        )  # -> (N, T, C)

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
        default="plots",
    )

    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--num_channels",
        type=int,
        default=10,
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

    # torch -> numpy
    if hasattr(all_generated_samples, "cpu"):
        all_generated_samples = all_generated_samples.cpu().numpy()

    if hasattr(all_target, "cpu"):
        all_target = all_target.cpu().numpy()

    if hasattr(all_evalpoint, "cpu"):
        all_evalpoint = all_evalpoint.cpu().numpy()

    if hasattr(all_observed_point, "cpu"):
        all_observed_point = all_observed_point.cpu().numpy()

    # Mean prediction across stochastic samples
    pred = np.mean(
        all_generated_samples,
        axis=1,
    )

    N, T, C = pred.shape

    sample_idx = min(args.sample_idx, N - 1)

    print("Prediction shape:", pred.shape)
    print("Target shape:", all_target.shape)
    print("Mask shape:", all_evalpoint.shape)

    # ----------------------------
    # Visualization
    # ----------------------------
    for ch in range(min(args.num_channels, C)):

        true = all_target[sample_idx, :, ch]
        prediction = pred[sample_idx, :, ch]
        mask = all_evalpoint[sample_idx, :, ch]

        missing = prediction.copy()
        missing[mask == 0] = np.nan

        plt.figure(figsize=(12, 4))

        plt.plot(
            true,
            "--",
            linewidth=2,
            color="black",
            label="ground truth",
        )

        plt.plot(
            missing,
            linewidth=3,
            color="red",
            label="prediction",
        )

        plt.title(
            f"sample {sample_idx} channel {ch}"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            out_dir / f"channel_{ch}.png"
        )

        plt.close()

    print("Saved plots to:", out_dir)

    # ----------------------------
    # Metrics on missing region
    # ----------------------------

    # In CSDI eval_points == 1 means evaluate (missing locations)
    missing_mask = all_evalpoint.astype(bool)

    mse_missing = np.mean(
        (pred[missing_mask] - all_target[missing_mask]) ** 2
    )

    mae_missing = np.mean(
        np.abs(
            pred[missing_mask]
            - all_target[missing_mask]
        )
    )

    crps_missing = calc_quantile_crps(
        target=all_target,
        forecast=all_generated_samples,
        mask=all_evalpoint,
    )

    metrics = {
        "mse": float(mse_missing),
        "mae": float(mae_missing),
        "crps": float(crps_missing),
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print()
    print("================================")
    print("Missing-region metrics")
    print("================================")
    print("MSE  :", mse_missing)
    print("MAE  :", mae_missing)
    print("CRPS :", crps_missing)
    print("Saved metrics to:", out_dir / "metrics.json")
    print()


if __name__ == "__main__":
    main()