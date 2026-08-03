import argparse
from pathlib import Path
import json

import numpy as np
import torch

from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler.observation_guidance_adapted import DDPMGuidance, DDIMGuidance


def quantile_loss(target, forecast, q):
    return 2 * np.sum(np.abs((forecast - target) * ((target <= forecast).astype(np.float32) - q)))


def compute_crps(all_preds, x_true, mask):
    missing = mask == 0
    quantiles = np.arange(0.05, 1.0, 0.05)
    target = x_true[missing]
    denom = np.sum(np.abs(target)) + 1e-8

    crps = 0.0
    for q in quantiles:
        q_pred = np.quantile(all_preds, q, axis=1)
        forecast = q_pred[missing]
        crps += quantile_loss(target, forecast, q) / denom

    return float(crps / len(quantiles))


def compute_metrics(all_preds, x_true, mask):
    x_pred = np.median(all_preds, axis=1)
    missing = mask == 0
    observed = mask == 1

    metrics = {
        "mse_missing": float(((x_pred - x_true)[missing] ** 2).mean()),
        "mae_missing": float(np.abs(x_pred - x_true)[missing].mean()),
        "mse_observed": float(((x_pred - x_true)[observed] ** 2).mean()),
        "crps_missing": compute_crps(all_preds, x_true, mask),
    }

    return metrics, x_pred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddpm")
    parser.add_argument("--guidance", choices=["MSE", "quantile"], default="quantile")
    parser.add_argument("--scale", type=float, default=1.0)

    parser.add_argument("--num_imputation_samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TSDiff.load_from_checkpoint(args.ckpt).to(device)
    model.eval()

    x = np.load(args.data).astype(np.float32)
    mask = np.load(args.mask).astype(np.float32)

    if x.shape != mask.shape:
        raise ValueError(f"data and mask shape mismatch: {x.shape} vs {mask.shape}")

    x_obs = x.copy()
    x_obs[mask == 0] = 0.0

    x_obs_t = torch.from_numpy(x_obs).float().to(device)
    mask_t = torch.from_numpy(mask).float().to(device)

    Sampler = DDPMGuidance if args.sampler == "ddpm" else DDIMGuidance

    sampler = Sampler(
        model=model,
        prediction_length=0,
        scale=args.scale,
        num_samples=1,
        guidance=args.guidance,
    ).to(device)

    preds = []

    for k in range(args.num_imputation_samples):
        print(f"Generating imputation {k+1}/{args.num_imputation_samples}")

        batch_size = 32

        outputs = []

        for start in range(0, len(x_obs_t), batch_size):
            end = start + batch_size

            batch = sampler.sample(
                observation=x_obs_t[start:end],
                observation_mask=mask_t[start:end],
                features=None,
                scale_params=None
            )

            outputs.append(batch.cpu())

        sample = torch.cat(outputs)

        preds.append(sample.detach().cpu().numpy())

    all_preds = np.stack(preds, axis=1)

    metrics, x_pred = compute_metrics(all_preds, x, mask)

    np.save(out_dir / "all_preds.npy", all_preds)
    np.save(out_dir / "median_pred.npy", x_pred)
    np.save(out_dir / "data.npy", x)
    np.save(out_dir / "mask.npy", mask)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(metrics)


if __name__ == "__main__":
    main()