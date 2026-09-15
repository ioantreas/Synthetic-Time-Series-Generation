import argparse
import json
from pathlib import Path

import numpy as np
import torch
import time
from matplotlib import pyplot as plt

from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.utils import extract
from tslearn.metrics import dtw
from sklearn.metrics import r2_score

def set_torch_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================
# MASK GENERATORS
# =========================================================

def mask_random(data, num_missing):
    mask = np.ones_like(data)
    B, T, C = data.shape

    for i in range(B):
        for c in range(C):
            idx = np.random.choice(T, size=num_missing, replace=False)
            mask[i, idx, c] = 0

    return mask


def mask_blackout(data, block_size):
    mask = np.ones_like(data)
    B, T, C = data.shape

    for i in range(B):
        start = np.random.randint(0, T - block_size + 1)
        mask[i, start:start + block_size, :] = 0

    return mask


def mask_single_block(data, block_size):
    mask = np.ones_like(data)
    B, T, C = data.shape

    for i in range(B):
        for c in range(C):
            start = np.random.randint(0, T - block_size + 1)
            mask[i, start:start + block_size, c] = 0

    return mask

def mask_forecast(data, block_size):
    mask = np.ones_like(data)
    B, T, C = data.shape

    for i in range(B):
        for c in range(C):
            start = T - block_size + 1
            mask[i, start:start + block_size, c] = 0

    return mask

# =========================================================
# DATA SPACE GUIDANCE
# =========================================================

def observed_data_loss(
        x0_hat,
        x_obs,
        observation_mask,
):
    squared_error = ((x0_hat - x_obs) * observation_mask).pow(2)

    loss_per_sample = squared_error.sum(dim=(1, 2)) / observation_mask.sum(dim=(1, 2)).clamp_min(1.0)

    return loss_per_sample.sum()


def refine_data_from_observations(x_initial, x_obs, observation_mask, steps, scale):
    x = x_initial.detach()

    for _ in range(steps):
        x = x.detach().requires_grad_(True)

        loss = observed_data_loss(x0_hat=x, x_obs=x_obs, observation_mask=observation_mask,)
        grad = torch.autograd.grad(loss, x, only_inputs=True,)[0]
        grad = grad / (grad.norm(dim=(1, 2), keepdim=True) + 1e-8)

        x = (x - scale * grad).detach()

    return x

# =========================================================
# SAMPLING
# =========================================================

def sample_data_guided(
        model,
        num_samples,
        seq_len,
        num_channels,
        device,
        x_obs,
        observation_mask,
        base_scale=1.0,
        base_repeats=40,
):
    x = torch.randn(num_samples, seq_len, num_channels, device=device,)

    for i in reversed(range(model.timesteps)):
        print(i)

        noise = torch.randn_like(x)

        t = torch.full(
            (num_samples,),
            i,
            device=device,
            dtype=torch.long,
        )

        with torch.no_grad():
            eps = model.backbone(x, t, None)

            # Current estimate of the clean normalized latent z_0.
            x0_hat = model.fast_denoise(x, t, None, noise=eps,)

        tau = i / max(model.timesteps - 1, 1)

        guidance_scale = max(
            base_scale - base_scale * tau,
            base_scale * 0.1,
        )

        steps = max(2 * base_repeats - int(2 * tau * base_repeats), 1)

        x0_guided = refine_data_from_observations(x0_hat, x_obs, observation_mask, steps, guidance_scale)

        alpha_bar_prev = extract(model.alphas_cumprod_prev, t, x.shape)

        sqrt_ab_prev = torch.sqrt(alpha_bar_prev)

        alpha_bar = extract(model.alphas_cumprod, t, x.shape)

        sigma_t = (torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev))

        safe_term = torch.clamp(1.0 - alpha_bar_prev - sigma_t.pow(2), min=1e-8)

        sqrt_one_minus_ab_prev = torch.sqrt(safe_term)

        if i > 0:
            x = (
                sqrt_ab_prev * x0_guided
                + sqrt_one_minus_ab_prev * eps
                + sigma_t * noise
            )
        else:
            x = x0_guided

    return x


# =========================================================
# METRICS
# =========================================================

def safe_corr(a, b):
    if a.size == 0 or b.size == 0:
        return np.nan
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return np.corrcoef(a, b)[0, 1]


def compute_masked_metrics(x_pred, x_true, mask):
    missing = mask == 0
    observed = mask == 1

    metrics = {}

    if missing.sum() == 0:
        metrics["mse_missing"] = np.nan
        metrics["mae_missing"] = np.nan
        metrics["corr_missing"] = np.nan
        metrics["r2_missing"] = np.nan
        metrics["dtw_missing"] = np.nan
        metrics["spectral_mse_missing"] = np.nan
    else:
        metrics["mse_missing"] = float(((x_pred - x_true)[missing] ** 2).mean())
        metrics["mae_missing"] = float(np.abs(x_pred - x_true)[missing].mean())
        metrics["corr_missing"] = float(safe_corr(x_pred[missing], x_true[missing]))

        try:
            metrics["r2_missing"] = float(r2_score(x_true[missing].flatten(), x_pred[missing].flatten()))
        except Exception:
            metrics["r2_missing"] = np.nan

        vals = []
        for i in range(min(len(x_pred), 10)):
            for c in range(x_pred.shape[2]):
                m = mask[i, :, c] == 0
                if m.sum() < 5:
                    continue
                vals.append(dtw(x_pred[i, :, c][m], x_true[i, :, c][m]))
        metrics["dtw_missing"] = float(np.mean(vals)) if vals else np.nan

        pred_fft = np.abs(np.fft.fft(x_pred, axis=1))
        true_fft = np.abs(np.fft.fft(x_true, axis=1))
        metrics["spectral_mse_missing"] = float(((pred_fft - true_fft)[mask == 0] ** 2).mean())

    if observed.sum() == 0:
        metrics["mse_observed"] = np.nan
    else:
        metrics["mse_observed"] = float(((x_pred - x_true)[observed] ** 2).mean())

    return metrics


def compute_full_metrics(x_pred, x_true):
    metrics = {}
    metrics["mse_full"] = float(((x_pred - x_true) ** 2).mean())
    metrics["mae_full"] = float(np.abs(x_pred - x_true).mean())
    metrics["corr_full"] = float(safe_corr(x_pred.flatten(), x_true.flatten()))

    try:
        metrics["r2_full"] = float(r2_score(x_true.flatten(), x_pred.flatten()))
    except Exception:
        metrics["r2_full"] = np.nan

    vals = []
    for i in range(min(len(x_pred), 10)):
        for c in range(x_pred.shape[2]):
            vals.append(dtw(x_pred[i, :, c], x_true[i, :, c]))
    metrics["dtw_full"] = float(np.mean(vals)) if vals else np.nan

    pred_fft = np.abs(np.fft.fft(x_pred, axis=1))
    true_fft = np.abs(np.fft.fft(x_true, axis=1))
    metrics["spectral_mse_full"] = float(((pred_fft - true_fft) ** 2).mean())

    return metrics


def deterministic_plot_channels(idxs, k=5):
    return idxs[:min(k, len(idxs))]


def quantile_loss(target, forecast, q):
    return 2 * np.sum(
        np.abs(
            (forecast - target)
            * ((target <= forecast).astype(np.float32) - q)
        )
    )

def compute_csdi_crps(
    all_preds,
    x_true,
    mask,
):
    """
    all_preds: [B,K,T,C]
    x_true:    [B,T,C]
    mask:      [B,T,C]
              0 = missing
              1 = observed

    Returns CSDI-style CRPS on missing region only.
    """

    missing = (mask == 0)

    quantiles = np.arange(0.05, 1.0, 0.05)

    target = x_true[missing]

    denom = np.sum(np.abs(target)) + 1e-8

    crps = 0.0

    for q in quantiles:

        q_pred = np.quantile(
            all_preds,
            q,
            axis=1,
        )  # [B,T,C]

        forecast = q_pred[missing]

        q_loss = quantile_loss(
            target,
            forecast,
            q,
        )

        crps += q_loss / denom

    return float(crps / len(quantiles))

# =========================================================
# SCENARIO
# =========================================================

def run_scenario(name, mask_np, out_dir):
    scenario_dir = out_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    np.save(scenario_dir / "mask.npy", mask_np)
    np.save(scenario_dir / "test_data.npy", x_full_np)

    guided_plot_dir = scenario_dir / "guided_features"
    guided_plot_dir.mkdir(parents=True, exist_ok=True)

    x_obs_np = x_full_np.copy()
    x_obs_np[mask_np == 0] = 0.0

    observation_mask = torch.from_numpy(mask_np).float().to(device)
    x_obs = torch.from_numpy(x_obs_np).float().to(device)

    set_torch_seed(args.inference_seed + 1_000_000)

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()

    inference_start = time.perf_counter()

    all_preds = []

    for k in range(args.num_imputation_samples):
        print(f"Generating sample {k + 1}/{args.num_imputation_samples}")

        samples = sample_data_guided(
            model=model,
            num_samples=args.num_samples,
            seq_len=x_full_np.shape[1],
            num_channels=num_channels,
            device=device,
            x_obs=x_obs,
            observation_mask=observation_mask,
            base_scale=args.base_scale,
            base_repeats=args.base_repeats,
        )

        all_preds.append(samples.detach().cpu().numpy())

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()

    inference_time_sec = time.perf_counter() - inference_start
    inference_time_per_imputation_sec = inference_time_sec / args.num_imputation_samples

    all_preds = np.stack(all_preds, axis=1)

    mask_sig = mask_np[:, :, :signal_channels]
    all_preds_sig = all_preds[:, :, :, :signal_channels]

    missing_predictions = all_preds_sig.transpose(1, 0, 2, 3)[:, mask_sig == 0]

    np.save(
        scenario_dir / "missing_predictions.npy",
        missing_predictions.astype(np.float32),
    )

    x_pred = np.median(all_preds_sig, axis=1)
    x_true = x_full_np[:, :, :signal_channels]

    guided_metrics = compute_masked_metrics(
        x_pred,
        x_true,
        mask_sig,
    )

    guided_metrics["crps_missing"] = compute_csdi_crps(
        all_preds_sig,
        x_true,
        mask_sig,
    )

    guided_metrics["inference_time_sec"] = inference_time_sec
    guided_metrics["inference_time_per_imputation_sec"] = inference_time_per_imputation_sec
    guided_metrics["total_method_time_sec"] = inference_time_sec

    full_metrics_real = compute_full_metrics(
        x_pred,
        x_true,
    )

    with open(scenario_dir / "metrics.txt", "w") as f:
        f.write("=== DATA-SPACE GUIDANCE: DIFFUSION vs REAL ===\n")
        for key, value in guided_metrics.items():
            f.write(f"{key}: {value}\n")

        f.write("\n=== FULL SAMPLE: DIFFUSION vs REAL ===\n")
        for key, value in full_metrics_real.items():
            f.write(f"{key}: {value}\n")

    print("\nSaved results to:", scenario_dir)
    print("\n=== DATA-SPACE GUIDANCE: DIFFUSION vs REAL ===")
    for key, value in guided_metrics.items():
        print(f"{key}: {value}")

    sample_idx = 0
    guided_idx = list(range(signal_channels))

    for ch in deterministic_plot_channels(guided_idx, k=10):
        plt.figure(figsize=(10, 4))

        true = x_true[sample_idx, :, ch]
        pred = x_pred[sample_idx, :, ch]
        current_mask = mask_sig[sample_idx, :, ch]

        pred_obs = pred.copy()
        pred_obs[current_mask == 0] = np.nan

        pred_miss = pred.copy()
        pred_miss[current_mask == 1] = np.nan

        obs_plot = x_obs_np[sample_idx, :, ch].copy()
        obs_plot[current_mask == 0] = np.nan

        plt.plot(true, "--", label="ground truth", linewidth=2)
        plt.plot(obs_plot, "--", label="observed", linewidth=2)
        plt.plot(pred_obs, label="pred observed", linewidth=2)
        plt.plot(pred_miss, label="pred missing", linewidth=2)

        plt.title(f"Data-space guided channel {ch}")
        plt.legend()
        plt.savefig(guided_plot_dir / f"channel_{ch}.png")
        plt.close()

# =========================================================
# MAIN
# =========================================================

def main():
    global args, model, device, num_channels
    global x_full_np, signal_channels

    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_data", type=str, required=True)

    parser.add_argument("--missing_ratio", type=float, required=True)
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "random", "blackout", "single_block", "forecast"])

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_seed", type=int, default=42, help="Seed for stochastic guidance construction and diffusion sampling",)
    parser.add_argument("--has_time_channels", action="store_true")

    parser.add_argument("--base_scale", type=float, default=1.0)
    parser.add_argument("--base_repeats", type=int, default=40)

    parser.add_argument("--num_imputation_samples", type=int, default=10)


    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(
        # f"../../../results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
        f"/media/aioannou/OS/aioannou_storage/results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
    )
    model = TSDiff.load_from_checkpoint(ckpt_path).to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    num_channels = model.backbone.input_init[0].in_features

    full_data = np.load(args.train_data)

    if args.num_samples > len(full_data):
        raise ValueError("num_samples too large")

    idx = np.random.choice(len(full_data), size=args.num_samples, replace=False)
    data = full_data[idx]
    x_full_np = data

    # original feature dimension, not diffusion latent dimension
    signal_channels = data.shape[2]

    if num_channels != signal_channels:
        raise ValueError(
            f"Checkpoint expects {num_channels} channels, "
            f"but the supplied data contains {signal_channels} channels."
        )

    T = data.shape[1]
    total_missing = int(T * args.missing_ratio)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scenario in ["all", "random"]:
        mask_np = mask_random(data, total_missing)
        if args.has_time_channels:
            mask_np[:, :, signal_channels:] = 1
        run_scenario("random", mask_np, out_dir)

    if args.scenario in ["all", "blackout"]:
        mask_np = mask_blackout(data, total_missing)
        if args.has_time_channels:
            mask_np[:, :, signal_channels:] = 1
        run_scenario("blackout", mask_np, out_dir)

    if args.scenario in ["all", "single_block"]:
        mask_np = mask_single_block(data, total_missing)
        if args.has_time_channels:
            mask_np[:, :, signal_channels:] = 1
        run_scenario("single_block", mask_np, out_dir)

    if args.scenario in ["all", "forecast"]:
        mask_np = mask_forecast(data, total_missing)
        if args.has_time_channels:
            mask_np[:, :, signal_channels:] = 1
        run_scenario("forecast", mask_np, out_dir)


if __name__ == "__main__":
    main()
