import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.utils import extract
from tslearn.metrics import dtw
from sklearn.metrics import r2_score

def set_torch_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sync_cuda():
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()

# =========================================================
# AUTOENCODER
# =========================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerTimeAE(nn.Module):
    """
    Input:
        x -> [B,C,T]
    Latent:
        z -> [B,L,D]
    """

    def __init__(
            self,
            channels,
            seq_len,
            latent_steps=16,
            d_model=128,
            nhead=8,
            num_layers=4,
    ):
        super().__init__()

        self.channels = channels
        self.seq_len = seq_len
        self.latent_steps = latent_steps
        self.d_model = d_model

        self.input_proj = nn.Linear(channels, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.latent_queries = nn.Parameter(torch.randn(latent_steps, d_model))
        self.cross_attn_enc = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.cross_attn_dec = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.output_proj = nn.Linear(d_model, channels)

    def encode(self, x):
        # [B,C,T] -> [B,T,C]
        x = x.permute(0, 2, 1)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)

        B = h.shape[0]
        q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        z, _ = self.cross_attn_enc(q, h, h)

        # [B,L,D]
        return z

    def decode(self, z):
        # z: [B,L,D]
        B = z.shape[0]
        seq_queries = torch.zeros(B, self.seq_len, self.d_model, device=z.device)
        q = self.pos_enc(seq_queries)
        h, _ = self.cross_attn_dec(q, z, z)
        h = self.decoder(h)
        x = self.output_proj(h)

        # [B,T,C] -> [B,C,T]
        return x.permute(0, 2, 1)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


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
# NAIVE DECODER / OBSERVATION GUIDANCE
# =========================================================

def observed_data_loss(
        z_norm,
        x_obs,
        observation_mask,
):
    """
    Decode the current clean-latent estimate and measure its
    reconstruction error only at observed data-space entries.

    z_norm:           [B,L,D], normalized latent
    x_obs:            [B,T,C]
    observation_mask: [B,T,C], 1 = observed, 0 = missing
    """

    z_denorm = z_norm * latent_std + latent_mean

    # Do not use torch.no_grad(): gradients must pass through
    # the decoder and back to z_norm.
    x_rec = ae.decode(z_denorm).permute(0, 2, 1)

    squared_error = ((x_rec - x_obs) * observation_mask).pow(2)

    # Compute one normalized loss per sequence so that sequences
    # with more observed entries do not dominate the batch.
    loss_per_sample = (
        squared_error.sum(dim=(1, 2))
        / observation_mask.sum(dim=(1, 2)).clamp_min(1.0)
    )

    return loss_per_sample.sum()


def refine_latent_from_observations(
        z_initial,
        x_obs,
        observation_mask,
        steps,
        scale,
):
    """
    Refine the predicted clean latent using observed-value
    reconstruction error through the autoencoder decoder.
    """

    z = z_initial.detach()

    for _ in range(steps):
        z = z.detach().requires_grad_(True)

        loss = observed_data_loss(
            z_norm=z,
            x_obs=x_obs,
            observation_mask=observation_mask,
        )

        grad = torch.autograd.grad(loss, z, only_inputs=True,)[0]

        # Normalize independently for each sequence.
        grad = grad / (grad.norm(dim=(1, 2), keepdim=True) + 1e-8)

        z = (z - scale * grad).detach()

    return z

# =========================================================
# NAIVE DECODER-GUIDED SAMPLING
# =========================================================

def sample_decoder_guided(
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

    server_to_client_bytes = 0
    client_to_server_bytes = 0
    communication_trace = []
    cumulative_bytes = 0

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

        steps = max(
            2 * base_repeats
            - int(2 * tau * base_repeats),
            1,
        )

        # Server -> client: send the current clean latent estimate.
        step_server_to_client_bytes = x0_hat.numel() * x0_hat.element_size()
        server_to_client_bytes += step_server_to_client_bytes

        # All refinement iterations are performed locally at the client.
        x0_guided = refine_latent_from_observations(
            z_initial=x0_hat,
            x_obs=x_obs,
            observation_mask=observation_mask,
            steps=steps,
            scale=guidance_scale,
        )

        # Client -> server: return the final refined latent estimate.
        step_client_to_server_bytes = x0_guided.numel() * x0_guided.element_size()
        client_to_server_bytes += step_client_to_server_bytes

        step_total_bytes = step_server_to_client_bytes + step_client_to_server_bytes
        cumulative_bytes += step_total_bytes

        communication_trace.append({
            "reverse_step": model.timesteps - i,
            "diffusion_t": i,
            "server_to_client_bytes": step_server_to_client_bytes,
            "client_to_server_bytes": step_client_to_server_bytes,
            "total_bytes": step_total_bytes,
            "cumulative_bytes": cumulative_bytes,
        })

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

    return x, server_to_client_bytes, client_to_server_bytes, communication_trace

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


# =========================================================
# DECODE / AE BASELINE
# =========================================================

def decode_full_latent(z_norm):
    z = z_norm * latent_std + latent_mean

    with torch.no_grad():
        x = ae.decode(z).permute(0, 2, 1)

    return x


def ae_reconstruct_full(x_true_torch):
    with torch.no_grad():
        z = ae.encode(x_true_torch.permute(0, 2, 1))
        x_rec = ae.decode(z).permute(0, 2, 1)

    return x_rec


def quantile_loss(target, forecast, q):
    return 2 * np.sum(
        np.abs(
            (forecast - target)
            * ((target <= forecast).astype(np.float32) - q)
        )
    )


def calc_denominator(target):
    return np.sum(np.abs(target))


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
    global mask, x_obs

    scenario_dir = out_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    np.save(scenario_dir / "mask.npy", mask_np)
    np.save(scenario_dir / "test_data.npy", x_full_np)

    args.current_scenario_dir = scenario_dir

    guided_plot_dir = scenario_dir / "guided_features"
    hidden_plot_dir = scenario_dir / "hidden_features"
    observed_plot_dir = scenario_dir / "other_silo_features"

    guided_plot_dir.mkdir(parents=True, exist_ok=True)
    hidden_plot_dir.mkdir(parents=True, exist_ok=True)
    observed_plot_dir.mkdir(parents=True, exist_ok=True)

    x_obs_np = x_full_np.copy()

    guided_idx = list(range(signal_channels))
    other_feature_idx = []

    x_obs_np[mask_np == 0] = 0.0

    mask = torch.from_numpy(mask_np).float().to(device)
    x_obs = torch.from_numpy(x_obs_np).float().to(device)
    x_full = torch.from_numpy(x_full_np).float().to(device)

    set_torch_seed(args.inference_seed + 1_000_000)

    sync_cuda()
    inference_start = time.perf_counter()

    all_preds = []
    server_to_client_bytes = 0
    client_to_server_bytes = 0
    communication_trace_sum = None

    for k in range(args.num_imputation_samples):
        print(
            f"Generating sample {k + 1}/{args.num_imputation_samples}"
        )

        (samples, sample_server_to_client_bytes, sample_client_to_server_bytes,
         sample_communication_trace) = sample_decoder_guided(
            model=model,
            num_samples=args.num_samples,
            seq_len=args.latent_steps,
            num_channels=num_channels,
            device=device,
            x_obs=x_obs,
            observation_mask=mask,
            base_scale=args.base_scale,
            base_repeats=args.base_repeats,
        )

        decoded = decode_full_latent(samples)

        all_preds.append(
            decoded.detach().cpu().numpy()
        )

        server_to_client_bytes += sample_server_to_client_bytes
        client_to_server_bytes += sample_client_to_server_bytes

        if communication_trace_sum is None:
            communication_trace_sum = [
                {
                    "reverse_step": row["reverse_step"],
                    "diffusion_t": row["diffusion_t"],
                    "server_to_client_bytes": row["server_to_client_bytes"],
                    "client_to_server_bytes": row["client_to_server_bytes"],
                    "total_bytes": row["total_bytes"],
                }
                for row in sample_communication_trace
            ]
        else:
            for total_row, sample_row in zip(communication_trace_sum, sample_communication_trace):
                total_row["server_to_client_bytes"] += sample_row["server_to_client_bytes"]
                total_row["client_to_server_bytes"] += sample_row["client_to_server_bytes"]
                total_row["total_bytes"] += sample_row["total_bytes"]

    sync_cuda()
    inference_time = time.perf_counter() - inference_start

    communication_total_bytes = server_to_client_bytes + client_to_server_bytes

    inference_time_per_imputation_sec = inference_time / args.num_imputation_samples
    communication_per_imputation_bytes = communication_total_bytes / args.num_imputation_samples

    timing = {
        "inference_time_sec": inference_time,
        "inference_time_per_imputation_sec": inference_time_per_imputation_sec,
        "total_method_time_sec": inference_time,
    }

    cumulative_per_imputation_bytes = 0.0

    for row in communication_trace_sum:
        row["server_to_client_bytes_per_imputation"] = row["server_to_client_bytes"] / args.num_imputation_samples
        row["client_to_server_bytes_per_imputation"] = row["client_to_server_bytes"] / args.num_imputation_samples
        row["total_bytes_per_imputation"] = row["total_bytes"] / args.num_imputation_samples

        cumulative_per_imputation_bytes += row["total_bytes_per_imputation"]
        row["cumulative_bytes_per_imputation"] = cumulative_per_imputation_bytes

    all_preds = np.stack(
        all_preds,
        axis=1,
    )

    # Keep only predictions at masked positions.
    all_preds_sig = all_preds[:, :, :, :signal_channels]
    mask_sig = mask_np[:, :, :signal_channels]

    # [B,K,T,C] -> [K,B,T,C] -> [K,N_missing]
    missing_predictions = all_preds_sig.transpose(1, 0, 2, 3)[:, mask_sig == 0]

    np.save(
        scenario_dir / "missing_predictions.npy",
        missing_predictions.astype(np.float32),
    )

    x_pred = np.median(
        all_preds,
        axis=1,
    )

    x_true = x_full_np

    with torch.no_grad():
        x_ae = ae_reconstruct_full(x_full).detach().cpu().numpy()

    x_pred = x_pred[:, :, :signal_channels]
    x_true = x_true[:, :, :signal_channels]
    x_ae = x_ae[:, :, :signal_channels]
    mask_sig = mask_np[:, :, :signal_channels]

    # target/guided features: evaluate missing regions
    x_pred_guided = x_pred[:, :, guided_idx]
    x_true_guided = x_true[:, :, guided_idx]
    x_ae_guided = x_ae[:, :, guided_idx]
    mask_guided = mask_sig[:, :, guided_idx]

    guided_metrics = compute_masked_metrics(x_pred_guided, x_true_guided, mask_guided)
    guided_metrics_ae_target = compute_masked_metrics(x_pred_guided, x_ae_guided, mask_guided)

    guided_metrics["crps_missing"] = compute_csdi_crps(
        all_preds[:, :, :, guided_idx],
        x_true_guided,
        mask_guided,
    )

    # other silos/features: should be preserved by anchors
    x_pred_other = x_pred[:, :, other_feature_idx]
    x_true_other = x_true[:, :, other_feature_idx]
    x_ae_other = x_ae[:, :, other_feature_idx]

    other_metrics_real = compute_full_metrics(x_pred_other, x_true_other) if len(other_feature_idx) > 0 else {}
    other_metrics_ae = compute_full_metrics(x_pred_other, x_ae_other) if len(other_feature_idx) > 0 else {}

    # full metrics for sanity
    full_metrics_real = compute_full_metrics(x_pred, x_true)
    full_metrics_ae = compute_full_metrics(x_pred, x_ae)

    communication_trace_path = scenario_dir / "communication_trace.csv"

    with open(communication_trace_path, "w") as f:
        f.write(
            "reverse_step,diffusion_t,"
            "server_to_client_bytes_per_imputation,"
            "client_to_server_bytes_per_imputation,"
            "total_bytes_per_imputation,"
            "cumulative_bytes_per_imputation\n"
        )

        for row in communication_trace_sum:
            f.write(
                f"{row['reverse_step']},"
                f"{row['diffusion_t']},"
                f"{row['server_to_client_bytes_per_imputation']},"
                f"{row['client_to_server_bytes_per_imputation']},"
                f"{row['total_bytes_per_imputation']},"
                f"{row['cumulative_bytes_per_imputation']}\n"
            )

    with open(scenario_dir / "metrics.txt", "w") as f:
        f.write("=== TARGET/GUIDED SILO: DIFFUSION vs REAL ===\n")
        for k, v in guided_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== TARGET/GUIDED SILO: DIFFUSION vs AE ===\n")
        for k, v in guided_metrics_ae_target.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== OTHER SILOS ANCHOR CHECK: DIFFUSION vs REAL ===\n")
        for k, v in other_metrics_real.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== OTHER SILOS ANCHOR CHECK: DIFFUSION vs AE ===\n")
        for k, v in other_metrics_ae.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== FULL SAMPLE: DIFFUSION vs REAL ===\n")
        for k, v in full_metrics_real.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== FULL SAMPLE: DIFFUSION vs AE ===\n")
        for k, v in full_metrics_ae.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== TIMING ===\n")
        for k, v in timing.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== COMMUNICATION ===\n")
        f.write(f"communication_server_to_client_bytes: {server_to_client_bytes}\n")
        f.write(f"communication_client_to_server_bytes: {client_to_server_bytes}\n")
        f.write(f"communication_total_bytes: {communication_total_bytes}\n")
        f.write(f"communication_per_imputation_bytes: {communication_per_imputation_bytes}\n")

    print("\nSaved results to:", scenario_dir)
    print("\n=== TARGET/GUIDED SILO: DIFFUSION vs REAL ===")
    for k, v in guided_metrics.items():
        print(f"{k}: {v}")

    print("\n=== OTHER SILOS ANCHOR CHECK: DIFFUSION vs REAL ===")
    for k, v in other_metrics_real.items():
        print(f"{k}: {v}")


    # plots target/guided features
    sample_idx = 0

    for ch in deterministic_plot_channels(guided_idx, k=10):
        plt.figure(figsize=(10, 4))

        true = x_true[sample_idx, :, ch]
        pred = x_pred[sample_idx, :, ch]
        ae_rec = x_ae[sample_idx, :, ch]
        m = mask_sig[sample_idx, :, ch]

        pred_obs = pred.copy()
        pred_obs[m == 0] = np.nan

        pred_miss = pred.copy()
        pred_miss[m == 1] = np.nan

        obs_plot = x_obs_np[sample_idx, :, ch].copy()
        obs_plot[m == 0] = np.nan

        plt.plot(true, "--", label="ground truth", linewidth=2)
        plt.plot(obs_plot, "--", label="observed", linewidth=2)
        plt.plot(ae_rec, label="AE", linewidth=2, color="orange")
        plt.plot(pred, color="firebrick", linewidth=2)
        plt.plot(pred_obs, color="darkgreen", label="pred observed", linewidth=2)
        plt.plot(pred_miss, color="firebrick", label="pred missing", linewidth=2)

        plt.title(f"Guided/target channel {ch}")
        plt.legend()
        plt.savefig(guided_plot_dir / f"channel_{ch}.png")
        plt.close()

    # plots other anchored features
    for ch in deterministic_plot_channels(other_feature_idx, k=5):
        plt.figure(figsize=(10, 4))

        true = x_true[sample_idx, :, ch]
        pred = x_pred[sample_idx, :, ch]
        ae_rec = x_ae[sample_idx, :, ch]

        plt.plot(true, "--", label="ground truth", linewidth=2)
        plt.plot(ae_rec, label="AE", linewidth=2, color="orange")
        plt.plot(pred, label="prediction", linewidth=2)

        plt.title(f"Other anchored channel {ch}")
        plt.legend()
        plt.savefig(observed_plot_dir / f"channel_{ch}.png")
        plt.close()


# =========================================================
# LOAD AE
# =========================================================

def load_autoencoder(root, latents_root, signal_channels):
    root = Path(root)

    with open(root / "config.json") as f:
        cfg = json.load(f)

    ae = TransformerTimeAE(
        channels=signal_channels if cfg.get("feature_idx") == "all" else len(cfg["feature_idx"]),
        seq_len=args.orig_seq_len,
        latent_steps=cfg["latent_steps"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
    )

    ae.load_state_dict(
        torch.load(
            root / "models" / f"transformer_ae_{cfg['latent_steps']}.pt",
            map_location=device,
        )
    )

    ae = ae.to(device).eval()

    latent_mean = torch.from_numpy(
        np.load(latents_root / "latent_mean.npy")
    ).float().to(device)

    latent_std = torch.from_numpy(
        np.load(latents_root / "latent_std.npy")
    ).float().to(device)

    return ae, latent_mean, latent_std


# =========================================================
# MAIN
# =========================================================

def main():
    global args, model, device, num_channels
    global x_full_np, signal_channels
    global ae, latent_mean, latent_std
    global guided_idx

    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--latent_steps", type=int, required=True)
    parser.add_argument("--orig_seq_len", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_data", type=str, required=True)

    parser.add_argument("--missing_ratio", type=float, required=True)
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "random", "blackout", "single_block", "forecast"])

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference_seed", type=int, default=42, help="Seed for stochastic guidance construction and diffusion sampling",)
    parser.add_argument("--has_time_channels", action="store_true")

    parser.add_argument("--ae_root", type=str, required=True, help="Folder containing the AE, config and latent statistics")
    parser.add_argument("--latents_root", type=str, required=True)

    parser.add_argument("--base_scale", type=float, default=1.0)
    parser.add_argument("--base_repeats", type=int, default=40)

    parser.add_argument("--client_steps", type=int, default=200)
    parser.add_argument("--client_lr", type=float, default=1e-2)
    parser.add_argument("--client_prior_weight", type=float, default=1e-3)
    parser.add_argument("--num_client_plot_channels", type=int, default=3)

    parser.add_argument("--num_imputation_samples", type=int, default=10)
    parser.add_argument(
        "--latent_keep_percent",
        type=float,
        default=0.10,
        help="Fraction of highest-weight latent entries to keep unchanged(e.g. 0.20 = top 20%)",
    )
    parser.add_argument("--mask_type", type=str, default="variance",
                        choices=["variance", "interpolation", "random", "full"])


    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(
        f"../../../results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
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

    ae, latent_mean, latent_std = load_autoencoder(args.ae_root, Path(args.latents_root), signal_channels)

    ae.eval()

    for parameter in ae.parameters():
        parameter.requires_grad_(False)

    guided_idx = list(range(signal_channels))

    for ch in guided_idx:
        if ch < 0 or ch >= signal_channels:
            raise ValueError(f"invalid guided channel {ch}")

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
