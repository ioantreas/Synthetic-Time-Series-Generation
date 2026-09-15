import argparse
import json
import math
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
        idx = np.random.choice(T, size=num_missing, replace=False)
        mask[i, idx, :] = 0

    return mask


def mask_multi_block(data, block_size, num_blocks=3):
    mask = np.ones_like(data)
    B, T, C = data.shape

    for i in range(B):
        for _ in range(num_blocks):
            start = np.random.randint(0, T - block_size + 1)
            mask[i, start:start + block_size, :] = 0

    return mask


def mask_single_block(data, block_size):
    mask = np.ones_like(data)
    B, T, C = data.shape

    for i in range(B):
        for c in range(C):
            start = np.random.randint(0, T - block_size + 1)
            # if i==0 and (c==0 or c==1):
            #     start =45
            mask[i, start:start + block_size, c] = 0

    return mask





# =========================================================
# GUIDED SAMPLING
# =========================================================

def real_space_guidance_fn(z, x_obs, mask):
    x_rec = decode_full_latent(z)

    loss = (((x_rec - x_obs) * mask) ** 2).sum()
    loss = loss / (mask.sum() + 1e-8)

    return loss


def refine_latent_real(z_full, steps, scale, x_obs, mask):
    z = z_full.detach()

    for _ in range(steps):
        z = z.detach().requires_grad_(True)

        loss = real_space_guidance_fn(z, x_obs, mask)

        grad = torch.autograd.grad(loss, z)[0]
        grad = grad / (grad.norm(dim=(1, 2), keepdim=True) + 1e-8)

        z = (z - scale * grad).detach()

    return z


# =========================================================
# GUIDED SAMPLING (REAL-SPACE BASELINE)
# =========================================================

def sample_guided_real_space(
        model,
        num_samples,
        seq_len,
        num_channels,
        device,
        x_obs,
        mask,
        base_scale=10.0,
        base_repeats=40,
):
    x = torch.randn(num_samples, seq_len, num_channels, device=device)

    for i in reversed(range(model.timesteps)):
        print(i)

        noise = torch.randn_like(x)
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)

        with torch.no_grad():
            eps = model.backbone(x, t, None)

        x0_hat = model.fast_denoise(x, t, None, noise=eps)

        tau = i / (model.timesteps - 1)
        guidance_scale = max(base_scale - base_scale * tau, base_scale * 0.1)
        steps = max(2 * base_repeats - int(2 * tau * base_repeats), 1)

        x0_guided = refine_latent_real(
            x0_hat,
            steps,
            guidance_scale,
            x_obs,
            mask,
        )

        alpha_bar_prev = extract(model.alphas_cumprod_prev, t, x.shape)
        sqrt_ab_prev = torch.sqrt(alpha_bar_prev)

        alpha_bar = extract(model.alphas_cumprod, t, x.shape)

        sigma_t = torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar)
        ) * torch.sqrt(
            1 - alpha_bar / alpha_bar_prev
        )

        safe_term = torch.clamp(1.0 - alpha_bar_prev - sigma_t ** 2, min=1e-8)
        sqrt_one_minus_ab_prev = torch.sqrt(safe_term)

        if i > 0:
            x = sqrt_ab_prev * x0_guided + sqrt_one_minus_ab_prev * eps + sigma_t * noise
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


# =========================================================
# DECODE / AE BASELINE
# =========================================================

def decode_full_latent(z_norm):
    B = z_norm.shape[0]
    x_full = torch.zeros(B, args.orig_seq_len, signal_channels, device=z_norm.device)

    for silo_id in args.silo_ids:
        s = silos[silo_id]

        z_local = z_norm[:, :, s["latent_idx"]]
        z_local = z_local * s["std"] + s["mean"]

        x_local = s["ae"].decode(z_local).permute(0, 2, 1)

        x_full[:, :, s["feature_idx"]] = x_local

    return x_full


def ae_reconstruct_full(x_true_torch):
    x_ae_full = torch.zeros_like(x_true_torch)

    for silo_id in args.silo_ids:
        s = silos[silo_id]

        x_local = x_true_torch[:, :, s["feature_idx"]]

        with torch.no_grad():
            z = s["ae"].encode(x_local.permute(0, 2, 1))
            x_rec = s["ae"].decode(z).permute(0, 2, 1)

        x_ae_full[:, :, s["feature_idx"]] = x_rec

    return x_ae_full


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

    # IMPORTANT:
    # Only the guided/target silo is partially masked.
    # Other silos stay fully observed and are used as cross-silo anchors.
    target_feature_idx = guided_idx
    all_features = list(range(signal_channels))
    other_feature_idx = [i for i in all_features if i not in target_feature_idx]

    # force other silos/features to observed
    mask_np[:, :, other_feature_idx] = 1

    # zero only the missing positions of the target guided features
    x_obs_np[mask_np == 0] = 0.0

    mask = torch.from_numpy(mask_np).float().to(device)
    x_obs = torch.from_numpy(x_obs_np).float().to(device)
    x_full = torch.from_numpy(x_full_np).float().to(device)

    all_preds = []

    for k in range(args.num_imputation_samples):

        print(
            f"Generating sample {k+1}/{args.num_imputation_samples}"
        )

        samples = sample_guided_real_space(
            model,
            args.num_samples,
            args.latent_steps,
            num_channels,
            device,
            x_obs,
            mask,
            base_scale=args.base_scale,
            base_repeats=args.base_repeats,
        )

        decoded = decode_full_latent(samples)

        all_preds.append(
            decoded.detach().cpu().numpy()
        )

    all_preds = np.stack(
        all_preds,
        axis=1,
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
# LOAD SILO
# =========================================================

def load_silo(silo_id):
    ae_dir = Path(args.silo_root_ae) / silo_id
    latent_dir = Path(args.silo_root_latents) / silo_id

    with open(ae_dir / "config.json") as f:
        cfg = json.load(f)

    feature_idx = cfg["feature_idx"]
    local_channels = len(feature_idx)

    ae_model = TransformerTimeAE(
        channels=local_channels,
        seq_len=args.orig_seq_len,
        latent_steps=cfg["latent_steps"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
    )

    model_path = ae_dir / "models" / f"transformer_ae_{cfg['latent_steps']}.pt"
    ae_model.load_state_dict(torch.load(model_path, map_location=device))
    ae_model = ae_model.to(device).eval()

    mean = torch.from_numpy(np.load(latent_dir / "latent_mean.npy")).float().to(device)
    std = torch.from_numpy(np.load(latent_dir / "latent_std.npy")).float().to(device)

    return {
        "id": silo_id,
        "ae": ae_model,
        "mean": mean,
        "std": std,
        "feature_idx": feature_idx,
        "channels": local_channels,
        "latent_dim": cfg["d_model"],
    }


# =========================================================
# MAIN
# =========================================================

def main():
    global args, model, device, num_channels
    global x_full_np, signal_channels
    global guided_idx, guided_latent_idx
    global silos, ae

    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--latent_steps", type=int, required=True)
    parser.add_argument("--orig_seq_len", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_data", type=str, required=True)

    parser.add_argument("--missing_ratio", type=float, required=True)
    parser.add_argument("--scenario", type=str, default="all", choices=["all", "random", "multi_block", "single_block"])

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--has_time_channels", action="store_true")

    parser.add_argument("--silo_root_ae", type=str, required=True)
    parser.add_argument("--silo_root_latents", type=str, required=True)
    parser.add_argument("--silo_ids", type=str, nargs="+", required=True)
    parser.add_argument("--guided_silo", type=str, required=True)

    parser.add_argument("--base_scale", type=float, default=10.0)
    parser.add_argument("--base_repeats", type=int, default=40)
    parser.add_argument("--anchor_weight", type=float, default=1.0)

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
    parser.add_argument("--mask_type", type=str, default="variance")
    parser.add_argument("--no_local_guidance", action="store_true")
    parser.add_argument("--no_global_guidance", action="store_true")


    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(
        f"../../../results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
    )
    model = TSDiff.load_from_checkpoint(ckpt_path).to(device)

    num_channels = model.backbone.input_init[0].in_features

    silos = {
        silo_id: load_silo(silo_id)
        for silo_id in args.silo_ids
    }

    start = 0
    for silo_id in args.silo_ids:
        s = silos[silo_id]
        end = start + s["latent_dim"]
        s["latent_idx"] = list(range(start, end))
        start = end

    if start != num_channels:
        raise ValueError(f"Latent mismatch: model={num_channels}, silos={start}")

    guided_silo = silos[args.guided_silo]
    guided_idx = guided_silo["feature_idx"]
    guided_latent_idx = guided_silo["latent_idx"]
    ae = guided_silo["ae"]

    full_data = np.load(args.train_data)

    if args.num_samples > len(full_data):
        raise ValueError("num_samples too large")

    idx = np.random.choice(len(full_data), size=args.num_samples, replace=False)
    data = full_data[idx]
    x_full_np = data

    # original feature dimension, not diffusion latent dimension
    signal_channels = data.shape[2]

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

    if args.scenario in ["all", "multi_block"]:
        mask_np = mask_multi_block(data, total_missing // 3, 3)
        if args.has_time_channels:
            mask_np[:, :, signal_channels:] = 1
        run_scenario("multi_block", mask_np, out_dir)

    if args.scenario in ["all", "single_block"]:
        mask_np = mask_single_block(data, total_missing)
        if args.has_time_channels:
            mask_np[:, :, signal_channels:] = 1
        run_scenario("single_block", mask_np, out_dir)


if __name__ == "__main__":
    main()
