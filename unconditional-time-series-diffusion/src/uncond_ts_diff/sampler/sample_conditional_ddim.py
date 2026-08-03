import argparse
import json
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

class DWBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.dw(x)
        h = self.pw(h)
        h = self.norm(h)
        h = self.act(h)
        return x + h


class SiloTimeOnlyAE(nn.Module):
    def __init__(self, channels, seq_len, latent_steps=16):
        super().__init__()

        self.channels = channels
        self.seq_len = seq_len
        self.latent_steps = latent_steps

        self.block1 = DWBlock(channels)
        self.down1 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

        self.block2 = DWBlock(channels)
        self.down2 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

        self.block3 = DWBlock(channels)
        self.down3 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

        self.block4 = DWBlock(channels)
        self.to_latent = nn.Conv1d(channels, channels, 1)

        self.up1 = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.block5 = DWBlock(channels)

        self.up2 = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.block6 = DWBlock(channels)

        self.up3 = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.block7 = DWBlock(channels)

    def encode(self, x):
        h = self.block1(x)
        h = self.down1(h)

        h = self.block2(h)
        h = self.down2(h)

        h = self.block3(h)
        h = self.down3(h)

        h = self.block4(h)
        z = self.to_latent(h)

        z = F.interpolate(
            z,
            size=self.latent_steps,
            mode="linear",
            align_corners=False
        )
        return z

    def decode(self, z):
        h = self.up1(z)
        h = self.block5(h)
        h = self.up2(h)
        h = self.block6(h)
        h = self.up3(h)
        h = self.block7(h)
        h = F.interpolate(h, size=self.seq_len, mode="linear", align_corners=False)
        return h

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
        start = np.random.randint(0, T - block_size + 1)
        if i == 0:
            start = 55
        mask[i, start:start + block_size, :] = 0

    return mask


# =========================================================
# GUIDANCE
# =========================================================

def guidance_fn(x_recon):
    # only the channels owned/observed by the active silo
    x_recon_local = x_recon
    x_obs_local = x_obs[:, :, guided_idx]
    mask_local = mask[:, :, guided_idx]

    obs_count = mask_local.sum(dim=(1, 2)) + 1e-8
    guidance_loss = (((x_recon_local - x_obs_local) * mask_local) ** 2).sum(dim=(1, 2)) / obs_count
    guidance_loss = guidance_loss.mean()

    dx_rec = x_recon_local[:, 1:] - x_recon_local[:, :-1]
    missing_pair = (1 - mask_local[:, 1:]) * (1 - mask_local[:, :-1])

    smooth_loss = (dx_rec.pow(2) * missing_pair).sum() / (missing_pair.sum().clamp_min(1.0))

    m = mask_local[0, :, 0].detach().cpu().numpy()

    missing_idx = np.where(m == 0)[0]

    if len(missing_idx) > 0:
        t0 = missing_idx[0]
        t1 = missing_idx[-1] + 1

        if t0 > 0 and t1 < x_recon_local.shape[1]:
            left = x_recon_local[:, t0-1]
            right = x_recon_local[:, t1]

            gap = x_recon_local[:, t0:t1]

            boundary_mean = 0.5 * (left + right).unsqueeze(1)
            gap_mean = gap.mean(dim=1, keepdim=True)

            trend_loss = ((gap_mean - boundary_mean) ** 2).mean()
        else:
            trend_loss = 0.0
    else:
        trend_loss = 0.0

    # print(guidance_loss, trend_loss, smooth_loss)
    return guidance_loss + 0.05 * smooth_loss + 0.05 * trend_loss


def refine_latent(z_local, decoder, steps, scale, latent_mean, latent_std):
    z = z_local.detach()

    for _ in range(steps):
        z = z.detach().requires_grad_(True)

        z_denorm = z * latent_std + latent_mean

        x_recon = decoder(z_denorm)

        loss = guidance_fn(x_recon)

        grad = torch.autograd.grad(loss, z)[0]
        grad = grad / (grad.norm(dim=(1, 2), keepdim=True) + 1e-8)

        z = (z - scale * grad).detach()

    return z


# =========================================================
# GUIDED SAMPLING
# =========================================================

def sample_guided(model, decoder, num_samples, seq_len, num_channels, device,
                  latent_mean, latent_std, base_scale=0.20, base_repeats=30):

    x = torch.randn(num_samples, seq_len, num_channels, device=device)

    for i in reversed(range(model.timesteps)):
        print(i)
        noise = torch.randn_like(x)
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)

        with torch.no_grad():
            eps = model.backbone(x, t, None)

        x0_hat = model.fast_denoise(x, t, None, noise=eps)

        # if i < 30:
        tau = i / (model.timesteps - 1)
        guidance_scale = max(base_scale - base_scale * tau, base_scale * 0.1)
        steps = max(2 * base_repeats - int(2 * tau * base_repeats), 1)

        # only pass silo-known channels
        z_local = x0_hat[:, :, guided_latent_idx]

        z_local_refined = refine_latent(
            z_local, decoder, steps, guidance_scale,
            latent_mean, latent_std
        )

        # merge back into full latent
        x0_guided = x0_hat.clone()
        x0_guided[:, :, guided_latent_idx] = z_local_refined

        # else:
        #     x0_guided = x0_hat

        alpha_bar_prev = extract(model.alphas_cumprod_prev, t, x.shape)
        sqrt_ab_prev = torch.sqrt(alpha_bar_prev)

        alpha_bar = extract(model.alphas_cumprod, t, x.shape)

        sigma_t = torch.sqrt(
            (1 - alpha_bar_prev) / (1 - alpha_bar)
        ) * torch.sqrt(
            1 - alpha_bar / alpha_bar_prev
        )

        safe_term = torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=1e-8)
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
    missing = (mask == 0)
    observed = (mask == 1)

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
            metrics["r2_missing"] = float(
                r2_score(x_true[missing].flatten(), x_pred[missing].flatten())
            )
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
# MAIN SCENARIO
# =========================================================

def run_scenario(name, mask_np, out_dir):
    global mask, x_obs, decoder

    scenario_dir = out_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    guided_plot_dir = scenario_dir / "guided_features"
    hidden_plot_dir = scenario_dir / "hidden_features"
    guided_plot_dir.mkdir(parents=True, exist_ok=True)
    hidden_plot_dir.mkdir(parents=True, exist_ok=True)

    x_obs_np = x_full_np.copy()

    hidden_local_idx = [i for i in range(signal_channels) if i not in guided_idx]
    if len(hidden_local_idx) > 0:
        mask_np[:, :, hidden_local_idx] = 0

    x_obs_np[mask_np == 0] = 0.0

    mask = torch.from_numpy(mask_np).float().to(device)
    x_obs = torch.from_numpy(x_obs_np).float().to(device)

    samples = sample_guided(
        model,
        guidance_decoder,
        args.num_samples,
        args.latent_steps,
        num_channels,
        device,
        latent_mean,
        latent_std
    )

    decoded = decode_full_latent(samples)

    x_pred = decoded.detach().cpu().numpy()
    x_true = x_full_np

    # =========================
    # AE BASELINE
    # =========================
    x_true_torch = torch.from_numpy(x_full_np).float().to(device)

    x_ae_full = torch.zeros_like(x_true_torch)

    for silo_id in args.silo_ids:
        s = silos[silo_id]

        x_local = x_true_torch[:, :, s["feature_idx"]]
        x_local = x_local.permute(0, 2, 1)

        with torch.no_grad():
            z = s["ae"].encode(x_local)
            x_rec = s["ae"].decode(z).permute(0, 2, 1)

        x_ae_full[:, :, s["feature_idx"]] = x_rec

    x_ae = x_ae_full.cpu().numpy()

    # signal only
    x_pred = x_pred[:, :, :signal_channels]
    x_true = x_true[:, :, :signal_channels]
    x_obs_sig = x_obs_np[:, :, :signal_channels]
    x_ae = x_ae[:, :, :signal_channels]
    mask_sig = mask_np[:, :, :signal_channels]

    hidden_idx_local = [i for i in range(signal_channels) if i not in guided_idx]

    x_pred_guided = x_pred[:, :, guided_idx]
    x_true_guided = x_true[:, :, guided_idx]
    x_ae_guided = x_ae[:, :, guided_idx]
    mask_guided = mask_sig[:, :, guided_idx]

    if len(hidden_idx_local) > 0:
        x_pred_hidden = x_pred[:, :, hidden_idx_local]
        x_true_hidden = x_true[:, :, hidden_idx_local]
        x_ae_hidden = x_ae[:, :, hidden_idx_local]
    else:
        x_pred_hidden = np.empty((x_pred.shape[0], x_pred.shape[1], 0))
        x_true_hidden = np.empty((x_true.shape[0], x_true.shape[1], 0))
        x_ae_hidden = np.empty((x_ae.shape[0], x_ae.shape[1], 0))

    # =========================
    # ORIGINAL METRICS
    # =========================
    guided_metrics = compute_masked_metrics(x_pred_guided, x_true_guided, mask_guided)
    hidden_metrics = compute_full_metrics(x_pred_hidden, x_true_hidden) if len(hidden_idx_local) > 0 else {}

    # =========================
    # AE AS TARGET
    # =========================
    guided_metrics_ae_target = compute_masked_metrics(x_pred_guided, x_ae_guided, mask_guided)
    hidden_metrics_ae_target = (
        compute_full_metrics(x_pred_hidden, x_ae_hidden)
        if len(hidden_idx_local) > 0 else {}
    )

    # =========================
    # WRITE FILE
    # =========================
    with open(scenario_dir / "metrics.txt", "w") as f:
        f.write("=== DIFFUSION vs REAL (GUIDED) ===\n")
        for k, v in guided_metrics.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== DIFFUSION vs AE (GUIDED) ===\n")
        for k, v in guided_metrics_ae_target.items():
            f.write(f"{k}: {v}\n")

        f.write("\n=== DIFFUSION vs REAL (HIDDEN) ===\n")
        if hidden_metrics:
            for k, v in hidden_metrics.items():
                f.write(f"{k}: {v}\n")

        f.write("\n=== DIFFUSION vs AE (HIDDEN) ===\n")
        if hidden_metrics_ae_target:
            for k, v in hidden_metrics_ae_target.items():
                f.write(f"{k}: {v}\n")

    # =========================
    # PLOTS
    # =========================
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

        plt.plot(true, "--", label="ground truth", linewidth=2)
        plt.plot(ae_rec, label="AE", linewidth=2, color="orange")
        plt.plot(pred, color="firebrick", linewidth=2)
        plt.plot(pred_obs, color="darkgreen", label="pred (observed)", linewidth=2)
        plt.plot(pred_miss, color="firebrick", label="pred (missing)", linewidth=2)

        plt.title(f"Guided channel {ch}")
        plt.legend()
        plt.savefig(guided_plot_dir / f"channel_{ch}.png")
        plt.close()

    for ch in deterministic_plot_channels(hidden_idx_local, k=5):
        plt.figure(figsize=(10, 4))

        true = x_true[sample_idx, :, ch]
        pred = x_pred[sample_idx, :, ch]
        ae_rec = x_ae[sample_idx, :, ch]

        plt.plot(true, "--", label="ground truth", linewidth=2)
        plt.plot(ae_rec, label="AE", linewidth=2, color="orange")
        plt.plot(pred, label="prediction", linewidth=2)

        plt.title(f"Hidden channel {ch}")
        plt.legend()
        plt.savefig(hidden_plot_dir / f"channel_{ch}.png")
        plt.close()

def load_silo(silo_id):
    ae_dir = Path(args.silo_root_ae) / silo_id
    latent_dir = Path(args.silo_root_latents) / silo_id

    with open(ae_dir / "config.json") as f:
        cfg = json.load(f)

    feature_idx = cfg["feature_idx"]
    local_channels = len(feature_idx)

    ae_model = SiloTimeOnlyAE(
        channels=local_channels,
        seq_len=args.orig_seq_len,
        latent_steps=args.latent_steps,
    )

    model_path = ae_dir / "models" / f"silo_time_only_ae_{args.latent_steps}.pt"
    ae_model.load_state_dict(torch.load(model_path, map_location=device))
    ae_model = ae_model.to(device).eval()

    mean = torch.from_numpy(
        np.load(latent_dir / "latent_mean.npy")
    ).float().to(device)

    std = torch.from_numpy(
        np.load(latent_dir / "latent_std.npy")
    ).float().to(device)

    return {
        "id": silo_id,
        "ae": ae_model,
        "mean": mean,
        "std": std,
        "feature_idx": feature_idx,
        "channels": local_channels,
    }

def guidance_decoder(z_denorm):
    z = z_denorm.permute(0, 2, 1)
    x = ae.decode(z)
    return x.permute(0, 2, 1)

def decode_full_latent(z_norm):
    B = z_norm.shape[0]
    x_full = torch.zeros(B, args.orig_seq_len, signal_channels, device=z_norm.device)

    for silo_id in args.silo_ids:
        s = silos[silo_id]

        z_local = z_norm[:, :, s["latent_idx"]]
        z_local = z_local * s["std"] + s["mean"]
        z_local = z_local.permute(0, 2, 1)

        with torch.no_grad():
            x_local = s["ae"].decode(z_local).permute(0, 2, 1)

        x_full[:, :, s["feature_idx"]] = x_local

    return x_full

# =========================================================
# MAIN
# =========================================================

def main():
    global args, model, device, num_channels
    global latent_mean, latent_std
    global latent_mean_full, latent_std_full
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
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "random", "multi_block", "single_block"])

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--has_time_channels", action="store_true")

    # multi-silo
    parser.add_argument("--silo_root_ae", type=str, required=True)
    parser.add_argument("--silo_root_latents", type=str, required=True)
    parser.add_argument("--silo_ids", type=str, nargs="+", required=True)
    parser.add_argument("--guided_silo", type=str, required=True)

    args = parser.parse_args()

    # -------------------------
    # seed
    # -------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    # -------------------------
    # diffusion model
    # -------------------------
    ckpt_path = Path(
        f"../../../results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
    )
    model = TSDiff.load_from_checkpoint(ckpt_path).to(device)

    num_channels = model.backbone.input_init[0].in_features

    # -------------------------
    # load silos
    # -------------------------
    silos = {
        silo_id: load_silo(silo_id)
        for silo_id in args.silo_ids
    }

    # -------------------------
    # assign latent indices
    # -------------------------
    start = 0
    for silo_id in args.silo_ids:
        s = silos[silo_id]
        end = start + s["channels"]

        s["latent_idx"] = list(range(start, end))

        start = end

    if start != num_channels:
        raise ValueError(
            f"Latent mismatch: model={num_channels}, silos={start}"
        )

    # -------------------------
    # build FULL latent stats
    # -------------------------
    latent_mean_full = torch.zeros(1, 1, num_channels, device=device)
    latent_std_full  = torch.zeros(1, 1, num_channels, device=device)

    for silo_id in args.silo_ids:
        s = silos[silo_id]
        idx = s["latent_idx"]

        latent_mean_full[:, :, idx] = s["mean"]
        latent_std_full[:, :, idx]  = s["std"]

    # -------------------------
    # guided silo
    # -------------------------
    guided_silo = silos[args.guided_silo]

    guided_idx = guided_silo["feature_idx"]
    guided_latent_idx = guided_silo["latent_idx"]

    ae = guided_silo["ae"]
    latent_mean = guided_silo["mean"]
    latent_std = guided_silo["std"]

    # -------------------------
    # load data
    # -------------------------
    full_data = np.load(args.train_data)

    if args.num_samples > len(full_data):
        raise ValueError("num_samples too large")

    idx = np.random.choice(len(full_data), size=args.num_samples, replace=False)
    data = full_data[idx]

    x_full_np = data

    signal_channels = num_channels - 4 if args.has_time_channels else num_channels

    # validate
    for ch in guided_idx:
        if ch < 0 or ch >= signal_channels:
            raise ValueError("invalid guided channel")

    # -------------------------
    # masks
    # -------------------------
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