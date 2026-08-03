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


# def mask_single_block(data, block_size):
#     mask = np.ones_like(data)
#     B, T, C = data.shape
#
#     for i in range(B):
#         start = np.random.randint(0, T - block_size + 1)
#         if i == 0:
#             start = 55
#         mask[i, start:start + block_size, :] = 0
#
#     return mask

def mask_single_block(data, block_size):

    mask = np.ones_like(data)

    B,T,C = data.shape

    for i in range(B):
        for c in range(C):

            start=np.random.randint(
                0,
                T-block_size+1
            )

            mask[
            i,
            start:start+block_size,
            c
            ]=0

    return mask

def estimate_latent_observability(
        ae,
        x_obs,
        mask,
        latent_steps,
        num_samples=32,
        floor=0.005,
):
    """
    Estimates how strongly each latent position is determined
    by observed values.

    Returns:
        weight: [B,L,C]
    """

    ae.eval()

    zs = []

    with torch.no_grad():

        for _ in range(num_samples):

            x_rand = x_obs.clone()

            # randomize only missing values
            rand_fill = torch.randn_like(x_rand)

            x_rand[mask == 0] = rand_fill[mask == 0]

            z = ae.encode(
                x_rand.permute(0, 2, 1)
            )

            z = z.permute(0, 2, 1)  # [B,L,C]

            zs.append(z)

    zs = torch.stack(zs, dim=0)  # [K,B,L,C]

    # latent variance under random missing fillings
    var = zs.var(dim=0)  # [B,L,C]

    # convert variance -> confidence
    weight = 1.0 / (var + 1e-6)

    # normalize to [0,1]
    weight = weight / (weight.amax(dim=(1,2), keepdim=True) + 1e-8)

    # never fully unconstrained
    weight = floor + (1.0 - floor) * weight

    return weight

def make_latent_weight(mask, latent_steps):
    """
    mask: [B, T, C]
    returns z_weight: [B, L, C]
    """

    # move to [B, C, T] for interpolation
    w = mask.permute(0, 2, 1).float()

    # downsample to latent length
    w = F.interpolate(
        w,
        size=latent_steps,
        mode="linear",
        align_corners=False
    )

    # back to [B, L, C]
    w = w.permute(0, 2, 1)

    # clamp just in case
    return w.clamp(0.0, 1.0)

# def make_latent_weight(mask, latent_steps, floor=0.01):
#     """
#     mask: [B,T,C]
#     returns: [B,L,C]
#     """
#
#     w = mask.permute(0, 2, 1).float()
#
#     # fraction of observed timesteps supporting each latent position
#     w = F.adaptive_avg_pool1d(w, latent_steps)
#
#     w = w.permute(0, 2, 1)
#
#     w = w.clamp(0.0, 1.0)
#
#     # never fully unconstrained
#     w = floor + (1.0 - floor) * w
#
#     return w

def smooth_loss_fn(x_rec, mask):
    """
    x_rec: [B, T, C]
    mask:  [B, T, C]  (1 = observed, 0 = missing)
    """

    dx = x_rec[:, 1:] - x_rec[:, :-1]  # [B, T-1, C]

    # only pairs fully inside missing region
    missing_pair = (1 - mask[:, 1:]) * (1 - mask[:, :-1])

    loss = (dx.pow(2) * missing_pair).sum()
    denom = missing_pair.sum().clamp_min(1.0)

    return loss / denom

def trend_loss_fn(x_rec, mask):
    """
    x_rec: [B, T, C]
    mask:  [B, T, C]
    """

    # pick reference mask (assumes same gap structure across batch/channels)
    m = mask[0, :, 0]  # [T]

    missing_idx = (m == 0).nonzero(as_tuple=True)[0]

    if len(missing_idx) == 0:
        return torch.tensor(0.0, device=x_rec.device)

    t0 = missing_idx[0].item()
    t1 = missing_idx[-1].item() + 1

    # need valid boundaries
    if t0 == 0 or t1 >= x_rec.shape[1]:
        return torch.tensor(0.0, device=x_rec.device)

    # boundaries
    left = x_rec[:, t0 - 1]      # [B, C]
    right = x_rec[:, t1]         # [B, C]

    # gap
    gap = x_rec[:, t0:t1]        # [B, gap_len, C]

    boundary_mean = 0.5 * (left + right)            # [B, C]
    gap_mean = gap.mean(dim=1)                      # [B, C]

    loss = ((gap_mean - boundary_mean) ** 2).mean()

    return loss

def client_get_guidance_latent(
        ae,
        x_obs,
        mask,
        latent_mean,
        latent_std,
        steps=200,
        lr=1e-2,
        prior_weight=1e-3,
        x_true=None,            # optional (for real missing eval)
        plot=True,
        plot_channels=3,
        save_dir=None,
):
    ae.eval()

    x_in = x_obs.clone()

    B, T, C = x_in.shape

    for b in range(B):
        for c in range(C):
            m = mask[b, :, c].cpu().numpy()  # 1 = observed, 0 = missing
            x = x_in[b, :, c].cpu().numpy()

            observed_idx = np.where(m == 1)[0]

            if len(observed_idx) < 2:
                continue  # not enough points to interpolate

            # interpolate missing regions
            full_idx = np.arange(T)
            x_interp = np.interp(full_idx, observed_idx, x[observed_idx])

            x_in[b, :, c] = torch.from_numpy(x_interp).to(x_in.device)

    # -------------------------
    # init latent
    # -------------------------
    with torch.no_grad():
        z_init = ae.encode(x_in.permute(0, 2, 1))   # [B,C,L]
        z_init = z_init.permute(0, 2, 1)            # [B,L,C]
        z_init_norm = (z_init - latent_mean) / latent_std

    z = z_init_norm.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    # -------------------------
    # optimize latent
    # -------------------------
    for _ in range(steps):
        opt.zero_grad()

        z_denorm = z * latent_std + latent_mean
        x_rec = ae.decode(z_denorm.permute(0, 2, 1)).permute(0, 2, 1)

        obs_loss = (((x_rec - x_obs) * mask) ** 2).sum() / mask.sum().clamp_min(1.0)
        prior_loss = ((z - z_init_norm) ** 2).mean()

        smooth_loss = smooth_loss_fn(x_rec, mask)
        trend_loss  = trend_loss_fn(x_rec, mask)

        loss = (
                obs_loss
                + prior_weight * prior_loss
                + 0.01 * smooth_loss
                + 0.01 * trend_loss
        )

        loss.backward()
        opt.step()

    # -------------------------
    # evaluation + plotting
    # -------------------------

    if save_dir is not None:
        save_dir = Path(save_dir)
    (save_dir / "plots").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # init recon
        z_init_denorm = z_init_norm * latent_std + latent_mean
        x_init_rec = ae.decode(z_init_denorm.permute(0, 2, 1)).permute(0, 2, 1)

        # final recon
        z_final_denorm = z * latent_std + latent_mean
        x_final_rec = ae.decode(z_final_denorm.permute(0, 2, 1)).permute(0, 2, 1)

        # full AE recon (no mask)
        if x_true is not None:
            z_full = ae.encode(x_true.permute(0, 2, 1))
            x_full_rec = ae.decode(z_full).permute(0, 2, 1)
        else:
            x_full_rec = None

        if x_full_rec is not None:
            ae_consistency_mse = (((x_init_rec - x_full_rec) * mask) ** 2).mean().item()
            print(f"AE consistency MSE (init vs full): {ae_consistency_mse:.6f}")

        observed = mask == 1
        missing = mask == 0

        mse_obs_init = ((x_init_rec - x_obs)[observed] ** 2).mean().item()
        mse_obs_final = ((x_final_rec - x_obs)[observed] ** 2).mean().item()
        mse_miss_latent_final = ((x_final_rec - x_full_rec)[missing] ** 2).mean().item()

        if x_true is not None and missing.sum() > 0:
            mse_miss_final = ((x_final_rec - x_true)[missing] ** 2).mean().item()
        else:
            mse_miss_final = float("nan")

        metrics = {
            "mse_obs_init": mse_obs_init,
            "mse_obs_final": mse_obs_final,
            "mse_missing_final": mse_miss_final,
            "mse_miss_latent_final": mse_miss_latent_final,
        }

        if save_dir is not None:
            import json
            with open(save_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
        else:
            print("\n[CLIENT LATENT CHECK]")
            print(metrics)

        # -------------------------
        # plotting
        # -------------------------
        if plot:
            import matplotlib.pyplot as plt

            sample_idx = 0
            C = x_obs.shape[2]
            num_plot = min(plot_channels, C)

            for ch in range(num_plot):
                true = x_true[sample_idx, :, ch].cpu().numpy() if x_true is not None else None
                obs = x_obs[sample_idx, :, ch].cpu().numpy()

                rec_init = x_init_rec[sample_idx, :, ch].cpu().numpy()
                rec_final = x_final_rec[sample_idx, :, ch].cpu().numpy()

                if x_full_rec is not None:
                    rec_full = x_full_rec[sample_idx, :, ch].cpu().numpy()

                m = mask[sample_idx, :, ch].cpu().numpy()

                rec_final_obs = rec_final.copy()
                rec_final_obs[m == 0] = np.nan

                rec_final_miss = rec_final.copy()
                rec_final_miss[m == 1] = np.nan

                obs_plot = obs.copy()
                obs_plot[m == 0] = np.nan

                plt.figure(figsize=(10, 4))

                if true is not None:
                    plt.plot(true, "--", label="ground truth", linewidth=2, color="black")

                plt.plot(obs_plot, "--", label="observed", linewidth=2)

                if x_full_rec is not None:
                    plt.plot(rec_full, label="AE full", linewidth=2, color="purple")

                plt.plot(rec_init, label="init", linewidth=2, color="orange")
                plt.plot(rec_final, label="optimized", linewidth=2, color="green")

                plt.plot(rec_final_obs, label="final (obs)", linewidth=3, color="red")
                plt.plot(rec_final_miss, label="final (miss)", linewidth=3, color="blue")

                plt.title(f"Channel {ch}")
                plt.legend()

                if save_dir is not None:
                    plt.savefig(save_dir / "plots" / f"channel_{ch}.png")
                    plt.close()
                else:
                    plt.show()

    z_weight = make_latent_weight(mask, 16)
    z_weight = (z_weight <= 0.2).float() * 0.005 + (z_weight > 0.2).float() * 1.0
    # z_weight = estimate_latent_observability(
    #     ae,
    #     x_obs,
    #     mask,
    #     latent_steps=16,
    #     num_samples=32,
    #     floor=0.005,
    # )
    # z_weight = z_weight * (z_weight > 0.2).float() * 1.0 + z_weight * (z_weight <= 0.2).float() * 0.5
    return z.detach(), z_weight

# =========================================================
# GUIDANCE
# =========================================================

def guidance_fn(z_local, z_target, z_weight):
    diff2 = (z_local - z_target) ** 2
    return (diff2 * z_weight).sum() / (z_weight.sum() + 1e-8)

def refine_latent(z_local, steps, scale, z_guidance_target, z_guidance_weights):
    z = z_local.detach()

    for _ in range(steps):
        z = z.detach().requires_grad_(True)

        loss = guidance_fn(z[:, :, guided_latent_idx], z_guidance_target, z_guidance_weights)

        grad = torch.autograd.grad(loss, z)[0]
        grad = grad / (grad.norm(dim=(1, 2), keepdim=True) + 1e-8)

        z = (z - scale * grad).detach()

    return z


# =========================================================
# GUIDED SAMPLING
# =========================================================

def sample_guided(model, num_samples, seq_len, num_channels, device,
                  z_guidance_target, z_guidance_weights, base_scale=0.05, base_repeats=10):

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

        # only pass silo-known channels
        z_local = x0_hat[:, :, guided_latent_idx]

        z_local_refined = refine_latent(
            x0_hat,
            steps,
            guidance_scale,
            z_guidance_target,
            z_guidance_weights
        )

        # merge back into full latent
        x0_guided = z_local_refined

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

    z_guidance_target, z_guidance_weights = client_get_guidance_latent(
        ae,
        x_obs[:, :, guided_idx],
        mask[:, :, guided_idx],
        latent_mean,
        latent_std,
        steps=200,
        lr=1e-2,
        x_true=torch.from_numpy(x_full_np).float().to(device)[:, :, guided_idx],
        prior_weight=1e-3,
        save_dir= scenario_dir / "client_debug",
    )

    samples = sample_guided(
        model,
        args.num_samples,
        args.latent_steps,
        num_channels,
        device,
        z_guidance_target,
        z_guidance_weights
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