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

def set_torch_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            # if i==0 and (c==0 or c==1):
            #     start =45
            mask[i, start:start + block_size, c] = 0

    return mask

def mask_forecast(data, block_size):
    mask = np.ones_like(data)
    B, T, C = data.shape

    for i in range(B):
        for c in range(C):
            start = T - block_size
            # if i==0 and (c==0 or c==1):
            #     start =45
            mask[i, start:start + block_size, c] = 0

    return mask



# =========================================================
# LATENT WEIGHT / OBSERVABILITY
# =========================================================
def make_latent_weight_interp(mask, latent_steps, latent_dim, floor=0.005):
    """
    Method 1: fast mask interpolation.

    mask: [B,T,C_local]
    returns: [B,L,D]
    """

    # collapse feature mask to one confidence signal
    w = mask.float().mean(dim=2, keepdim=True)  # [B,T,1]

    # [B,T,1] -> [B,1,T]
    w = w.permute(0, 2, 1)

    w = F.interpolate(
        w,
        size=latent_steps,
        mode="linear",
        align_corners=False
    )

    # [B,1,L] -> [B,L,1]
    w = w.permute(0, 2, 1)

    # expand to transformer embedding dimension
    w = w.expand(-1, -1, latent_dim)

    w = w.clamp(0.0, 1.0)

    return floor + (1.0 - floor) * w

def estimate_latent_observability(
        ae,
        x_obs,
        mask,
        latent_steps,
        num_samples=32,
        floor=0.005,
):
    """
    Transformer AE version.

    x_obs: [B,T,C_local]
    mask:  [B,T,C_local]

    Returns:
        weight: [B,L,D]
    """
    ae.eval()

    zs = []

    with torch.no_grad():
        for _ in range(num_samples):
            x_rand = x_obs.clone()

            rand_fill = torch.randn_like(x_rand)
            x_rand[mask == 0] = rand_fill[mask == 0]

            z = ae.encode(x_rand.permute(0, 2, 1))  # [B,L,D]

            zs.append(z)

    zs = torch.stack(zs, dim=0)  # [K,B,L,D]
    var = zs.var(dim=0)          # [B,L,D]

    weight = 1.0 / (var + 1e-6)
    weight = weight / (weight.amax(dim=(1, 2), keepdim=True) + 1e-8)
    weight = floor + (1.0 - floor) * weight

    return weight


# =========================================================
# CLIENT LOSSES
# =========================================================

def smooth_loss_fn(x_rec, mask):
    dx = x_rec[:, 1:] - x_rec[:, :-1]
    missing_pair = (1 - mask[:, 1:]) * (1 - mask[:, :-1])

    loss = (dx.pow(2) * missing_pair).sum()
    denom = missing_pair.sum().clamp_min(1.0)

    return loss / denom


def trend_loss_fn(x_rec, mask):
    m = mask[0, :, 0]
    missing_idx = (m == 0).nonzero(as_tuple=True)[0]

    if len(missing_idx) == 0:
        return torch.tensor(0.0, device=x_rec.device)

    t0 = missing_idx[0].item()
    t1 = missing_idx[-1].item() + 1

    if t0 == 0 or t1 >= x_rec.shape[1]:
        return torch.tensor(0.0, device=x_rec.device)

    left = x_rec[:, t0 - 1]
    right = x_rec[:, t1]
    gap = x_rec[:, t0:t1]

    boundary_mean = 0.5 * (left + right)
    gap_mean = gap.mean(dim=1)

    return ((gap_mean - boundary_mean) ** 2).mean()


# =========================================================
# ENCODING HELPERS
# =========================================================

def encode_full_silo_to_norm_latent(silo, x_full):
    """
    Encode one silo's full observed features into normalized transformer latent.

    x_full: [B,T,C_total]
    returns: [B,L,D]
    """
    x_local = x_full[:, :, silo["feature_idx"]]

    with torch.no_grad():
        z = silo["ae"].encode(x_local.permute(0, 2, 1))  # [B,L,D]
        z_norm = (z - silo["mean"]) / silo["std"]

    return z_norm


def client_get_partial_guidance_latent(
        ae,
        x_obs,
        mask,
        latent_mean,
        latent_std,
        steps=200,
        lr=1e-2,
        prior_weight=1e-3,
        x_true=None,
        plot=True,
        plot_channels=3,
        save_dir=None,
):
    """
    Target silo local anchor from partially observed local features.

    Returns:
        z_target_local: [B,L,C_target]
        z_weight_local: [B,L,C_target]
    """
    ae.eval()

    x_in = x_obs.clone()
    B, T, C = x_in.shape

    # interpolation init using observed values only
    for b in range(B):
        for c in range(C):
            m = mask[b, :, c].detach().cpu().numpy()
            x = x_in[b, :, c].detach().cpu().numpy()

            observed_idx = np.where(m == 1)[0]
            if len(observed_idx) < 2:
                continue

            full_idx = np.arange(T)
            x_interp = np.interp(full_idx, observed_idx, x[observed_idx])
            x_in[b, :, c] = torch.from_numpy(x_interp).to(x_in.device)

    # init latent
    with torch.no_grad():
        z_init = ae.encode(x_in.permute(0, 2, 1))  # [B,L,D]
        z_init_norm = (z_init - latent_mean) / latent_std

    z = z_init_norm.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)

    # optimize local target latent against observed target values only
    for _ in range(steps):
        opt.zero_grad()

        z_denorm = z * latent_std + latent_mean
        x_rec = ae.decode(z_denorm).permute(0, 2, 1)

        obs_loss = (((x_rec - x_obs) * mask) ** 2).sum() / mask.sum().clamp_min(1.0)
        prior_loss = ((z - z_init_norm) ** 2).mean()
        smooth_loss = smooth_loss_fn(x_rec, mask)
        trend_loss = trend_loss_fn(x_rec, mask)

        loss = obs_loss + prior_weight * prior_loss + 0.01 * smooth_loss + 0.01 * trend_loss
        loss.backward()
        opt.step()

    if save_dir is not None:
        save_dir = Path(save_dir)
        (save_dir / "plots").mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        z_init_denorm = z_init_norm * latent_std + latent_mean
        x_init_rec = ae.decode(z_init_denorm).permute(0, 2, 1)

        z_final_denorm = z * latent_std + latent_mean
        x_final_rec = ae.decode(z_final_denorm).permute(0, 2, 1)

        if x_true is not None:
            z_full = ae.encode(x_true.permute(0, 2, 1))
            x_full_rec = ae.decode(z_full).permute(0, 2, 1)
        else:
            x_full_rec = None

        observed = mask == 1
        missing = mask == 0

        mse_obs_init = ((x_init_rec - x_obs)[observed] ** 2).mean().item()
        mse_obs_final = ((x_final_rec - x_obs)[observed] ** 2).mean().item()

        if x_true is not None and missing.sum() > 0:
            mse_missing_final = ((x_final_rec - x_true)[missing] ** 2).mean().item()
        else:
            mse_missing_final = float("nan")

        metrics = {
            "mse_obs_init": mse_obs_init,
            "mse_obs_final": mse_obs_final,
            "mse_missing_final": mse_missing_final,
        }

        if x_full_rec is not None:
            metrics["ae_consistency_mse_init_vs_full_observed"] = (
                (((x_init_rec - x_full_rec) * mask) ** 2).sum()
                / mask.sum().clamp_min(1.0)
            ).item()

        if save_dir is not None:
            with open(save_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
        else:
            print("\n[CLIENT LATENT CHECK]")
            print(metrics)

        if plot:
            sample_idx = 0
            num_plot = min(plot_channels, C)

            for ch in range(num_plot):
                true = x_true[sample_idx, :, ch].cpu().numpy() if x_true is not None else None
                obs = x_obs[sample_idx, :, ch].cpu().numpy()
                rec_init = x_init_rec[sample_idx, :, ch].cpu().numpy()
                rec_final = x_final_rec[sample_idx, :, ch].cpu().numpy()
                rec_full = x_full_rec[sample_idx, :, ch].cpu().numpy() if x_full_rec is not None else None
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
                if rec_full is not None:
                    plt.plot(rec_full, label="AE full", linewidth=2, color="purple")
                plt.plot(rec_init, label="init", linewidth=2, color="orange")
                plt.plot(rec_final, label="optimized", linewidth=2, color="green")
                plt.plot(rec_final_obs, label="final obs", linewidth=3, color="red")
                plt.plot(rec_final_miss, label="final missing", linewidth=3, color="blue")
                plt.title(f"Target silo anchor channel {ch}")
                plt.legend()

                if save_dir is not None:
                    plt.savefig(save_dir / "plots" / f"channel_{ch}.png")
                    plt.close()
                else:
                    plt.show()

    if args.latent_keep_percent == -1:
        # No guidance: all weights are exactly 0.
        z_weight = torch.zeros_like(z)

        print("Mask type: no guidance")
        print("Keeping 0.00% of latent entries")

    elif args.mask_type == "full":
        # Full guidance: all weights are exactly 1.
        z_weight = torch.ones_like(z)

        print("Mask type: full")
        print("Keeping 100.00% of latent entries")

    elif args.mask_type == "random":
        # Random guidance: exactly latent_keep_percent entries are 1,
        # and all remaining entries are 0.
        if not 0.0 <= args.latent_keep_percent <= 1.0:
            raise ValueError(
                f"latent_keep_percent must be between 0 and 1, "
                f"got {args.latent_keep_percent}"
            )

        z_weight = torch.zeros_like(z)
        flat_weight = z_weight.flatten()

        num_keep = round(args.latent_keep_percent * flat_weight.numel())

        if num_keep > 0:
            random_indices = torch.randperm(
                flat_weight.numel(),
                device=z.device,
            )[:num_keep]

            flat_weight[random_indices] = 1.0

        z_weight = flat_weight.reshape_as(z)

        print("Mask type: random")
        print(
            f"Keeping {(z_weight.mean().item() * 100):.2f}% "
            f"of latent entries as 1"
        )

    elif args.mask_type in ["variance", "interpolation"]:
        if not 0.0 <= args.latent_keep_percent <= 1.0:
            raise ValueError(
                f"latent_keep_percent must be between 0 and 1, "
                f"got {args.latent_keep_percent}"
            )

        if args.mask_type == "interpolation":
            z_weight = make_latent_weight_interp(
                mask,
                latent_steps=z.shape[1],
                latent_dim=z.shape[2],
                floor=0.005,
            )
        else:
            z_weight = estimate_latent_observability(
                ae,
                x_obs,
                mask,
                latent_steps=z.shape[1],
                num_samples=32,
                floor=0.005,
            )

        threshold = torch.quantile(
            z_weight.flatten(),
            1.0 - args.latent_keep_percent,
        )

        keep_mask = z_weight >= threshold

        print(f"Mask type: {args.mask_type}")
        print(
            f"Keeping {(keep_mask.float().mean().item() * 100):.2f}% "
            f"of latent weights unchanged "
            f"(threshold={threshold.item():.4f})"
        )

        z_weight = torch.where(
            keep_mask,
            z_weight,
            z_weight * 0.001,
        )

    else:
        raise ValueError(
            f"Unknown mask_type '{args.mask_type}'. "
            "Expected variance, interpolation, random, or full."
        )

    print("target z_weight mean:", z_weight.mean().item())

    return z.detach(), z_weight


# =========================================================
# BUILD HYBRID GUIDANCE
# =========================================================

def build_guidance(x_full, x_obs, mask):

    z_target, z_weight = client_get_partial_guidance_latent(
        ae,
        x_obs,
        mask,
        latent_mean,
        latent_std,
        steps=args.client_steps,
        lr=args.client_lr,
        prior_weight=args.client_prior_weight,
        x_true=x_full,
        plot=True,
        plot_channels=args.num_client_plot_channels,
        save_dir=args.current_scenario_dir / "client_debug",
    )

    print("Guidance target:", tuple(z_target.shape))
    print("Guidance weights:", tuple(z_weight.shape))
    print("Guidance weight mean:", z_weight.mean().item())

    return z_target, z_weight


# =========================================================
# GUIDANCE
# =========================================================

def guidance_fn(z, z_target, z_weight):
    diff2 = (z - z_target) ** 2
    return (diff2 * z_weight).sum() / (z_weight.sum() + 1e-8)


def refine_latent(z_full, steps, scale, z_guidance_target, z_guidance_weights):
    z = z_full.detach()

    for _ in range(steps):
        z = z.detach().requires_grad_(True)

        loss = guidance_fn(z, z_guidance_target, z_guidance_weights)

        grad = torch.autograd.grad(loss, z)[0]
        grad = grad / (grad.norm(dim=(1, 2), keepdim=True) + 1e-8)

        z = (z - scale * grad).detach()

    return z


# =========================================================
# GUIDED SAMPLING
# =========================================================

def sample_guided(
        model,
        num_samples,
        seq_len,
        num_channels,
        device,
        z_guidance_target,
        z_guidance_weights,
        base_scale=1.0,
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

        x0_guided = refine_latent(
            x0_hat,
            steps,
            guidance_scale,
            z_guidance_target,
            z_guidance_weights,
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

    set_torch_seed(args.inference_seed)

    z_guidance_target, z_guidance_weights = build_guidance(
        x_full=x_full,
        x_obs=x_obs,
        mask=mask,
    )

    set_torch_seed(args.inference_seed + 1_000_000)

    all_preds = []

    for k in range(args.num_imputation_samples):

        print(
            f"Generating sample {k+1}/{args.num_imputation_samples}"
        )

        samples = sample_guided(
            model,
            args.num_samples,
            args.latent_steps,
            num_channels,
            device,
            z_guidance_target,
            z_guidance_weights,
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
        # f"../../../results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
        f"/media/aioannou/OS/aioannou_storage/results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
    )
    model = TSDiff.load_from_checkpoint(ckpt_path).to(device)

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
