import argparse
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
        # start = np.random.randint(0, T - block_size + 1)
        start = 20
        mask[i, start:start + block_size, :] = 0

    return mask


# =========================================================
# GUIDANCE FUNCTION
# =========================================================

def guidance_fn(x_recon):
    obs_count = mask.sum(dim=(1, 2), keepdim=True) + 1e-8
    guidance_loss = (((x_recon - x_obs) * mask) ** 2).sum(dim=(1, 2)) / obs_count.squeeze()
    guidance_loss = guidance_loss.mean()

    dx_rec = x_recon[:, 1:] - x_recon[:, :-1]

    # first derivative interior smoothness
    missing_pair = (1 - mask[:, 1:]) * (1 - mask[:, :-1])
    smooth_loss = (dx_rec.pow(2) * missing_pair).sum() / (missing_pair.sum().clamp_min(1.0))

    # second derivative interior smoothness
    d2x = x_recon[:, 2:] - 2 * x_recon[:, 1:-1] + x_recon[:, :-2]
    missing_triplet = (1 - mask[:, 2:]) * (1 - mask[:, 1:-1]) * (1 - mask[:, :-2])
    curv_loss = (d2x.pow(2) * missing_triplet).sum() / (missing_triplet.sum().clamp_min(1.0))

    # print(guidance_loss, smooth_loss)
    return guidance_loss + 0.05 * smooth_loss

# =========================================================
# GUIDED SAMPLING
# =========================================================

def sample_guided(model, decoder, num_samples, seq_len, num_channels, device, latent_mean, latent_std, base_scale=1, base_repeats=10):

    x = torch.randn(num_samples, seq_len, num_channels, device=device)

    for i in reversed(range(model.timesteps)):
        tau = i / (model.timesteps - 1)
        guidance_scale = base_scale * (4 * tau * (1 - tau))
        noise = torch.randn_like(x)
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)

        with torch.no_grad():
            eps = model.backbone(x, t, None)

        x0_hat = model.fast_denoise(x, t, None, noise=eps)

        x0_guided = x0_hat.detach().requires_grad_(True)

        for _ in range(int((((4 * tau * (1 - tau)) + 1) * base_repeats))):
            x0_guided = x0_guided.detach().requires_grad_(True)

            x0_guided_denorm = x0_guided * latent_std + latent_mean
            x_recon = decoder(x0_guided_denorm)

            loss = guidance_fn(x_recon)

            grad = torch.autograd.grad(loss, x0_guided)[0]
            grad = grad / (grad.norm(dim=(1, 2), keepdim=True) + 1e-8)

            step = guidance_scale * grad
            x0_guided = (x0_guided - step).detach()

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
# EVALUATION
# =========================================================

def compute_oscillation_scores(x_true, mask):
    """
    Returns one score per (sample, channel) based on hidden region.
    Higher = more oscillatory
    """
    B, T, C = x_true.shape
    scores = np.zeros((B, C))

    for i in range(B):
        for c in range(C):
            m = mask[i, :, c] == 0
            if m.sum() < 5:
                continue

            x = x_true[i, :, c][m]

            dx = np.diff(x)
            scores[i, c] = np.mean(dx ** 2)  # simple but effective

    return scores

def run_scenario(name, mask_np, out_dir):

    global mask, x_obs

    scenario_dir = out_dir / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    x_obs_np = x_full_np.copy()
    x_obs_np[mask_np == 0] = 0.0

    mask = torch.from_numpy(mask_np).float().to(device)
    x_obs = torch.from_numpy(x_obs_np).float().to(device)

    samples = sample_guided(
        model,
        decoder,
        args.num_samples,
        args.latent_steps,
        num_channels,
        device,
        latent_mean,
        latent_std
    )

    samples_denorm = samples * latent_std + latent_mean
    decoded = decoder(samples_denorm)

    x_pred = decoded.detach().cpu().numpy()
    x_true = x_full_np

    osc_scores = compute_oscillation_scores(x_true, mask_np)

    # flatten valid entries
    valid = osc_scores > 0
    vals = osc_scores[valid]

    low_thr = np.percentile(vals, 30)
    high_thr = np.percentile(vals, 70)

    smooth_mask = osc_scores <= low_thr
    osc_mask = osc_scores >= high_thr

    missing = (mask_np == 0)
    observed = (mask_np == 1)

    mse_missing = ((x_pred - x_true)[missing] ** 2).mean()
    mse_observed = ((x_pred - x_true)[observed] ** 2).mean()

    mae_missing = np.abs(x_pred - x_true)[missing].mean()

    def safe_corr(a, b):
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    corr_missing = safe_corr(x_pred[missing], x_true[missing])

    r2_missing = r2_score(x_true[missing].flatten(), x_pred[missing].flatten())

    def compute_dtw_batch(x_pred, x_true, mask):
        vals = []
        for i in range(min(len(x_pred), 10)):
            for c in range(x_pred.shape[2]):
                m = mask[i, :, c] == 0
                if m.sum() < 5:
                    continue
                vals.append(dtw(x_pred[i, :, c][m], x_true[i, :, c][m]))
        return np.mean(vals) if vals else 0.0

    dtw_missing = compute_dtw_batch(x_pred, x_true, mask_np)

    def spectral_mse(x_pred, x_true, mask):
        pred_fft = np.abs(np.fft.fft(x_pred, axis=1))
        true_fft = np.abs(np.fft.fft(x_true, axis=1))
        return ((pred_fft - true_fft)[mask == 0] ** 2).mean()

    spec_mse = spectral_mse(x_pred, x_true, mask_np)

    if name == "single_block":
        def compute_subset_metrics(subset_mask):

            sel = []
            for i in range(x_pred.shape[0]):
                for c in range(x_pred.shape[2]):
                    if subset_mask[i, c]:
                        m = mask_np[i, :, c] == 0
                        if m.sum() < 5:
                            continue

                        pred = x_pred[i, :, c][m]
                        true = x_true[i, :, c][m]

                        sel.append((pred, true))

            if len(sel) == 0:
                return {}

            preds = np.concatenate([p for p, _ in sel])
            trues = np.concatenate([t for _, t in sel])

            mse = ((preds - trues) ** 2).mean()

            if np.std(preds) < 1e-8 or np.std(trues) < 1e-8:
                corr = 0.0
            else:
                corr = np.corrcoef(preds, trues)[0, 1]

            pred_fft = np.abs(np.fft.fft(preds))
            true_fft = np.abs(np.fft.fft(trues))
            spec = ((pred_fft - true_fft) ** 2).mean()

            return {
                "mse": float(mse),
                "corr": float(corr),
                "spec_mse": float(spec),
                "count": len(sel),
            }


        smooth_metrics = compute_subset_metrics(smooth_mask)
        osc_metrics = compute_subset_metrics(osc_mask)

    with open(scenario_dir / "metrics.txt", "w") as f:
        f.write(f"MSE missing: {mse_missing}\n")
        f.write(f"MSE observed: {mse_observed}\n")
        f.write(f"MAE missing: {mae_missing}\n")
        f.write(f"Correlation: {corr_missing}\n")
        f.write(f"R2: {r2_missing}\n")
        f.write(f"DTW: {dtw_missing}\n")
        f.write(f"Spectral MSE: {spec_mse}\n")

        if name == "single_block":
            f.write("\n--- Smooth regions ---\n")
            for k, v in smooth_metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\n--- Oscillatory regions ---\n")
            for k, v in osc_metrics.items():
                f.write(f"{k}: {v}\n")

    # plots for index 0
    sample_idx = 0

    num_channels_to_plot = min(5, x_pred.shape[2])

    for ch in range(num_channels_to_plot):

        plt.figure(figsize=(10, 4))

        observed = x_obs_np[sample_idx, :, 2*ch]
        true = x_true[sample_idx, :, 2*ch]
        pred = x_pred[sample_idx, :, 2*ch]
        m = mask_np[sample_idx, :, 2*ch]

        # plot observed and ground truth
        plt.plot(true, '--', label="ground truth", linewidth=2)

        # split prediction into observed vs missing regions
        pred_obs = pred.copy()
        pred_obs[m == 0] = np.nan

        pred_miss = pred.copy()
        pred_miss[m == 1] = np.nan

        # plot prediction in two colors
        plt.plot(pred, color="gray", alpha=0.4, linewidth=2, label="prediction")
        plt.plot(pred_obs, label="pred (observed region)", linewidth=2)
        plt.plot(pred_miss, label="pred (missing region)", linewidth=2)

        plt.title(f"Sample {sample_idx} | Channel {2*ch}")
        plt.legend()

        plt.savefig(scenario_dir / f"channel_{2*ch}.png")
        plt.close()


# =========================================================
# MAIN
# =========================================================

def main():
    global args, model, decoder, device, num_channels
    global latent_mean, latent_std, x_full_np

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--decoder_path", type=str, required=True)
    parser.add_argument("--latent_steps", type=int, required=True)
    parser.add_argument("--orig_seq_len", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_data", type=str, required=True)

    parser.add_argument("--missing_ratio", type=float, required=True)
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "random", "multi_block", "single_block"])

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--latent_mean_path", type=str, required=True)
    parser.add_argument("--latent_std_path", type=str, required=True)

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(f"../../../results/lightning_logs/version_{args.version}/checkpoints/last.ckpt")
    model = TSDiff.load_from_checkpoint(ckpt_path).to(device)

    num_channels = model.backbone.input_init[0].in_features

    ae = SiloTimeOnlyAE(num_channels, args.orig_seq_len, args.latent_steps)
    ae.load_state_dict(torch.load(args.decoder_path, map_location=device))
    ae = ae.to(device).eval()

    def decoder(z):
        z = z.permute(0, 2, 1)
        x = ae.decode(z)
        return x.permute(0, 2, 1)

    latent_mean = torch.from_numpy(np.load(args.latent_mean_path)).float().to(device)
    latent_std = torch.from_numpy(np.load(args.latent_std_path)).float().to(device)

    data = np.load(args.train_data)[:args.num_samples]
    x_full_np = data

    T = data.shape[1]
    total_missing = int(T * args.missing_ratio)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scenario in ["all", "random"]:
        mask_np = mask_random(data, total_missing)
        run_scenario("random", mask_np, out_dir)

    if args.scenario in ["all", "multi_block"]:
        mask_np = mask_multi_block(data, total_missing // 3, 3)
        run_scenario("multi_block", mask_np, out_dir)

    if args.scenario in ["all", "single_block"]:
        mask_np = mask_single_block(data, total_missing)
        run_scenario("single_block", mask_np, out_dir)


if __name__ == "__main__":
    main()