import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from tslearn.metrics import dtw

from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.utils import extract


# =========================================================
# POSITIONAL ENCODING
# =========================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=4096):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1).float()

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


# =========================================================
# TRANSFORMER AE
# =========================================================

class TransformerTimeAE(nn.Module):

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

        self.input_proj = nn.Linear(
            channels,
            d_model
        )

        self.pos_enc = PositionalEncoding(
            d_model,
            max_len=seq_len
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers
        )

        self.latent_queries = nn.Parameter(
            torch.randn(latent_steps, d_model)
        )

        self.cross_attn_enc = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True
        )

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.decoder = nn.TransformerEncoder(
            dec_layer,
            num_layers=num_layers
        )

        self.cross_attn_dec = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True
        )

        self.output_proj = nn.Linear(
            d_model,
            channels
        )

    def encode(self, x):

        # [B,C,T] -> [B,T,C]
        x = x.permute(0, 2, 1)

        h = self.input_proj(x)

        h = self.pos_enc(h)

        h = self.encoder(h)

        B = h.shape[0]

        q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)

        z, _ = self.cross_attn_enc(
            q,
            h,
            h
        )

        # [B,L,D]
        return z

    def decode(self, z):

        B = z.shape[0]

        seq_queries = torch.zeros(
            B,
            self.seq_len,
            self.d_model,
            device=z.device
        )

        q = self.pos_enc(seq_queries)

        h, _ = self.cross_attn_dec(
            q,
            z,
            z
        )

        h = self.decoder(h)

        x = self.output_proj(h)

        # [B,T,C] -> [B,C,T]
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):

        z = self.encode(x)

        xhat = self.decode(z)

        return xhat, z


# =========================================================
# METRICS
# =========================================================

def safe_corr(a, b):

    if a.size == 0 or b.size == 0:
        return np.nan

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0

    return np.corrcoef(a, b)[0, 1]


def compute_full_metrics(x_pred, x_true):

    metrics = {}

    metrics["mse_full"] = float(
        ((x_pred - x_true) ** 2).mean()
    )

    metrics["mae_full"] = float(
        np.abs(x_pred - x_true).mean()
    )

    metrics["corr_full"] = float(
        safe_corr(
            x_pred.flatten(),
            x_true.flatten()
        )
    )

    try:

        metrics["r2_full"] = float(
            r2_score(
                x_true.flatten(),
                x_pred.flatten()
            )
        )

    except Exception:

        metrics["r2_full"] = np.nan

    vals = []

    for i in range(min(len(x_pred), 10)):
        for c in range(x_pred.shape[2]):

            vals.append(
                dtw(
                    x_pred[i, :, c],
                    x_true[i, :, c]
                )
            )

    metrics["dtw_full"] = float(np.mean(vals))

    pred_fft = np.abs(np.fft.fft(x_pred, axis=1))
    true_fft = np.abs(np.fft.fft(x_true, axis=1))

    metrics["spectral_mse_full"] = float(
        ((pred_fft - true_fft) ** 2).mean()
    )

    return metrics


def deterministic_plot_channels(idxs, k=8):

    return idxs[:min(k, len(idxs))]


# =========================================================
# GUIDANCE
# =========================================================

def guidance_fn(z, z_target, z_weight):

    diff2 = (z - z_target) ** 2

    return (
        (diff2 * z_weight).sum()
        / (z_weight.sum() + 1e-8)
    )


def refine_latent(
        z_full,
        steps,
        scale,
        z_guidance_target,
        z_guidance_weights,
):

    z = z_full.detach()

    for _ in range(steps):

        z = z.detach().requires_grad_(True)

        loss = guidance_fn(
            z,
            z_guidance_target,
            z_guidance_weights,
        )

        grad = torch.autograd.grad(loss, z)[0]

        grad_norm = grad.norm(
            dim=(1, 2),
            keepdim=True
        )

        grad = grad / grad_norm.clamp(min=1.0)

        z = (z - scale * grad).detach()

    return z


# =========================================================
# ENCODE SILO
# =========================================================

def encode_silo_to_norm_latent(silo, x_full):

    x_local = x_full[:, :, silo["feature_idx"]]

    with torch.no_grad():

        z = silo["ae"].encode(
            x_local.permute(0, 2, 1)
        )

        z_norm = (
                (z - silo["mean"])
                / silo["std"]
        )

    return z_norm


# =========================================================
# BUILD GUIDANCE
# =========================================================

def build_other_silo_guidance(
        x_full,
        target_silo_id,
):

    B = x_full.shape[0]

    L = args.latent_steps

    z_target_full = torch.zeros(
        B,
        L,
        num_channels,
        device=x_full.device,
    )

    z_weight_full = torch.zeros_like(
        z_target_full
    )

    for silo_id in args.silo_ids:

        s = silos[silo_id]

        idx = s["latent_idx"]

        z_local = encode_silo_to_norm_latent(
            s,
            x_full
        )

        z_target_full[:, :, idx] = z_local

        if silo_id != target_silo_id:

            z_weight_full[:, :, idx] = (
                args.anchor_weight
            )

        else:

            z_weight_full[:, :, idx] = 0.0

    return z_target_full, z_weight_full


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
        base_scale,
        base_repeats,
):

    x = torch.randn(
        num_samples,
        seq_len,
        num_channels,
        device=device
    )

    for i in reversed(range(model.timesteps)):

        print(i)

        noise = torch.randn_like(x)

        t = torch.full(
            (num_samples,),
            i,
            device=device,
            dtype=torch.long
        )

        with torch.no_grad():

            eps = model.backbone(
                x,
                t,
                None
            )

        x0_hat = model.fast_denoise(
            x,
            t,
            None,
            noise=eps
        )

        tau = i / (model.timesteps - 1)

        guidance_scale = max(
            base_scale - base_scale * tau,
            base_scale * 0.1
        )

        steps = max(
            2 * base_repeats
            - int(2 * tau * base_repeats),
            1
        )

        x0_guided = refine_latent(
            x0_hat,
            steps,
            guidance_scale,
            z_guidance_target,
            z_guidance_weights,
        )

        alpha_bar_prev = extract(
            model.alphas_cumprod_prev,
            t,
            x.shape
        )

        sqrt_ab_prev = torch.sqrt(
            alpha_bar_prev
        )

        alpha_bar = extract(
            model.alphas_cumprod,
            t,
            x.shape
        )

        sigma_t = torch.sqrt(
            (1 - alpha_bar_prev)
            / (1 - alpha_bar)
        ) * torch.sqrt(
            1 - alpha_bar / alpha_bar_prev
        )

        safe_term = torch.clamp(
            1.0
            - alpha_bar_prev
            - sigma_t ** 2,
            min=1e-8
        )

        sqrt_one_minus_ab_prev = torch.sqrt(
            safe_term
        )

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
# DECODE
# =========================================================

def decode_full_latent(z_norm):

    B = z_norm.shape[0]

    x_full = torch.zeros(
        B,
        args.orig_seq_len,
        signal_channels,
        device=z_norm.device,
    )

    for silo_id in args.silo_ids:

        s = silos[silo_id]

        z_local = z_norm[:, :, s["latent_idx"]]

        z_local = (
                z_local * s["std"]
                + s["mean"]
        )

        with torch.no_grad():

            x_local = s["ae"].decode(
                z_local
            ).permute(0, 2, 1)

        x_full[:, :, s["feature_idx"]] = x_local

    return x_full


# =========================================================
# AE RECON
# =========================================================

def ae_reconstruct_full(x_true_torch):

    x_ae_full = torch.zeros_like(
        x_true_torch
    )

    for silo_id in args.silo_ids:

        s = silos[silo_id]

        x_local = x_true_torch[
                  :,
                  :,
                  s["feature_idx"]
                  ]

        with torch.no_grad():

            z = s["ae"].encode(
                x_local.permute(0, 2, 1)
            )

            x_rec = s["ae"].decode(
                z
            ).permute(0, 2, 1)

        x_ae_full[:, :, s["feature_idx"]] = x_rec

    return x_ae_full


# =========================================================
# RUN
# =========================================================

def run_target_silo_generation(out_dir):

    scenario_dir = (
            out_dir
            / f"target_silo_{args.target_silo}"
    )

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    target_silo = silos[
        args.target_silo
    ]

    target_features = (
        target_silo["feature_idx"]
    )

    x_full = torch.from_numpy(
        x_full_np
    ).float().to(device)

    z_guidance_target, z_guidance_weights = (
        build_other_silo_guidance(
            x_full,
            target_silo_id=args.target_silo,
        )
    )

    print(
        "Guidance target:",
        tuple(z_guidance_target.shape)
    )

    print(
        "Guidance weights:",
        tuple(z_guidance_weights.shape)
    )

    print(
        "Guidance weight mean:",
        z_guidance_weights.mean().item()
    )

    print(
        "Target latent weight sum:",
        z_guidance_weights[
            :,
            :,
            target_silo["latent_idx"]
        ].sum().item()
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

    x_pred = decoded.detach().cpu().numpy()

    x_true = x_full_np

    with torch.no_grad():

        x_ae = ae_reconstruct_full(
            x_full
        ).detach().cpu().numpy()

    x_pred_target = x_pred[
                    :,
                    :,
                    target_features
                    ]

    x_true_target = x_true[
                    :,
                    :,
                    target_features
                    ]

    x_ae_target = x_ae[
                  :,
                  :,
                  target_features
                  ]

    metrics_real = compute_full_metrics(
        x_pred_target,
        x_true_target
    )

    metrics_ae = compute_full_metrics(
        x_pred_target,
        x_ae_target
    )

    observed_silos = [
        s for s in args.silo_ids
        if s != args.target_silo
    ]

    observed_feature_idx = []

    for s_id in observed_silos:

        observed_feature_idx.extend(
            silos[s_id]["feature_idx"]
        )

    observed_feature_idx = sorted(
        observed_feature_idx
    )

    x_pred_obs = x_pred[
                 :,
                 :,
                 observed_feature_idx
                 ]

    x_true_obs = x_true[
                 :,
                 :,
                 observed_feature_idx
                 ]

    x_ae_obs = x_ae[
               :,
               :,
               observed_feature_idx
               ]

    metrics_obs_real = compute_full_metrics(
        x_pred_obs,
        x_true_obs,
    )

    metrics_obs_ae = compute_full_metrics(
        x_pred_obs,
        x_ae_obs,
    )

    with open(
            scenario_dir / "metrics.txt",
            "w"
    ) as f:

        f.write(
            "=== TARGET SILO vs REAL ===\n"
        )

        for k, v in metrics_real.items():
            f.write(f"{k}: {v}\n")

        f.write(
            "\n=== TARGET SILO vs AE ===\n"
        )

        for k, v in metrics_ae.items():
            f.write(f"{k}: {v}\n")

        f.write(
            "\n=== OBSERVED SILOS vs REAL ===\n"
        )

        for k, v in metrics_obs_real.items():
            f.write(f"{k}: {v}\n")

        f.write(
            "\n=== OBSERVED SILOS vs AE ===\n"
        )

        for k, v in metrics_obs_ae.items():
            f.write(f"{k}: {v}\n")

    plot_dir = (
            scenario_dir
            / "target_feature_plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_idx = 0

    for global_ch in deterministic_plot_channels(
            target_features,
            k=args.num_plot_channels
    ):

        plt.figure(figsize=(10, 4))

        true = x_true[
            sample_idx,
            :,
            global_ch
        ]

        pred = x_pred[
            sample_idx,
            :,
            global_ch
        ]

        ae_rec = x_ae[
            sample_idx,
            :,
            global_ch
        ]

        plt.plot(
            true,
            "--",
            label="ground truth",
            linewidth=2
        )

        plt.plot(
            ae_rec,
            label="AE full recon",
            linewidth=2,
            color="orange"
        )

        plt.plot(
            pred,
            label="cross-silo generated",
            linewidth=2,
            color="firebrick"
        )

        plt.title(
            f"Target silo {args.target_silo} | feature {global_ch}"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            plot_dir / f"feature_{global_ch}.png"
        )

        plt.close()


# =========================================================
# LOAD SILO
# =========================================================

def load_silo(silo_id):

    ae_dir = (
            Path(args.silo_root_ae)
            / silo_id
    )

    latent_dir = (
            Path(args.silo_root_latents)
            / silo_id
    )

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

    model_path = (
            ae_dir
            / "models"
            / f"transformer_ae_{cfg['latent_steps']}.pt"
    )

    ae_model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    ae_model = ae_model.to(device).eval()

    mean = torch.from_numpy(
        np.load(
            latent_dir / "latent_mean.npy"
        )
    ).float().to(device)

    std = torch.from_numpy(
        np.load(
            latent_dir / "latent_std.npy"
        )
    ).float().to(device)

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

    global args
    global model
    global device
    global num_channels
    global x_full_np
    global signal_channels
    global silos

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--version",
        type=int,
        required=True
    )

    parser.add_argument(
        "--latent_steps",
        type=int,
        required=True
    )

    parser.add_argument(
        "--orig_seq_len",
        type=int,
        required=True
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=500
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    parser.add_argument(
        "--train_data",
        type=str,
        required=True
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--silo_root_ae",
        type=str,
        required=True
    )

    parser.add_argument(
        "--silo_root_latents",
        type=str,
        required=True
    )

    parser.add_argument(
        "--silo_ids",
        type=str,
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--target_silo",
        type=str,
        required=True
    )

    parser.add_argument(
        "--base_scale",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--base_repeats",
        type=int,
        default=40
    )

    parser.add_argument(
        "--anchor_weight",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--num_plot_channels",
        type=int,
        default=8
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = (
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    ckpt_path = Path(
        # f"../../../results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
        f"/media/aioannou/OS/aioannou_storage/results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
    )

    model = TSDiff.load_from_checkpoint(
        ckpt_path
    ).to(device)

    num_channels = (
        model.backbone.input_init[0].in_features
    )

    silos = {
        silo_id: load_silo(silo_id)
        for silo_id in args.silo_ids
    }

    if args.target_silo not in silos:

        raise ValueError(
            f"target_silo {args.target_silo} is not in silo_ids"
        )

    start = 0

    for silo_id in args.silo_ids:

        s = silos[silo_id]

        end = start + s["latent_dim"]

        s["latent_idx"] = list(
            range(start, end)
        )

        start = end

    if start != num_channels:

        raise ValueError(
            f"Latent mismatch: diffusion expects {num_channels}, "
            f"but silos provide {start}"
        )

    full_data = np.load(args.train_data)

    if args.num_samples > len(full_data):

        raise ValueError(
            "num_samples too large"
        )

    idx = np.random.choice(
        len(full_data),
        size=args.num_samples,
        replace=False
    )

    data = full_data[idx]

    x_full_np = data

    signal_channels = data.shape[2]

    out_dir = Path(args.out_dir)

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    run_target_silo_generation(out_dir)


if __name__ == "__main__":

    main()