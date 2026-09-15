import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 500
MISSING_RATIO = 0.5
SEED = 42
INFERENCE_SEED = 1
CLIENT_STEPS = 200
CLIENT_LR = 1e-2
CLIENT_PRIOR_WEIGHT = 1e-3

# Run this from:
# unconditional-time-series-diffusion/src/uncond_ts_diff/sampler
OUTPUT_ROOT = Path("../../../results/anchor_check")

DATASETS = {
    "metro": {
        "seq_len": 168,
        "test_data": "../../../../data/diffusion_ready/metro/test_seq.npy",
        "ae_root": "../../../../autoencoders/out/metro/full",
        "latents_root": "../../../../data/diffusion_ready/metro/full",
    },
    "tep": {
        "seq_len": 64,
        "test_data": "../../../../data/diffusion_ready/tep/64/test_seq.npy",
        "ae_root": "../../../../autoencoders/out/tep/full",
        "latents_root": "../../../../data/diffusion_ready/tep/full",
    },
    "appliances": {
        "seq_len": 168,
        "test_data": "../../../../data/diffusion_ready/appliances/test_seq_no_time.npy",
        "ae_root": "../../../../autoencoders/out/appliances/full",
        "latents_root": "../../../../data/diffusion_ready/appliances/full",
    },
    "air_quality": {
        "seq_len": 168,
        "test_data": "../../../../data/diffusion_ready/air_quality/test_seq_no_time.npy",
        "ae_root": "../../../../autoencoders/out/air_quality/full",
        "latents_root": "../../../../data/diffusion_ready/air_quality/full",
    },
    "har": {
        "seq_len": 128,
        "test_data": "../../../../data/diffusion_ready/har/test_seq.npy",
        "ae_root": "../../../../autoencoders/out/har/full",
        "latents_root": "../../../../data/diffusion_ready/har/full",
    },
}

SCENARIOS = ["single_block", "blackout", "forecast", "random"]
ANCHOR_TYPES = ["interpolation", "recovered"]


# =========================================================
# AUTOENCODER -- same architecture as sample_multiple.py
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerTimeAE(nn.Module):
    def __init__(self, channels, seq_len, latent_steps=16, d_model=128, nhead=8, num_layers=4):
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
        x = x.permute(0, 2, 1)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        B = h.shape[0]
        q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        z, _ = self.cross_attn_enc(q, h, h)
        return z

    def decode(self, z):
        B = z.shape[0]
        seq_queries = torch.zeros(B, self.seq_len, self.d_model, device=z.device)
        q = self.pos_enc(seq_queries)
        h, _ = self.cross_attn_dec(q, z, z)
        h = self.decoder(h)
        x = self.output_proj(h)
        return x.permute(0, 2, 1)


# =========================================================
# MASKS -- copied from sample_multiple.py
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
            mask[i, start:start + block_size, c] = 0
    return mask


def mask_forecast(data, block_size):
    mask = np.ones_like(data)
    B, T, C = data.shape
    for i in range(B):
        for c in range(C):
            # Intentionally preserves your existing code exactly.
            start = T - block_size + 1
            mask[i, start:start + block_size, c] = 0
    return mask


# =========================================================
# RECOVERED-ANCHOR LOSSES -- copied from sample_multiple.py
# =========================================================

def smooth_loss_fn(x_rec, mask):
    dx = x_rec[:, 1:] - x_rec[:, :-1]
    missing_pair = (1 - mask[:, 1:]) * (1 - mask[:, :-1])
    loss = (dx.pow(2) * missing_pair).sum()
    denom = missing_pair.sum().clamp_min(1.0)
    return loss / denom


def get_trend_intervals(mask, min_gap_ratio=0.05):
    intervals = []
    B, T, C = mask.shape
    min_gap_length = max(2, math.ceil(min_gap_ratio * T))

    with torch.no_grad():
        for b in range(B):
            for c in range(C):
                missing = (mask[b, :, c] == 0).detach().cpu().numpy()
                padded = np.pad(missing.astype(np.int8), (1, 1))
                changes = np.diff(padded)
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0] - 1

                for start, end in zip(starts, ends):
                    gap_length = end - start + 1
                    if start == 0 or end == T - 1 or gap_length < min_gap_length:
                        continue
                    intervals.append((b, c, int(start), int(end)))

    return intervals


def trend_loss_fn(x_rec, intervals):
    if not intervals:
        return x_rec.new_tensor(0.0)

    losses = []
    for b, c, start, end in intervals:
        gap_mean = x_rec[b, start:end + 1, c].mean()
        boundary_mean = 0.5 * (x_rec[b, start - 1, c] + x_rec[b, end + 1, c])
        losses.append((gap_mean - boundary_mean).pow(2))

    return torch.stack(losses).mean()


# =========================================================
# LOAD MODEL / BUILD ANCHOR
# =========================================================

def load_autoencoder(root, latents_root, signal_channels, seq_len):
    root = Path(root)
    latents_root = Path(latents_root)

    with open(root / "config.json") as f:
        cfg = json.load(f)

    ae = TransformerTimeAE(
        channels=signal_channels if cfg.get("feature_idx") == "all" else len(cfg["feature_idx"]),
        seq_len=seq_len,
        latent_steps=cfg["latent_steps"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
    )

    ae.load_state_dict(
        torch.load(
            root / "models" / f"transformer_ae_{cfg['latent_steps']}.pt",
            map_location=DEVICE,
        )
    )
    ae = ae.to(DEVICE).eval()

    latent_mean = torch.from_numpy(np.load(latents_root / "latent_mean.npy")).float().to(DEVICE)
    latent_std = torch.from_numpy(np.load(latents_root / "latent_std.npy")).float().to(DEVICE)

    return ae, latent_mean, latent_std


def make_interpolation_initialization(ae, x_obs, mask, latent_mean, latent_std):
    x_in = x_obs.clone()
    B, T, C = x_in.shape

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

    with torch.no_grad():
        z_init = ae.encode(x_in.permute(0, 2, 1))
        z_init_norm = (z_init - latent_mean) / latent_std

    return z_init_norm


def decode_normalized_latent(ae, z_norm, latent_mean, latent_std):
    with torch.no_grad():
        z = z_norm * latent_std + latent_mean
        return ae.decode(z).permute(0, 2, 1)


def recover_anchor(ae, z_init_norm, x_obs, mask, latent_mean, latent_std):
    z = z_init_norm.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=CLIENT_LR)
    trend_intervals = get_trend_intervals(mask, min_gap_ratio=0.05)

    for step in range(CLIENT_STEPS):
        opt.zero_grad()

        z_denorm = z * latent_std + latent_mean
        x_rec = ae.decode(z_denorm).permute(0, 2, 1)

        obs_loss = (((x_rec - x_obs) * mask) ** 2).sum() / mask.sum().clamp_min(1.0)
        prior_loss = ((z - z_init_norm) ** 2).mean()
        smooth_loss = smooth_loss_fn(x_rec, mask)
        trend_loss = trend_loss_fn(x_rec, trend_intervals)

        loss = (
            obs_loss
            + CLIENT_PRIOR_WEIGHT * prior_loss
            + 0.01 * smooth_loss
            + 0.01 * trend_loss
        )

        loss.backward()
        opt.step()

        if (step + 1) % 25 == 0 or step == 0:
            print(
                f"    Adam {step + 1:3d}/{CLIENT_STEPS} | "
                f"loss={loss.item():.6f} | obs={obs_loss.item():.6f}"
            )

    return z.detach()


# =========================================================
# METRICS
# =========================================================

def summarize_per_sample(values):
    values = values.detach().cpu().double().numpy()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "sem": float(values.std(ddof=1) / np.sqrt(len(values))),
        "n": int(len(values)),
    }


def compute_metrics(x_anchor, x_true, mask):
    sq = (x_anchor - x_true).pow(2)
    observed = mask
    missing = 1.0 - mask

    obs_count = observed.sum(dim=(1, 2)).clamp_min(1.0)
    miss_count = missing.sum(dim=(1, 2)).clamp_min(1.0)

    mse_obs_per_sample = (sq * observed).sum(dim=(1, 2)) / obs_count
    mse_missing_per_sample = (sq * missing).sum(dim=(1, 2)) / miss_count

    observed_bool = mask.bool()
    missing_bool = ~observed_bool

    return {
        "mse_observed": summarize_per_sample(mse_obs_per_sample),
        "mse_missing": summarize_per_sample(mse_missing_per_sample),
        "mse_observed_pooled": float(sq[observed_bool].mean().item()),
        "mse_missing_pooled": float(sq[missing_bool].mean().item()),
    }


def save_json_atomic(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def write_summary_csv():
    rows = []

    for path in sorted(OUTPUT_ROOT.glob("*/*/*.json")):
        with open(path) as f:
            r = json.load(f)

        rows.append({
            "dataset": r["dataset"],
            "scenario": r["scenario"],
            "anchor_type": r["anchor_type"],
            "mse_observed_mean": r["mse_observed"]["mean"],
            "mse_observed_std": r["mse_observed"]["std"],
            "mse_observed_sem": r["mse_observed"]["sem"],
            "mse_missing_mean": r["mse_missing"]["mean"],
            "mse_missing_std": r["mse_missing"]["std"],
            "mse_missing_sem": r["mse_missing"]["sem"],
            "mse_observed_pooled": r["mse_observed_pooled"],
            "mse_missing_pooled": r["mse_missing_pooled"],
            "num_samples": r["num_samples"],
            "actual_missing_ratio": r["actual_missing_ratio"],
        })

    if not rows:
        return

    dataset_order = {"appliances": 0, "har": 1, "metro": 2, "tep": 3, "air_quality": 4}
    scenario_order = {"single_block": 0, "forecast": 1, "random": 2, "blackout": 3}
    anchor_order = {"recovered": 0, "interpolation": 1}

    rows.sort(
        key=lambda r: (
            dataset_order.get(r["dataset"], 99),
            scenario_order.get(r["scenario"], 99),
            anchor_order.get(r["anchor_type"], 99),
        )
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_ROOT / "anchor_mse_summary.csv"

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated summary: {out}")


# =========================================================
# ONE DATASET / SCENARIO
# =========================================================

def run_case(dataset, scenario, cfg):
    interpolation_file = OUTPUT_ROOT / dataset / scenario / "interpolation.json"
    recovered_file = OUTPUT_ROOT / dataset / scenario / "recovered.json"

    need_interpolation = not interpolation_file.exists()
    need_recovered = not recovered_file.exists()

    if not need_interpolation and not need_recovered:
        print(f"SKIPPING completed: {dataset} / {scenario}")
        return

    print(f"\n===== {dataset} | {scenario} =====")

    # IMPORTANT: reset exactly as a fresh sample_multiple.py invocation.
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    full_data = np.load(cfg["test_data"])
    idx = np.random.choice(len(full_data), size=NUM_SAMPLES, replace=False)
    data = full_data[idx]

    T = data.shape[1]
    total_missing = int(T * MISSING_RATIO)

    if scenario == "random":
        mask_np = mask_random(data, total_missing)
    elif scenario == "blackout":
        mask_np = mask_blackout(data, total_missing)
    elif scenario == "single_block":
        mask_np = mask_single_block(data, total_missing)
    elif scenario == "forecast":
        mask_np = mask_forecast(data, total_missing)
    else:
        raise ValueError(scenario)

    x_full = torch.from_numpy(data).float().to(DEVICE)
    mask = torch.from_numpy(mask_np).float().to(DEVICE)
    x_obs = x_full.clone()
    x_obs[mask == 0] = 0.0

    ae, latent_mean, latent_std = load_autoencoder(
        cfg["ae_root"],
        cfg["latents_root"],
        data.shape[2],
        cfg["seq_len"],
    )

    # Same reset position as set_torch_seed(args.inference_seed) before build_guidance().
    torch.manual_seed(INFERENCE_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(INFERENCE_SEED)

    z_init_norm = make_interpolation_initialization(
        ae, x_obs, mask, latent_mean, latent_std
    )

    if need_interpolation:
        x_interp_anchor = decode_normalized_latent(
            ae, z_init_norm, latent_mean, latent_std
        )
        metrics = compute_metrics(x_interp_anchor, x_full, mask)
        result = {
            "dataset": dataset,
            "scenario": scenario,
            "anchor_type": "interpolation",
            "num_samples": NUM_SAMPLES,
            "seed": SEED,
            "inference_seed": INFERENCE_SEED,
            "client_steps": 0,
            "actual_missing_ratio": float((mask == 0).float().mean().item()),
            **metrics,
        }
        save_json_atomic(result, interpolation_file)
        print(f"Saved {interpolation_file}")
        print(
            f"  interpolation observed MSE = {metrics['mse_observed']['mean']:.6f} "
            f"± {metrics['mse_observed']['std']:.6f}"
        )
        print(
            f"  interpolation missing  MSE = {metrics['mse_missing']['mean']:.6f} "
            f"± {metrics['mse_missing']['std']:.6f}"
        )
        write_summary_csv()
        del x_interp_anchor

    if need_recovered:
        z_recovered = recover_anchor(
            ae, z_init_norm, x_obs, mask, latent_mean, latent_std
        )
        x_recovered_anchor = decode_normalized_latent(
            ae, z_recovered, latent_mean, latent_std
        )
        metrics = compute_metrics(x_recovered_anchor, x_full, mask)
        result = {
            "dataset": dataset,
            "scenario": scenario,
            "anchor_type": "recovered",
            "num_samples": NUM_SAMPLES,
            "seed": SEED,
            "inference_seed": INFERENCE_SEED,
            "client_steps": CLIENT_STEPS,
            "client_lr": CLIENT_LR,
            "client_prior_weight": CLIENT_PRIOR_WEIGHT,
            "actual_missing_ratio": float((mask == 0).float().mean().item()),
            **metrics,
        }
        save_json_atomic(result, recovered_file)
        print(f"Saved {recovered_file}")
        print(
            f"  recovered observed MSE = {metrics['mse_observed']['mean']:.6f} "
            f"± {metrics['mse_observed']['std']:.6f}"
        )
        print(
            f"  recovered missing  MSE = {metrics['mse_missing']['mean']:.6f} "
            f"± {metrics['mse_missing']['std']:.6f}"
        )
        write_summary_csv()
        del z_recovered, x_recovered_anchor

    del ae, latent_mean, latent_std, z_init_norm, x_full, x_obs, mask
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# MAIN
# =========================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    failures = []

    for dataset, cfg in DATASETS.items():
        for scenario in SCENARIOS:
            try:
                run_case(dataset, scenario, cfg)
            except Exception as e:
                failures.append((dataset, scenario, repr(e)))
                print(f"FAILED: {dataset} / {scenario}: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_summary_csv()

    if failures:
        print("\nFailures:")
        for dataset, scenario, error in failures:
            print(f"  {dataset} / {scenario}: {error}")
        print("\nRe-run the same command after fixing the issue; completed JSON files will be skipped.")
    else:
        print("\nAll anchor checks completed successfully.")


if __name__ == "__main__":
    main()
