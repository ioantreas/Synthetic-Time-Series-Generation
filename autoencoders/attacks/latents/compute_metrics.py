import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from statsmodels.tsa.stattools import acf


# ============================================================
# Dataset
# ============================================================

class LatentDataset(Dataset):
    def __init__(self, latents, sequences):
        self.z = torch.tensor(latents, dtype=torch.float32)
        self.x = torch.tensor(sequences, dtype=torch.float32)

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.z[idx], self.x[idx]


# ============================================================
# Weak attacker
# ============================================================

class Attacker(nn.Module):
    def __init__(self, channels, target_len):
        super().__init__()
        self.target_len = target_len

        self.net = nn.Sequential(
            nn.Conv1d(channels, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, channels, 3, padding=1)
        )

    def forward(self, z):
        z = z.permute(0, 2, 1)
        z = nn.functional.interpolate(z, size=self.target_len, mode="linear", align_corners=False)
        out = self.net(z)
        return out.permute(0, 2, 1)


# ============================================================
# Strong attacker
# ============================================================

class DWBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.pw(self.dw(x))))


class AttackerDecoder(nn.Module):
    def __init__(self, channels, target_len):
        super().__init__()
        self.target_len = target_len

        self.up1 = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.block1 = DWBlock(channels)

        self.up2 = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.block2 = DWBlock(channels)

        self.up3 = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.block3 = DWBlock(channels)

    def forward(self, z):
        z = z.permute(0, 2, 1)

        h = self.block1(self.up1(z))
        h = self.block2(self.up2(h))
        h = self.block3(self.up3(h))

        h = nn.functional.interpolate(h, size=self.target_len, mode="linear", align_corners=False)
        return h.permute(0, 2, 1)


# ============================================================
# Metrics
# ============================================================

def rmse(x, y):
    return np.sqrt(((x - y) ** 2).mean())


def mape_safe(x, y):
    mask = np.abs(x) > 1e-3
    return (np.abs((x[mask] - y[mask]) / x[mask])).mean() * 100

def acf_diff(real, pred, nlags=40):
    """
    real, pred: (N, T, C)
    """

    errs = []
    C = real.shape[2]

    for c in range(C):

        # flatten across batch → more stable
        rseries = real[:, :, c].reshape(-1)
        pseries = pred[:, :, c].reshape(-1)

        # skip degenerate signals
        if np.std(rseries) < 1e-8 or np.std(pseries) < 1e-8:
            continue

        r = acf(rseries, nlags=nlags, fft=True)
        p = acf(pseries, nlags=nlags, fft=True)

        errs.append(np.mean(np.abs(r - p)))

    if len(errs) == 0:
        return 0.0

    return np.mean(errs)

# ============================================================
# Main
# ============================================================

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    latents = np.load(args.latents)
    seq = np.load(args.seq)

    idx = np.random.choice(len(latents), args.num_samples, replace=False)

    latents = latents[idx]
    seq = seq[idx]

    dataset = LatentDataset(latents, seq)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    channels = seq.shape[2]
    target_len = seq.shape[1]

    # -----------------------------
    # Load model
    # -----------------------------
    if args.model_type == "weak":
        model = Attacker(channels, target_len)
    else:
        model = AttackerDecoder(channels, target_len)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device).eval()

    print("Running predictions...")

    preds = []
    reals = []

    with torch.no_grad():
        for z, x in loader:
            z = z.to(device)

            pred = model(z).cpu().numpy()
            preds.append(pred)
            reals.append(x.numpy())

    preds = np.concatenate(preds, axis=0)
    reals = np.concatenate(reals, axis=0)

    # -----------------------------
    # Metrics
    # -----------------------------
    print("\n===== ATTACK METRICS =====")

    mse = ((preds - reals) ** 2).mean()
    rmse_val = rmse(reals, preds)
    mape_val = mape_safe(reals, preds)
    acf_val = acf_diff(reals, preds)

    print("MSE:", mse)
    print("RMSE:", rmse_val)
    print("MAPE:", mape_val)
    print("ACF diff:", acf_val)

    # -----------------------------
    # Save metrics
    # -----------------------------
    model_path = Path(args.model_path)
    out_dir = model_path.parent

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse_val),
        "mape": float(mape_val),
        "acf_diff": float(acf_val),
        "num_samples": int(len(preds))
    }

    with open(out_dir / "metrics.json", "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    print(f"\nSaved metrics to {out_dir / 'metrics.json'}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--latents", type=str, required=True)
    parser.add_argument("--seq", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--model_type", choices=["weak", "strong"], required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=2000)

    args = parser.parse_args()

    main(args)