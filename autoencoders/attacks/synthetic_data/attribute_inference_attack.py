import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, TensorDataset

import sys
from pathlib import Path as P

# Robust import
for p in P(__file__).resolve().parents:
    if (p / "minimal_autoencoder.py").exists():
        sys.path.append(str(p))
        break

from minimal_autoencoder import SiloTimeOnlyAE


# ============================================================
# Decode latents
# ============================================================

def decode_latents(model, latents, device, batch_size=64):

    dataset = TensorDataset(torch.from_numpy(latents).float())
    loader = DataLoader(dataset, batch_size=batch_size)

    outputs = []

    model.eval()

    with torch.no_grad():
        for (z,) in loader:

            z = z.to(device)
            z = z.permute(0,2,1)

            xhat = model.decode(z)
            xhat = xhat.permute(0,2,1).cpu().numpy()

            outputs.append(xhat)

    return np.concatenate(outputs)


# ============================================================
# Dataset
# ============================================================

class ChannelDataset(Dataset):
    def __init__(self, x, known_idx, secret_idx):
        self.x_known = torch.from_numpy(x[:, :, known_idx]).float()
        self.x_secret = torch.from_numpy(x[:, :, secret_idx]).float()

    def __len__(self):
        return len(self.x_known)

    def __getitem__(self, i):
        return self.x_known[i], self.x_secret[i]


# ============================================================
# Model
# ============================================================

class InferenceNet(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(c_in, 128, 5, padding=2),
            nn.ReLU(),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.ReLU(),

            nn.Conv1d(128, 64, 5, padding=2),
            nn.ReLU(),

            nn.Conv1d(64, c_out, 3, padding=1)
        )

    def forward(self, x):
        x = x.permute(0,2,1)
        y = self.net(x)
        return y.permute(0,2,1)


# ============================================================
# Train / Eval
# ============================================================

def train(model, loader, opt, loss_fn, device):

    model.train()
    total = 0

    for xk, xs in loader:
        xk, xs = xk.to(device), xs.to(device)

        pred = model(xk)
        loss = loss_fn(pred, xs)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total += loss.item()

    return total / len(loader)


def evaluate(model, loader, device):

    model.eval()

    preds = []
    tgts = []

    with torch.no_grad():
        for xk, xs in loader:
            xk = xk.to(device)
            pred = model(xk).cpu().numpy()

            preds.append(pred)
            tgts.append(xs.numpy())

    pred = np.concatenate(preds)
    tgt = np.concatenate(tgts)

    mse = ((pred - tgt) ** 2).mean()
    mae = np.abs(pred - tgt).mean()

    return mse, mae


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_real", required=True)
    parser.add_argument("--test_real", required=True)

    parser.add_argument("--generated_latents", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--latent_steps", type=int, default=16)

    parser.add_argument("--secret_channels", type=int, nargs="+", required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--output", required=True)
    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")

    train_real = np.load(args.train_real)
    test_real = np.load(args.test_real)
    latents = np.load(args.generated_latents)

    N, L, C = test_real.shape

    # --------------------------------
    # Denormalize latents
    # --------------------------------

    if args.normalize:
        base = Path(args.train_real).parent

        mean = np.load(base / "latents/normalized/latent_mean.npy")
        std  = np.load(base / "latents/normalized/latent_std.npy")

        latents = latents * std + mean

        print("Latents denormalized")

    # --------------------------------
    # Load decoder
    # --------------------------------

    model = SiloTimeOnlyAE(
        channels=C,
        seq_len=args.seq_len,
        latent_steps=args.latent_steps
    )

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # --------------------------------
    # Decode synthetic
    # --------------------------------

    print("Decoding latents...")
    synth = decode_latents(model, latents, device, args.batch_size)

    print("Synthetic:", synth.shape)

    # --------------------------------
    # Match training sizes
    # --------------------------------

    n_synth = len(synth)
    n_real = len(train_real)

    print(f"\nMatching dataset sizes: synth={n_synth}, real={n_real}")

    if n_real > n_synth:
        idx = np.random.choice(n_real, n_synth, replace=False)
        train_real = train_real[idx]
        print(f"Subsampled real train → {train_real.shape}")

    elif n_real < n_synth:
        idx = np.random.choice(n_synth, n_real, replace=False)
        synth = synth[idx]
        print(f"Subsampled synthetic → {synth.shape}")

    # --------------------------------
    # Setup channels
    # --------------------------------

    secret_idx = args.secret_channels
    known_idx = [i for i in range(C) if i not in secret_idx]

    # --------------------------------
    # Datasets
    # --------------------------------

    train_synth_ds = ChannelDataset(synth, known_idx, secret_idx)
    train_real_ds = ChannelDataset(train_real, known_idx, secret_idx)
    test_ds = ChannelDataset(test_real, known_idx, secret_idx)

    train_synth_loader = DataLoader(train_synth_ds, batch_size=args.batch_size, shuffle=True)
    train_real_loader = DataLoader(train_real_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # --------------------------------
    # Train synthetic attacker
    # --------------------------------

    print("\nTraining attacker on synthetic data")

    model_synth = InferenceNet(len(known_idx), len(secret_idx)).to(device)
    opt = torch.optim.Adam(model_synth.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for ep in range(args.epochs):
        loss = train(model_synth, train_synth_loader, opt, loss_fn, device)
        print(f"[Synth] epoch {ep} loss {loss:.6f}")

    mse_synth, mae_synth = evaluate(model_synth, test_loader, device)

    # --------------------------------
    # Train real baseline
    # --------------------------------

    print("\nTraining baseline on real data")

    model_real = InferenceNet(len(known_idx), len(secret_idx)).to(device)
    opt = torch.optim.Adam(model_real.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        loss = train(model_real, train_real_loader, opt, loss_fn, device)
        print(f"[Real] epoch {ep} loss {loss:.6f}")

    mse_real, mae_real = evaluate(model_real, test_loader, device)

    # --------------------------------
    # Results
    # --------------------------------

    results = {
        "synthetic_attack": {
            "mse": float(mse_synth),
            "mae": float(mae_synth)
        },
        "real_baseline": {
            "mse": float(mse_real),
            "mae": float(mae_real)
        },
        "ratios": {
            "mse_ratio": float(mse_synth / mse_real),
            "mae_ratio": float(mae_synth / mae_real)
        }
    }

    print("\n===== ATTRIBUTE INFERENCE =====")
    print(results)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "metrics.json", "w") as f:
        json.dump({
            "secret": secret_idx,
            **results
        }, f, indent=2)


if __name__ == "__main__":
    main()