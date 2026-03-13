#!/usr/bin/env python3
"""
Unconditional SSSD-S4 diffusion training (WaveStitch-style), adapted from training_SSSD.py.

What this does:
- Trains a pure unconditional DDPM denoiser on sliding windows from the Preprocessor dataframe
- Uses ONLY the unconditional backbone: SSSDS4Imputer (args.backbone = "s4")
- No masks, no conditioning, no imputation loss trickery
- Predicts epsilon (noise) everywhere with MSE

Expected shapes:
- batch: (B, T, C)
- model internally permutes to (B, C, T) and returns (B, C, T)
- we permute output back to (B, T, C) before loss

Run example:
python train_sssd_unconditional.py -d MetroTraffic -backbone s4 -window_size 168 -stride 1 -batch_size 256 -epochs 200 -timesteps 200

Notes:
- If your dataset is already pre-windowed in .npy (N,T,C), you can skip Preprocessor and feed it directly.
"""

import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data_utils import Preprocessor
from training_utils import MyDataset, fetchModel, fetchDiffusionConfig


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sliding_windows(values_2d: np.ndarray, window_size: int, stride: int) -> torch.Tensor:
    """
    values_2d: (N_time, C)
    returns: (N_windows, window_size, C) as torch.float32
    """
    x = torch.from_numpy(values_2d).float()  # (N_time, C)
    # unfold along time dimension (dim=0)
    windows = x.unfold(0, window_size, stride)  # (N_windows, C, window_size)
    windows = windows.transpose(1, 2).contiguous()  # (N_windows, window_size, C)
    return windows


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-dataset", "-d", type=str,
        help="MetroTraffic, BeijingAirQuality, AustraliaTourism, RossmanSales, PanamaEnergy",
        required=True
    )
    parser.add_argument(
        "-backbone", type=str,
        help="Must be 's4' for unconditional SSSD-S4 in this script.",
        default="s4"
    )

    # Diffusion schedule
    parser.add_argument("-beta_0", type=float, default=0.0001, help="initial variance schedule")
    parser.add_argument("-beta_T", type=float, default=0.02, help="last variance schedule")
    parser.add_argument("-timesteps", "-T", type=int, default=200, help="training/inference timesteps")

    # Training
    parser.add_argument("-lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("-batch_size", type=int, default=256, help="batch size")
    parser.add_argument("-epochs", type=int, default=200, help="training epochs")
    parser.add_argument("-save_every", type=int, default=0, help="save checkpoint every N epochs (0 disables)")
    parser.add_argument("-seed", type=int, default=42)

    # Windowing
    parser.add_argument("-window_size", type=int, default=168, help="training window length T")
    parser.add_argument("-stride", type=int, default=1, help="stride length for sliding windows")

    # SSSD-S4 / model hparams (match your WaveStitch args)
    parser.add_argument("-num_res_layers", type=int, default=8, help="number of residual layers")
    parser.add_argument("-res_channels", type=int, default=128, help="residual channels (capacity)")
    parser.add_argument("-skip_channels", type=int, default=128, help="skip channels")
    parser.add_argument("-diff_step_embed_in", type=int, default=64)
    parser.add_argument("-diff_step_embed_mid", type=int, default=128)
    parser.add_argument("-diff_step_embed_out", type=int, default=128)
    parser.add_argument("-s4_lmax", type=int, default=256)
    parser.add_argument("-s4_dstate", type=int, default=64)
    parser.add_argument("-s4_dropout", type=float, default=0.0)
    parser.add_argument("-s4_bidirectional", type=bool, default=True)
    parser.add_argument("-s4_layernorm", type=bool, default=True)

    # Preprocessor option
    parser.add_argument("-propCycEnc", type=bool, default=False)

    # Saving
    parser.add_argument("-save_dir", type=str, default="saved_models_unconditional")

    args = parser.parse_args()

    if args.backbone.lower() != "s4":
        raise ValueError(
            f"This script is for unconditional training and expects -backbone s4. Got: {args.backbone}"
        )

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    train_windows = torch.from_numpy(
        np.load(args.dataset)
    ).float()

    print(f"[info] loaded train_windows shape = {train_windows.shape}")  # (N, T, C)

    in_dim = train_windows.shape[-1]
    out_dim = in_dim
    print(f"[info] channels C = {in_dim}")

    # Dataset / loader
    training_dataset = MyDataset(train_windows)
    dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Model + diffusion config
    model = fetchModel(in_dim, out_dim, args).to(device)
    diffusion_config = fetchDiffusionConfig(args)

    # Pre-move alpha_bars once (we'll index on device)
    alpha_bars = diffusion_config["alpha_bars"].to(device)  # shape (T,)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Output path
    save_path = os.path.join(args.save_dir, args.dataset)
    os.makedirs(save_path, exist_ok=True)
    final_ckpt = os.path.join(save_path, "model_sssds4_unconditional.pth")

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        n_batches = 0

        for batch in dataloader:
            # batch: (B, T, C)
            batch = batch.to(device)

            B = batch.shape[0]
            # Sample random timesteps per sample
            t = torch.randint(low=0, high=diffusion_config["T"], size=(B,), device=device)  # (B,)
            # Noise
            eps = torch.randn_like(batch)

            # Forward diffusion: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps
            a_bar = alpha_bars[t].view(B, 1, 1)                  # (B,1,1)
            x_t = torch.sqrt(a_bar) * batch + torch.sqrt(1.0 - a_bar) * eps

            # Model expects timesteps shaped (B,1) in this codebase
            t_in = t.view(B, 1)

            # Predict epsilon
            # model returns (B, C, T) for SSSDS4Imputer -> permute to (B, T, C)
            eps_hat = model(x_t, t_in).permute(0, 2, 1).contiguous()

            loss = criterion(eps_hat, eps)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            n_batches += 1

        avg_loss = running / max(1, n_batches)
        print(f"epoch {epoch:04d}/{args.epochs}  loss={avg_loss:.6f}")

        if args.save_every and (epoch % args.save_every == 0):
            ckpt = os.path.join(save_path, f"model_epoch_{epoch:04d}.pth")
            torch.save(model.state_dict(), ckpt)

    torch.save(model.state_dict(), final_ckpt)
    print(f"[done] saved final model to: {final_ckpt}")


if __name__ == "__main__":
    train()