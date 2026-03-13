import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import acf
from torch.utils.data import DataLoader, TensorDataset

from minimal_autoencoder import SiloTimeOnlyAE


# --------------------------------
# Decode latent sequences
# --------------------------------

def decode_latents(model, latents, device, batch_size=64):

    dataset = TensorDataset(torch.from_numpy(latents))
    loader = DataLoader(dataset, batch_size=batch_size)

    outputs = []

    model.eval()

    with torch.no_grad():

        for (z,) in loader:

            z = z.to(device)

            z = z.permute(0,2,1)  # (N,T,C) → (N,C,T)

            xhat = model.decode(z)

            xhat = xhat.permute(0,2,1).cpu().numpy()

            outputs.append(xhat)

    return np.concatenate(outputs)


# --------------------------------
# Metrics
# --------------------------------

def mse(real, synth):

    return ((real - synth) ** 2).mean()


def mse_per_channel(real, synth):

    return ((real - synth) ** 2).mean(axis=(0,1))


def acf_error(real, synth, nlags=40):

    errs = []

    C = real.shape[-1]

    for c in range(C):

        r = acf(real[:,:,c].reshape(-1), nlags=nlags)
        s = acf(synth[:,:,c].reshape(-1), nlags=nlags)

        errs.append(np.mean(np.abs(r - s)))

    return np.mean(errs)


def corr_matrix(batch):

    X = batch.reshape(-1, batch.shape[-1])

    return np.corrcoef(X, rowvar=False)


def corr_difference(real, synth):

    Cr = corr_matrix(real)
    Cs = corr_matrix(synth)

    return np.mean(np.abs(Cr-Cs))


def fft_difference(real, synth):

    rf = np.mean(np.abs(np.fft.rfft(real, axis=1))**2, axis=(0,2))
    sf = np.mean(np.abs(np.fft.rfft(synth, axis=1))**2, axis=(0,2))

    return np.mean(np.abs(rf-sf))


def histogram_distance(real, synth, bins=100):

    r = real.flatten()
    s = synth.flatten()

    hr,_ = np.histogram(r, bins=bins, density=True)
    hs,_ = np.histogram(s, bins=bins, density=True)

    return np.mean(np.abs(hr-hs))


# --------------------------------
# Plots
# --------------------------------

def plot_sequences(real, synth, outdir, channel=0):

    plt.figure(figsize=(10,4))

    plt.plot(real[0,:,channel], label="real")
    plt.plot(synth[0,:,channel], label="synthetic")

    plt.legend()

    plt.title(f"Sequence comparison (channel {channel})")

    plt.savefig(outdir / f"sequence_{channel}.png")

    plt.close()


def plot_fft(real, synth, outdir):

    rf = np.mean(np.abs(np.fft.rfft(real, axis=1))**2, axis=(0,2))
    sf = np.mean(np.abs(np.fft.rfft(synth, axis=1))**2, axis=(0,2))

    plt.figure(figsize=(8,4))

    plt.plot(rf, label="real")
    plt.plot(sf, label="synthetic")

    plt.legend()

    plt.title("Average power spectrum")

    plt.savefig(outdir / "fft.png")

    plt.close()


def plot_histogram(real, synth, outdir):

    plt.figure(figsize=(8,4))

    plt.hist(real.flatten(), bins=100, density=True, alpha=0.5, label="real")
    plt.hist(synth.flatten(), bins=100, density=True, alpha=0.5, label="synthetic")

    plt.legend()

    plt.title("Value distribution")

    plt.savefig(outdir / "histogram.png")

    plt.close()


def plot_corr(real, synth, outdir):

    Cr = corr_matrix(real)
    Cs = corr_matrix(synth)

    vmax = 1

    for M,name in [(Cr,"real"),(Cs,"synthetic"),(Cs-Cr,"diff")]:

        plt.figure(figsize=(6,5))

        plt.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax)

        plt.colorbar()

        plt.title(name)

        plt.savefig(outdir / f"corr_{name}.png")

        plt.close()


# --------------------------------
# Main
# --------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--real_data", required=True)
    parser.add_argument("--generated_latents", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--latent_steps", type=int, default=16)

    parser.add_argument("--num_samples", type=int, default=2000)

    parser.add_argument("--output", required=True)

    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    real = np.load(args.real_data)

    print("Real dataset:", real.shape)

    N,L,C = real.shape

    latents = np.load(args.generated_latents)

    print("Generated latents:", latents.shape)

    if args.normalize:

        path = Path(args.real_data)
        last_folder = Path(path.parent)
        mean = np.load(last_folder / "latents/normalized/latent_mean.npy")
        std  = np.load(last_folder / "latents/normalized/latent_std.npy")

        latents = latents * std + mean

    model = SiloTimeOnlyAE(
        channels=C,
        seq_len=args.seq_len,
        latent_steps=args.latent_steps
    )

    model.load_state_dict(torch.load(args.model, map_location=device))

    model.to(device)

    # --------------------------------
    # Random sampling
    # --------------------------------

    idx_lat = np.random.choice(len(latents), args.num_samples, replace=False)
    idx_real = np.random.choice(len(real), args.num_samples, replace=False)

    synth = decode_latents(model, latents[idx_lat], device)

    real = real[idx_real]

    # --------------------------------
    # Metrics
    # --------------------------------

    total_mse = mse(real, synth)
    per_channel = mse_per_channel(real, synth)

    acf_err = acf_error(real, synth)
    corr_err = corr_difference(real, synth)
    fft_err = fft_difference(real, synth)
    hist_err = histogram_distance(real, synth)

    metrics = {
        "synthetic_vs_real_mse": float(total_mse),
        "acf_difference": float(acf_err),
        "correlation_difference": float(corr_err),
        "fft_difference": float(fft_err),
        "histogram_distance": float(hist_err),
        "mse_per_channel": per_channel.tolist(),
        "num_samples": int(real.shape[0]),
    }

    # --------------------------------
    # Save metrics
    # --------------------------------

    outdir = Path(args.output)
    outdir.mkdir(exist_ok=True)

    # JSON (best for later analysis)
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Human readable
    with open(outdir / "metrics.txt", "w") as f:

        f.write("===== METRICS =====\n\n")

        f.write(f"Synthetic vs Real MSE: {total_mse}\n")
        f.write(f"ACF difference: {acf_err}\n")
        f.write(f"Correlation difference: {corr_err}\n")
        f.write(f"FFT difference: {fft_err}\n")
        f.write(f"Histogram distance: {hist_err}\n\n")

        f.write("MSE per channel:\n")

        for i, v in enumerate(per_channel):
            f.write(f"{i}: {v}\n")

    print("\n===== METRICS =====")
    for k, v in metrics.items():
        if k != "mse_per_channel":
            print(k, ":", v)

    print("\nMetrics saved to:", outdir)

    # --------------------------------
    # Plots
    # --------------------------------

    outdir = Path(args.output)

    outdir.mkdir(exist_ok=True)

    plot_sequences(real, synth, outdir, 0)
    plot_sequences(real, synth, outdir, 10)
    plot_sequences(real, synth, outdir, 20)

    plot_fft(real, synth, outdir)
    plot_histogram(real, synth, outdir)
    plot_corr(real, synth, outdir)

    print("\nSaved plots to", outdir)


if __name__ == "__main__":

    main()