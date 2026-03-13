import argparse
import json

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import acf
from torch.utils.data import DataLoader, TensorDataset

from minimal_autoencoder import SiloTimeOnlyAE
from minimal_global_autoencoder import TemporalAE_Global


# --------------------------------
# Encode -> Decode (batched)
# --------------------------------

def encode_decode(model, data, device, batch_size=64):

    dataset = TensorDataset(torch.from_numpy(data).float())
    loader = DataLoader(dataset, batch_size=batch_size)

    xs = []
    xhats = []
    latents_all = []

    model.eval()

    with torch.no_grad():

        for (x,) in loader:

            x = x.to(device)

            # (N,L,C) → (N,C,L)
            x_in = x.permute(0,2,1)

            z = None
            xhat = None

            if hasattr(model, "encode") and hasattr(model, "decode"):

                z = model.encode(x_in)
                xhat = model.decode(z)

            else:

                out = model(x_in)

                if isinstance(out, tuple):

                    if len(out) >= 2:
                        xhat, z = out[0], out[1]
                    elif len(out) == 1:
                        xhat = out[0]

                else:
                    xhat = out

                if z is None and hasattr(model,"encode"):
                    z = model.encode(x_in)

                if xhat is None and z is not None and hasattr(model,"decode"):
                    xhat = model.decode(z)

            if xhat is None:

                raise RuntimeError(
                    "Could not obtain reconstruction."
                )

            x_np = x_in.permute(0,2,1).cpu().numpy()
            xhat_np = xhat.permute(0,2,1).cpu().numpy()

            xs.append(x_np)
            xhats.append(xhat_np)

            if z is not None:
                latents_all.append(z.detach().cpu().numpy())

    xs = np.concatenate(xs)
    xhats = np.concatenate(xhats)

    latents = None

    if len(latents_all) > 0:
        latents = np.concatenate(latents_all)

    return xs, xhats, latents


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

        rseries = real[:,:,c].reshape(-1)
        sseries = synth[:,:,c].reshape(-1)

        if np.std(rseries) < 1e-8 or np.std(sseries) < 1e-8:
            continue

        r = acf(rseries, nlags=nlags)
        s = acf(sseries, nlags=nlags)

        errs.append(np.mean(np.abs(r-s)))

    if len(errs) == 0:
        return 0.0

    return np.mean(errs)


def corr_matrix(batch):

    X = batch.reshape(-1, batch.shape[-1])

    std = np.std(X, axis=0)

    valid = std > 1e-8

    if np.sum(valid) < 2:
        return np.zeros((1,1))

    X = X[:,valid]

    return np.corrcoef(X, rowvar=False)


def corr_difference(real, synth):

    Cr = corr_matrix(real)
    Cs = corr_matrix(synth)

    m = min(Cr.shape[0], Cs.shape[0])

    Cr = Cr[:m,:m]
    Cs = Cs[:m,:m]

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
# Debug statistics
# --------------------------------

def debug_channel_stats(real, synth):

    print("\n===== CHANNEL STATISTICS =====")

    real_std = np.std(real, axis=(0,1))
    synth_std = np.std(synth, axis=(0,1))

    for i in range(real.shape[-1]):

        print(
            f"channel {i:02d} | "
            f"real std = {real_std[i]:.6f} | "
            f"recon std = {synth_std[i]:.6f}"
        )

    flat_real = np.where(real_std < 1e-8)[0]
    flat_synth = np.where(synth_std < 1e-8)[0]

    print("\nFlat channels in REAL:", flat_real)
    print("Flat channels in RECON:", flat_synth)

    print("\n===== RANGE CHECK =====")

    for i in range(real.shape[-1]):

        rmin = real[:,:,i].min()
        rmax = real[:,:,i].max()

        smin = synth[:,:,i].min()
        smax = synth[:,:,i].max()

        print(
            f"channel {i:02d} | "
            f"real [{rmin:.4f},{rmax:.4f}] | "
            f"recon [{smin:.4f},{smax:.4f}]"
        )


# --------------------------------
# Plots
# --------------------------------

def plot_sequences(real, synth, outdir, channel=0):

    if channel >= real.shape[-1]:
        print(f"Skipping sequence plot for channel {channel}")
        return

    plt.figure(figsize=(10,4))

    plt.plot(real[0,:,channel], label="real")
    plt.plot(synth[0,:,channel], label="reconstructed")

    plt.legend()
    plt.title(f"Sequence comparison (channel {channel})")

    plt.savefig(outdir / f"sequence_{channel}.png")

    plt.close()


def plot_fft(real, synth, outdir):

    rf = np.mean(np.abs(np.fft.rfft(real, axis=1))**2, axis=(0,2))
    sf = np.mean(np.abs(np.fft.rfft(synth, axis=1))**2, axis=(0,2))

    plt.figure(figsize=(8,4))

    plt.plot(rf, label="real")
    plt.plot(sf, label="reconstructed")

    plt.legend()
    plt.title("Average power spectrum")

    plt.savefig(outdir / "fft.png")

    plt.close()


def plot_histogram(real, synth, outdir):

    plt.figure(figsize=(8,4))

    plt.hist(real.flatten(), bins=100, density=True, alpha=0.5, label="real")
    plt.hist(synth.flatten(), bins=100, density=True, alpha=0.5, label="reconstructed")

    plt.legend()
    plt.title("Value distribution")

    plt.savefig(outdir / "histogram.png")

    plt.close()


def plot_corr(real, synth, outdir):

    Cr = corr_matrix(real)
    Cs = corr_matrix(synth)

    for M,name in [(Cr,"real"),(Cs,"reconstructed"),(Cs-Cr,"diff")]:

        plt.figure(figsize=(6,5))

        plt.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()

        plt.title(name)

        plt.savefig(outdir / f"corr_{name}.png")

        plt.close()


# --------------------------------
# Main
# --------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--model_type",
        choices=["temporal","global"],
        required=True
    )

    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = np.load(args.dataset).astype(np.float32)

    print("Dataset:", data.shape)

    N,L,C = data.shape

    # ----------------------
    # Model
    # ----------------------

    if args.model_type == "temporal":

        model = SiloTimeOnlyAE(
            channels=C,
            seq_len=L,
            latent_steps=args.latent_dim
        )

    else:

        model = TemporalAE_Global(
            C,
            L,
            latent_dim=args.latent_dim
        )

    model.load_state_dict(torch.load(args.model, map_location=device))

    model.to(device)

    # ----------------------
    # Encode → Decode
    # ----------------------

    real, synth, latents = encode_decode(
        model,
        data[:args.num_samples],
        device,
        batch_size=args.batch_size
    )

    if latents is not None:
        print("Latents shape:", latents.shape)

    # ----------------------
    # Debug channels
    # ----------------------

    debug_channel_stats(real, synth)

    # ----------------------
    # Metrics
    # ----------------------

    total_mse = mse(real, synth)
    per_channel = mse_per_channel(real, synth)

    acf_err = acf_error(real, synth)
    corr_err = corr_difference(real, synth)
    fft_err = fft_difference(real, synth)
    hist_err = histogram_distance(real, synth)

    metrics = {

        "reconstruction_mse": float(total_mse),
        "acf_difference": float(acf_err),
        "correlation_difference": float(corr_err),
        "fft_difference": float(fft_err),
        "histogram_distance": float(hist_err),
        "mse_per_channel": per_channel.tolist(),
        "num_samples": int(real.shape[0]),
    }

    if latents is not None:
        metrics["latents_shape"] = list(latents.shape)

    # ----------------------
    # Save metrics
    # ----------------------

    outdir = Path(args.output)
    outdir.mkdir(exist_ok=True, parents=True)

    with open(outdir / "metrics.json","w") as f:
        json.dump(metrics,f,indent=2)

    print("\n===== METRICS =====")

    for k,v in metrics.items():
        if k != "mse_per_channel":
            print(k,":",v)

    # ----------------------
    # Plots
    # ----------------------

    plot_sequences(real, synth, outdir, 0)
    plot_sequences(real, synth, outdir, 10)
    plot_sequences(real, synth, outdir, 20)

    plot_fft(real, synth, outdir)
    plot_histogram(real, synth, outdir)
    plot_corr(real, synth, outdir)

    print("\nSaved plots to",outdir)


if __name__ == "__main__":

    main()