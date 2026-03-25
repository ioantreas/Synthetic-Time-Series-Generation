import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------------------------
# Import your model (robust)
# -------------------------------------------------
import sys
from pathlib import Path as P

for p in P(__file__).resolve().parents:
    if (p / "minimal_autoencoder.py").exists():
        sys.path.append(str(p))
        break

from minimal_autoencoder import SiloTimeOnlyAE


# -------------------------------------------------
# Decode latents → synthetic data
# -------------------------------------------------

def decode_latents(model, latents, device, batch_size=64):

    dataset = TensorDataset(torch.from_numpy(latents).float())
    loader = DataLoader(dataset, batch_size=batch_size)

    outputs = []

    model.eval()

    with torch.no_grad():
        for (z,) in loader:
            z = z.to(device)
            z = z.permute(0, 2, 1)

            xhat = model.decode(z)
            xhat = xhat.permute(0, 2, 1).cpu().numpy()

            outputs.append(xhat)

    return np.concatenate(outputs)


# -------------------------------------------------
# Flatten
# -------------------------------------------------

def flatten(x):
    return x.reshape(x.shape[0], -1)


# -------------------------------------------------
# Memory-safe NN distances
# -------------------------------------------------

def compute_nn_distances(real, synth, batch_size=128):

    real_f = flatten(real)
    synth_f = flatten(synth)

    distances = []

    for i in range(0, len(real_f), batch_size):

        batch = real_f[i:i+batch_size]

        dists = np.sqrt(
            ((batch[:, None, :] - synth_f[None, :, :]) ** 2).sum(axis=2)
        )

        nn = dists.min(axis=1)
        distances.append(nn)

    return np.concatenate(distances)


# -------------------------------------------------
# ROC AUC
# -------------------------------------------------

def compute_auc(labels, scores):
    """
    labels: 1 = train, 0 = test
    scores: higher → more likely train
    """

    # sort by score descending
    order = np.argsort(-scores)
    labels = labels[order]

    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)

    tp_rate = tp / tp[-1]
    fp_rate = fp / fp[-1]

    auc = np.trapz(tp_rate, fp_rate)

    return auc


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_real", required=True)
    parser.add_argument("--test_real", required=True)
    parser.add_argument("--generated_latents", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--latent_steps", type=int, default=16)

    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--output", required=True)
    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")

    train = np.load(args.train_real)
    test = np.load(args.test_real)
    latents = np.load(args.generated_latents)

    print("Train:", train.shape)
    print("Test:", test.shape)
    print("Latents:", latents.shape)

    # -------------------------------------------------
    # Denormalize latents
    # -------------------------------------------------

    if args.normalize:
        base = Path(args.train_real).parent

        mean = np.load(base / "latents/normalized/latent_mean.npy")
        std  = np.load(base / "latents/normalized/latent_std.npy")

        latents = latents * std + mean
        print("Latents denormalized")

    # -------------------------------------------------
    # Load model
    # -------------------------------------------------

    N, L, C = train.shape

    model = SiloTimeOnlyAE(
        channels=C,
        seq_len=args.seq_len,
        latent_steps=args.latent_steps
    )

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # -------------------------------------------------
    # Decode synthetic
    # -------------------------------------------------

    print("Decoding latents...")
    synth = decode_latents(model, latents, device, args.batch_size)

    print("Synthetic:", synth.shape)

    # -------------------------------------------------
    # Sample equal number of train/test
    # -------------------------------------------------

    n = min(args.num_samples, len(train), len(test))

    idx_train = np.random.choice(len(train), n, replace=False)
    idx_test  = np.random.choice(len(test), n, replace=False)

    train_sample = train[idx_train]
    test_sample  = test[idx_test]

    # -------------------------------------------------
    # Compute distances
    # -------------------------------------------------

    print("Computing distances...")

    d_train = compute_nn_distances(train_sample, synth)
    d_test  = compute_nn_distances(test_sample, synth)

    # -------------------------------------------------
    # Membership scores
    # -------------------------------------------------

    # smaller distance = more likely train → invert
    scores = -np.concatenate([d_train, d_test])
    labels = np.concatenate([
        np.ones(len(d_train)),   # train
        np.zeros(len(d_test))    # test
    ])

    # -------------------------------------------------
    # Compute AUC
    # -------------------------------------------------

    auc = compute_auc(labels, scores)

    # -------------------------------------------------
    # Stats
    # -------------------------------------------------

    results = {
        "auc": float(auc),
        "train_mean": float(np.mean(d_train)),
        "test_mean": float(np.mean(d_test)),
        "train_std": float(np.std(d_train)),
        "test_std": float(np.std(d_test)),
    }

    print("\n===== MEMBERSHIP INFERENCE =====")
    for k, v in results.items():
        print(k, ":", v)

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    np.save(outdir / "train_distances.npy", d_train)
    np.save(outdir / "test_distances.npy", d_test)

    # -------------------------------------------------
    # Plot distributions
    # -------------------------------------------------

    plt.figure(figsize=(8,4))

    plt.hist(d_train, bins=50, alpha=0.5, label="train")
    plt.hist(d_test, bins=50, alpha=0.5, label="test")

    plt.legend()
    plt.title("Membership inference distances")

    plt.savefig(outdir / "distance_hist.png")
    plt.close()

    print("\nSaved to:", outdir)


if __name__ == "__main__":
    main()