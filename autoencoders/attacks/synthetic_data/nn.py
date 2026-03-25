import argparse
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path as P

# Robust import
for p in P(__file__).resolve().parents:
    if (p / "minimal_autoencoder.py").exists():
        sys.path.append(str(p))
        break

from minimal_autoencoder import SiloTimeOnlyAE


# --------------------------------
# Decode latent sequences
# --------------------------------

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


# --------------------------------
# Flatten
# --------------------------------

def flatten(x):
    return x.reshape(x.shape[0], -1)


# --------------------------------
# Memory-safe NN computation
# --------------------------------

def compute_nn_distances(real, target, real_bs=128, target_bs=256):

    real_flat = flatten(real)
    target_flat = flatten(target)

    distances = []

    for i in range(0, len(real_flat), real_bs):

        real_batch = real_flat[i:i+real_bs]
        min_dist = np.full(len(real_batch), np.inf)

        for j in range(0, len(target_flat), target_bs):

            target_batch = target_flat[j:j+target_bs]

            dists = np.sqrt(
                ((real_batch[:, None, :] - target_batch[None, :, :]) ** 2).sum(axis=2)
            )

            min_dist = np.minimum(min_dist, dists.min(axis=1))

        distances.append(min_dist)

    return np.concatenate(distances)


# --------------------------------
# Stats helper
# --------------------------------

def stats(x):
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p1": float(np.percentile(x, 1)),
        "p5": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x, 10)),
    }


# --------------------------------
# Main
# --------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--generated_latents", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--latent_steps", type=int, default=16)

    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--output", required=True)
    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")

    train = np.load(args.train_data)
    test = np.load(args.test_data)
    latents = np.load(args.generated_latents)

    print("Train:", train.shape)
    print("Test:", test.shape)
    print("Latents:", latents.shape)

    N, L, C = test.shape

    # --------------------------------
    # Denormalize latents
    # --------------------------------

    if args.normalize:

        path = Path(args.train_data)
        base = Path(path.parent)

        mean = np.load(base / "latents/normalized/latent_mean.npy")
        std  = np.load(base / "latents/normalized/latent_std.npy")

        latents = latents * std + mean

        print("Latents denormalized")

    # --------------------------------
    # Load model
    # --------------------------------

    model = SiloTimeOnlyAE(
        channels=C,
        seq_len=args.seq_len,
        latent_steps=args.latent_steps
    )

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # --------------------------------
    # Sampling
    # --------------------------------

    idx_test = np.random.choice(len(test), args.num_samples, replace=False)
    idx_train = np.random.choice(len(train), args.num_samples, replace=False)
    idx_lat = np.random.choice(len(latents), args.num_samples, replace=False)

    test_sample = test[idx_test]
    train_sample = train[idx_train]
    latents_sample = latents[idx_lat]

    # --------------------------------
    # Decode
    # --------------------------------

    print("\nDecoding latents...")
    synth = decode_latents(model, latents_sample, device, args.batch_size)

    # --------------------------------
    # Compute NN
    # --------------------------------

    print("\nComputing NN (test → synthetic)...")
    nn_test_synth = compute_nn_distances(test_sample, synth)

    print("Computing NN (test → train)...")
    nn_test_train = compute_nn_distances(test_sample, train_sample)

    print("Computing NN (train → synthetic)...")
    nn_train_synth = compute_nn_distances(train_sample, synth)

    # --------------------------------
    # Stats
    # --------------------------------

    stats_test_synth = stats(nn_test_synth)
    stats_test_train = stats(nn_test_train)
    stats_train_synth = stats(nn_train_synth)

    # Ratios
    ratio_test = stats_test_synth["mean"] / stats_test_train["mean"]
    ratio_mem = stats_train_synth["mean"] / stats_test_train["mean"]

    results = {
        "test_to_synth": stats_test_synth,
        "test_to_train": stats_test_train,
        "train_to_synth": stats_train_synth,
        "ratios": {
            "distribution_ratio": float(ratio_test),
            "memorization_ratio": float(ratio_mem)
        },
        "num_samples": int(len(nn_test_synth))
    }

    # --------------------------------
    # Print
    # --------------------------------

    print("\n===== NN PRIVACY RESULTS =====")

    print("\n--- test → synthetic ---")
    for k, v in stats_test_synth.items():
        print(k, ":", v)

    print("\n--- test → train ---")
    for k, v in stats_test_train.items():
        print(k, ":", v)

    print("\n--- train → synthetic ---")
    for k, v in stats_train_synth.items():
        print(k, ":", v)

    print("\n--- ratios ---")
    print("distribution_ratio:", ratio_test)
    print("memorization_ratio:", ratio_mem)

    # --------------------------------
    # Save
    # --------------------------------

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "nn_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to:", outdir)


# --------------------------------

if __name__ == "__main__":
    main()