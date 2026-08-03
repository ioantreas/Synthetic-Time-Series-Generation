import argparse
import numpy as np
import torch
import json
from pathlib import Path

from minimal_autoencoder import SiloTimeOnlyAE


def encode_dataset(model, data, device):

    model.eval()
    latents = []

    with torch.no_grad():

        for x in data:

            x = torch.tensor(x).unsqueeze(0).to(device)  # (1,L,C)
            x = x.permute(0,2,1)  # -> (1,C,L)

            _, z = model(x)       # (1,C,T_latent)

            latents.append(z.cpu().numpy()[0])

    latents = np.stack(latents)

    # (N,C,T) → (N,T,C)
    latents = latents.transpose(0,2,1)

    return latents


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--model", required=True)

    parser.add_argument("--latent_steps", type=int, default=16)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument(
        "--feature_idx",
        type=int,
        nargs="+",
        default=None,
        help="Indices of features to use. Default: all features"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading datasets...")
    train = np.load(args.train_data)
    test = np.load(args.test_data)

    original_C = train.shape[2]

    if args.feature_idx is not None:
        train = train[:, :, args.feature_idx]
        test = test[:, :, args.feature_idx]

    print("Train shape:", train.shape)
    print("Test shape:", test.shape)

    L = train.shape[1]
    C = train.shape[2]

    print("Loading autoencoder...")
    model = SiloTimeOnlyAE(
        channels=C,
        seq_len=L,
        latent_steps=args.latent_steps
    )

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    print("Encoding train dataset...")
    train_latents = encode_dataset(model, train, device)

    print("Encoding test dataset...")
    test_latents = encode_dataset(model, test, device)

    # -------------------------------------------------
    # NORMALIZE USING TRAIN STATISTICS
    # -------------------------------------------------

    print("Computing latent normalization...")

    mean = train_latents.mean(axis=(0,1), keepdims=True)
    std = train_latents.std(axis=(0,1), keepdims=True) + 1e-8

    train_latents = (train_latents - mean) / std
    test_latents = (test_latents - mean) / std

    # -------------------------------------------------

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.feature_idx is None:
        feature_idx_resolved = list(range(original_C))
    else:
        feature_idx_resolved = args.feature_idx

    config = {
        "train_data": args.train_data,
        "test_data": args.test_data,
        "model": args.model,
        "latent_steps": args.latent_steps,
        "feature_idx": feature_idx_resolved,
        "num_features": len(feature_idx_resolved),
        "seq_len": L
    }

    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    np.save(out / "train_latents.npy", train_latents)
    np.save(out / "test_latents.npy", test_latents)

    np.save(out / "latent_mean.npy", mean)
    np.save(out / "latent_std.npy", std)

    print("Saved:")
    print("train:", train_latents.shape)
    print("test:", test_latents.shape)
    print("mean/std saved")


if __name__ == "__main__":
    main()