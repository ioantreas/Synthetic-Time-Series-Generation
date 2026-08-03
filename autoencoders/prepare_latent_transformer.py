import argparse
import json
from pathlib import Path

import numpy as np
import torch

from transformer_autoencoder import TransformerTimeAE


# ============================================================
# Encode Dataset
# ============================================================

def encode_dataset(model, data, device):

    model.eval()

    latents = []

    with torch.no_grad():

        for x in data:

            # (L,C) -> (1,L,C)
            x = torch.tensor(x).unsqueeze(0).to(device)

            # transformer expects (B,C,L)
            x = x.permute(0, 2, 1)

            _, z = model(x)

            # z: (1, latent_steps, d_model)
            latents.append(z.cpu().numpy()[0])

    latents = np.stack(latents)

    return latents


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", required=True)

    parser.add_argument("--test_data", required=True)

    parser.add_argument("--model", required=True)

    parser.add_argument("--latent_steps", type=int, default=16)

    parser.add_argument("--d_model", type=int, default=128)

    parser.add_argument("--nhead", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument(
        "--feature_idx",
        type=int,
        nargs="+",
        default=None,
        help="Indices of features to use"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========================================================
    # Load datasets
    # ========================================================

    print("Loading datasets...")

    train = np.load(args.train_data)

    test = np.load(args.test_data)

    original_C = train.shape[2]

    # --------------------------------------------------------

    if args.feature_idx is not None:

        train = train[:, :, args.feature_idx]

        test = test[:, :, args.feature_idx]

    print("Train shape:", train.shape)

    print("Test shape:", test.shape)

    # ========================================================
    # Model
    # ========================================================

    L = train.shape[1]

    C = train.shape[2]

    print("Loading transformer autoencoder...")

    model = TransformerTimeAE(
        channels=C,
        seq_len=L,
        latent_steps=args.latent_steps,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    )

    model.load_state_dict(
        torch.load(args.model, map_location=device)
    )

    model.to(device)

    # ========================================================
    # Encode datasets
    # ========================================================

    print("Encoding train dataset...")

    train_latents = encode_dataset(
        model,
        train,
        device
    )

    print("Encoding test dataset...")

    test_latents = encode_dataset(
        model,
        test,
        device
    )

    print("Train latent shape:", train_latents.shape)

    print("Test latent shape:", test_latents.shape)

    # ========================================================
    # Normalize latent space
    # ========================================================

    print("Computing latent normalization...")

    # normalize per embedding dimension
    mean = train_latents.mean(
        axis=(0,1),
        keepdims=True
    )

    std = train_latents.std(
        axis=(0,1),
        keepdims=True
    ) + 1e-8

    train_latents = (
                            train_latents - mean
                    ) / std

    test_latents = (
                           test_latents - mean
                   ) / std

    # ========================================================
    # Save
    # ========================================================

    out = Path(args.out_dir)

    out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------

    if args.feature_idx is None:

        feature_idx_resolved = list(range(original_C))

    else:

        feature_idx_resolved = args.feature_idx

    # --------------------------------------------------------

    config = {
        "train_data": args.train_data,
        "test_data": args.test_data,
        "model": args.model,
        "latent_steps": args.latent_steps,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "feature_idx": feature_idx_resolved,
        "num_input_features": len(feature_idx_resolved),
        "latent_dim": args.d_model,
        "seq_len": L
    }

    with open(out / "config.json", "w") as f:

        json.dump(config, f, indent=4)

    # --------------------------------------------------------

    np.save(
        out / "train_latents.npy",
        train_latents
    )

    np.save(
        out / "test_latents.npy",
        test_latents
    )

    np.save(
        out / "latent_mean.npy",
        mean
    )

    np.save(
        out / "latent_std.npy",
        std
    )

    # ========================================================

    print("\nSaved:")

    print("train:", train_latents.shape)

    print("test:", test_latents.shape)

    print("mean/std saved")


# ============================================================

if __name__ == "__main__":

    main()