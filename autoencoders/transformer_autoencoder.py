import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

class WindowDataset(Dataset):

    def __init__(self, path, feature_idx=None):

        x = np.load(path).astype(np.float32)   # (N,L,C)

        if feature_idx is not None:
            x = x[:, :, feature_idx]

        self.x = torch.from_numpy(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=4096):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# ============================================================
# Transformer Autoencoder
# ============================================================

class TransformerTimeAE(nn.Module):

    """
    Input:
        (B,C,L)

    Latent:
        (B,latent_steps,d_model)

    Output:
        (B,C,L)
    """

    def __init__(
            self,
            channels,
            seq_len,
            latent_steps=16,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1
    ):

        super().__init__()

        self.channels = channels
        self.seq_len = seq_len
        self.latent_steps = latent_steps
        self.d_model = d_model

        # ------------------------------------------------
        # Input projection
        # ------------------------------------------------

        self.input_proj = nn.Linear(channels, d_model)

        self.pos_enc = PositionalEncoding(
            d_model,
            max_len=seq_len
        )

        # ------------------------------------------------
        # Encoder
        # ------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ------------------------------------------------
        # Learned latent queries
        # ------------------------------------------------

        self.latent_queries = nn.Parameter(
            torch.randn(latent_steps, d_model)
        )

        self.cross_attn_enc = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True
        )

        # ------------------------------------------------
        # Decoder
        # ------------------------------------------------

        self.cross_attn_dec = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, channels)

    # ====================================================
    # Encode
    # ====================================================

    def encode(self, x):

        # (B,C,L) -> (B,L,C)
        x = x.permute(0, 2, 1)

        h = self.input_proj(x)

        h = self.pos_enc(h)

        h = self.encoder(h)

        B = h.shape[0]

        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)

        z, _ = self.cross_attn_enc(
            queries,
            h,
            h
        )

        return z

    # ====================================================
    # Decode
    # ====================================================

    def decode(self, z):

        B = z.shape[0]

        # sequence queries
        seq_queries = torch.zeros(
            B,
            self.seq_len,
            self.d_model,
            device=z.device
        )

        seq_queries = self.pos_enc(seq_queries)

        h, _ = self.cross_attn_dec(
            seq_queries,
            z,
            z
        )

        h = self.decoder(h)

        xhat = self.output_proj(h)

        # (B,L,C) -> (B,C,L)
        xhat = xhat.permute(0, 2, 1)

        return xhat

    # ====================================================

    def forward(self, x):

        z = self.encode(x)

        xhat = self.decode(z)

        return xhat, z


# ============================================================
# Training
# ============================================================

def train(model, loader, epochs, lr, device):

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    loss_fn = nn.SmoothL1Loss(beta=0.5)

    for ep in range(epochs):

        model.train()

        total = 0.0

        for x in loader:

            x = x.to(device)

            x = x.permute(0, 2, 1)

            xhat, _ = model(x)

            loss = loss_fn(xhat, x)

            optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total += loss.item()

        print(
            f"epoch {ep:03d} | "
            f"loss {total/len(loader):.6f}"
        )


# ============================================================
# Export Latents
# ============================================================

def export_latents(model, loader, device):

    model.eval()

    zs = []

    with torch.no_grad():

        for x in loader:

            x = x.to(device)

            x = x.permute(0, 2, 1)

            _, z = model(x)

            zs.append(z.cpu())

    return torch.cat(zs).numpy()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)

    parser.add_argument("--epochs", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--latent_steps", type=int, default=16)

    parser.add_argument("--d_model", type=int, default=128)

    parser.add_argument("--nhead", type=int, default=8)

    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument("--out_dir", default="transformer_ae_out")

    parser.add_argument(
        "--feature_idx",
        type=int,
        nargs="+",
        default=None
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------

    ds = WindowDataset(
        args.dataset,
        args.feature_idx
    )

    L, C = ds[0].shape

    print("Dataset shape:", len(ds), L, C)

    # ------------------------------------------------

    train_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True
    )

    export_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False
    )

    # ------------------------------------------------

    model = TransformerTimeAE(
        channels=C,
        seq_len=L,
        latent_steps=args.latent_steps,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers
    )

    # ------------------------------------------------

    train(
        model,
        train_loader,
        args.epochs,
        args.lr,
        device
    )

    # ------------------------------------------------

    out = Path(args.out_dir)

    (out / "models").mkdir(
        parents=True,
        exist_ok=True
    )

    (out / "latents").mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------

    config = {
        "dataset": args.dataset,
        "latent_steps": args.latent_steps,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "num_features": C,
        "feature_idx": (
            args.feature_idx
            if args.feature_idx is not None
            else "all"
        ),
        "seq_len": L,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr
    }

    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # ------------------------------------------------

    torch.save(
        model.state_dict(),
        out / "models" / f"transformer_ae_{args.latent_steps}.pt"
    )

    # ------------------------------------------------

    latents = export_latents(
        model,
        export_loader,
        device
    )

    np.save(
        out / "latents" / f"latent_dataset_{args.latent_steps}.npy",
        latents
    )

    print("Latent shape:", latents.shape)


if __name__ == "__main__":

    main()