import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


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
# SAME BLOCK AS AE
# ============================================================

class DWBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.dw = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels
        )

        self.pw = nn.Conv1d(
            channels,
            channels,
            kernel_size=1
        )

        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.dw(x)
        h = self.pw(h)
        h = self.norm(h)
        h = self.act(h)
        return x + h


# ============================================================
# DECODER-LIKE ATTACKER
# ============================================================

class AttackerDecoder(nn.Module):

    def __init__(self, channels, target_len):
        super().__init__()

        self.target_len = target_len

        self.up1 = nn.ConvTranspose1d(
            channels, channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.block1 = DWBlock(channels)

        self.up2 = nn.ConvTranspose1d(
            channels, channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.block2 = DWBlock(channels)

        self.up3 = nn.ConvTranspose1d(
            channels, channels,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.block3 = DWBlock(channels)

    def forward(self, z):
        # (B, latent_len, C) → (B, C, latent_len)
        z = z.permute(0, 2, 1)

        h = self.up1(z)
        h = self.block1(h)

        h = self.up2(h)
        h = self.block2(h)

        h = self.up3(h)
        h = self.block3(h)

        h = nn.functional.interpolate(
            h,
            size=self.target_len,
            mode="linear",
            align_corners=False
        )

        return h.permute(0, 2, 1)


# ============================================================
# Training
# ============================================================

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for z, x in loader:
        z, x = z.to(device), x.to(device)

        pred = model(z)
        loss = criterion(pred, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, device):
    model.eval()
    total_mse = 0

    with torch.no_grad():
        for z, x in loader:
            z, x = z.to(device), x.to(device)
            pred = model(z)
            mse = ((pred - x) ** 2).mean()
            total_mse += mse.item()

    return total_mse / len(loader)


def get_predictions(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for z, _ in loader:
            z = z.to(device)
            pred = model(z)
            preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0)


# ============================================================
# Main
# ============================================================

def main(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading data...")

    train_latents = np.load(args.train_latents)
    test_latents = np.load(args.test_latents)

    train_seq = np.load(args.train_seq)
    test_seq = np.load(args.test_seq)

    print("Train latents:", train_latents.shape)
    print("Train seq:", train_seq.shape)

    channels = train_seq.shape[2]
    target_len = train_seq.shape[1]

    train_ds = LatentDataset(train_latents, train_seq)
    test_ds = LatentDataset(test_latents, test_seq)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = AttackerDecoder(channels, target_len).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.5)

    print("\nTraining decoder-aware attacker...")

    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_mse = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.6f} | Test MSE: {test_mse:.6f}")

    final_mse = evaluate(model, test_loader, device)

    print("\nFinal Test MSE:", final_mse)

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "attacker_decoder.pt")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"test_mse": final_mse}, f, indent=2)

    if args.save_preds:
        preds = get_predictions(model, test_loader, device)
        np.save(out_dir / "predictions.npy", preds)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_latents", type=str, required=True)
    parser.add_argument("--test_latents", type=str, required=True)
    parser.add_argument("--train_seq", type=str, required=True)
    parser.add_argument("--test_seq", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--save_preds", action="store_true")

    args = parser.parse_args()

    main(args)