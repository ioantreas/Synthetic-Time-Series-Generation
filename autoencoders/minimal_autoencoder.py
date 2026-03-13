import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

class WindowDataset(Dataset):

    def __init__(self, path):
        x = np.load(path).astype(np.float32)   # (N,L,C)
        self.x = torch.from_numpy(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


# ============================================================
# Depthwise + Pointwise Block
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

    def forward(self,x):

        h = self.dw(x)
        h = self.pw(h)
        h = self.norm(h)
        h = self.act(h)

        return x + h


# ============================================================
# Model
# ============================================================

class SiloTimeOnlyAE(nn.Module):

    """
    Compress only time:
        (N,C,L) -> (N,C,latent_steps)

    Channels remain preserved.
    """

    def __init__(self, channels, seq_len, latent_steps=16):

        super().__init__()

        self.channels = channels
        self.seq_len = seq_len
        self.latent_steps = latent_steps


        # ---------------- Encoder ----------------

        self.block1 = DWBlock(channels)
        self.down1 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

        self.block2 = DWBlock(channels)
        self.down2 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

        self.block3 = DWBlock(channels)
        self.down3 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

        self.block4 = DWBlock(channels)

        self.to_latent = nn.Conv1d(channels, channels, 1)


        # ---------------- Decoder ----------------

        self.up1 = nn.ConvTranspose1d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block5 = DWBlock(channels)

        self.up2 = nn.ConvTranspose1d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block6 = DWBlock(channels)

        self.up3 = nn.ConvTranspose1d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.block7 = DWBlock(channels)


    # ------------------------------------------------

    def encode(self,x):

        h = self.block1(x)
        h = self.down1(h)

        h = self.block2(h)
        h = self.down2(h)

        h = self.block3(h)
        h = self.down3(h)

        h = self.block4(h)

        z = self.to_latent(h)

        z = torch.nn.functional.interpolate(
            z,
            size=self.latent_steps,
            mode="linear",
            align_corners=False
        )

        return z


    # ------------------------------------------------

    def decode(self,z):

        h = self.up1(z)
        h = self.block5(h)

        h = self.up2(h)
        h = self.block6(h)

        h = self.up3(h)
        h = self.block7(h)

        h = torch.nn.functional.interpolate(
            h,
            size=self.seq_len,
            mode="linear",
            align_corners=False
        )

        return h


    # ------------------------------------------------

    def forward(self,x):

        z = self.encode(x)
        xhat = self.decode(z)

        return xhat,z


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

        total = 0

        for x in loader:

            x = x.to(device)
            x = x.permute(0,2,1)

            xhat,_ = model(x)

            loss = loss_fn(xhat,x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()

        print(f"epoch {ep:03d} | loss {total/len(loader):.6f}")


# ============================================================
# Export Latents
# ============================================================

def export_latents(model, loader, device):

    model.eval()

    zs = []

    with torch.no_grad():

        for x in loader:

            x = x.to(device)
            x = x.permute(0,2,1)

            _,z = model(x)

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
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--latent_steps", type=int, default=16)
    parser.add_argument("--out_dir", default="silo_ae_out")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"


    ds = WindowDataset(args.dataset)

    L,C = ds[0].shape

    print("Dataset shape:",len(ds),L,C)


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


    model = SiloTimeOnlyAE(
        channels=C,
        seq_len=L,
        latent_steps=args.latent_steps
    )


    train(model,train_loader,args.epochs,args.lr,device)


    out = Path(args.out_dir)

    (out/"models").mkdir(parents=True,exist_ok=True)
    (out/"latents").mkdir(parents=True,exist_ok=True)


    torch.save(
        model.state_dict(),
        out/"models"/f"silo_time_only_ae_{args.latent_steps}.pt"
    )


    latents = export_latents(model,export_loader,device)


    np.save(
        out/"latents"/f"latent_dataset_{args.latent_steps}.npy",
        latents
    )


    print("Latent shape:",latents.shape)


if __name__ == "__main__":

    main()