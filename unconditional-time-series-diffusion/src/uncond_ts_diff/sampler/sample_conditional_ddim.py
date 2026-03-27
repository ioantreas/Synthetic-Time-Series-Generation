import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.utils import extract


# =========================================================
# AUTOENCODER
# =========================================================

class DWBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.pw = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.dw(x)
        h = self.pw(h)
        h = self.norm(h)
        h = self.act(h)
        return x + h


class SiloTimeOnlyAE(nn.Module):

    def __init__(self, channels, seq_len, latent_steps=16):

        super().__init__()

        self.channels = channels
        self.seq_len = seq_len
        self.latent_steps = latent_steps

        # -------- Encoder --------
        self.block1 = DWBlock(channels)
        self.down1 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

        self.block2 = DWBlock(channels)
        self.down2 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

        self.block3 = DWBlock(channels)
        self.down3 = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

        self.block4 = DWBlock(channels)
        self.to_latent = nn.Conv1d(channels, channels, 1)

        # -------- Decoder --------
        self.up1 = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.block5 = DWBlock(channels)

        self.up2 = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.block6 = DWBlock(channels)

        self.up3 = nn.ConvTranspose1d(channels, channels, 4, 2, 1)
        self.block7 = DWBlock(channels)

    def decode(self, z):
        h = self.up1(z)
        h = self.block5(h)
        h = self.up2(h)
        h = self.block6(h)
        h = self.up3(h)
        h = self.block7(h)
        h = F.interpolate(h, size=self.seq_len, mode="linear", align_corners=False)
        return h


# =========================================================
# GUIDANCE FUNCTION
# =========================================================

def guidance_fn(x_recon, x_latent, t):
    mid = x_recon.shape[1] // 2
    return ((x_recon[:, mid-10:mid+10, 0] - 3.0)**2).mean()

# =========================================================
# GUIDED SAMPLING
# =========================================================

def sample_guided(model, decoder, num_samples, seq_len, num_channels, device, guidance_scale=5000):

    x = torch.randn(num_samples, seq_len, num_channels, device=device)

    sigma_t = 0.1

    for i in reversed(range(model.timesteps)):
        noise = torch.randn_like(x)
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)

        # predict noise
        with torch.no_grad():
            eps = model.backbone(x, t, None)

        # clean estimate
        x0_hat = model.fast_denoise(x, t, None, noise=eps)

        # guidance
        x0_hat = x0_hat.detach().requires_grad_(True)

        x_recon = decoder(x0_hat)
        loss = guidance_fn(x_recon, x0_hat, t)

        grad = torch.autograd.grad(loss, x0_hat)[0]
        grad = torch.clamp(grad, -1.0, 1.0)

        x0_guided = (x0_hat - guidance_scale * grad).detach()

        # reverse step
        alpha_bar_prev = extract(model.alphas_cumprod_prev, t, x.shape)
        sqrt_ab_prev = torch.sqrt(alpha_bar_prev)
        safe_term = torch.clamp(1.0 - alpha_bar_prev - sigma_t**2, min=1e-8)
        sqrt_one_minus_ab_prev = torch.sqrt(safe_term)

        if i > 0:
            x = sqrt_ab_prev * x0_guided + sqrt_one_minus_ab_prev * eps + sigma_t * noise
        else:
            x = x0_guided

    return x


# =========================================================
# BASELINE SAMPLING
# =========================================================

@torch.no_grad()
def sample_unconditional(model, num_samples, seq_len, num_channels, device):
    x = torch.randn(num_samples, seq_len, num_channels, device=device)

    for i in reversed(range(model.timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        x = model.p_sample(x, t, i, features=None)

    return x


# =========================================================
# MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--decoder_path", type=str, required=True)

    parser.add_argument("--latent_steps", type=int, required=True)
    parser.add_argument("--orig_seq_len", type=int, required=True)

    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--train_subset", type=int, default=1024)

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ------------------------
    # Load diffusion model
    # ------------------------
    ckpt_path = Path(f"../../../results/lightning_logs/version_{args.version}/checkpoints/last.ckpt")

    print("Loading diffusion:", ckpt_path)
    model = TSDiff.load_from_checkpoint(ckpt_path).to(device)

    num_channels = model.backbone.input_init[0].in_features

    # ------------------------
    # Load Training Data for guidance
    # ------------------------
    print("Loading train data:", args.train_data)

    train_data = np.load(args.train_data)   # (N, T, C)

    # random subset
    idx = np.random.choice(len(train_data), size=args.train_subset, replace=False)
    global train_bank
    train_bank = torch.from_numpy(train_data[idx]).float().to(device)

    print("Train subset shape:", train_bank.shape)

    # ------------------------
    # Load AE
    # ------------------------
    print("Loading AE:", args.decoder_path)

    ae = SiloTimeOnlyAE(
        channels=num_channels,
        seq_len=args.orig_seq_len,
        latent_steps=args.latent_steps
    )

    ae.load_state_dict(torch.load(args.decoder_path, map_location=device))
    ae = ae.to(device).eval()

    # wrapper
    def decoder(z):
        z = z.permute(0, 2, 1)
        x = ae.decode(z)
        x = x.permute(0, 2, 1)
        return x

    # ------------------------
    # Sampling
    # ------------------------
    samples = sample_guided(
        model,
        decoder,
        args.num_samples,
        args.latent_steps,
        num_channels,
        device,
    )

    np.save("samples.npy", samples.cpu().numpy())

    print("Done.")
    print("Shape:", samples.shape)


if __name__ == "__main__":
    main()