import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

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
# MASK / IMPUTATION HELPERS
# =========================================================

def block_mask(data, block_size=40):
    mask = np.ones_like(data, dtype=np.float32)
    for i in range(data.shape[0]):
        start = np.random.randint(0, data.shape[1] - block_size)
        mask[i, start:start + block_size, :] = 0.0
    return mask


def make_observed(full_data, mask):
    x_obs = full_data.copy()
    x_obs[mask == 0] = 0.0
    return x_obs


# =========================================================
# GUIDANCE FUNCTION
# =========================================================

def guidance_fn(x_recon, x_obs, mask, smooth_weight=0.0):
    # observed matching loss
    obs_count = mask.sum(dim=(1, 2)) + 1e-8
    obs_loss = ((((x_recon - x_obs) * mask) ** 2).sum(dim=(1, 2)) / obs_count).mean()

    if smooth_weight > 0:
        dx = x_recon[:, 1:] - x_recon[:, :-1]
        missing_mid = (1.0 - mask[:, 1:]) * (1.0 - mask[:, :-1])
        smooth_loss = (dx.pow(2) * missing_mid).mean()
    else:
        smooth_loss = torch.tensor(0.0, device=x_recon.device)

    return obs_loss + smooth_weight * smooth_loss


# =========================================================
# GUIDED DDPM SAMPLING
# =========================================================

def sample_guided(
        model,
        decoder,
        x_obs,
        mask,
        num_samples,
        seq_len,
        num_channels,
        device,
        guidance_scale=100.0,
        inner_steps=5,
        smooth_weight=0.0,
        grad_normalize=True,
        grad_clip_value=None,
        use_schedule=True,
):
    x = torch.randn(num_samples, seq_len, num_channels, device=device)

    for i in reversed(range(model.timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)

        # ------------------------
        # predict eps
        # ------------------------
        with torch.no_grad():
            eps = model.backbone(x, t, None)

        # ------------------------
        # compute x0_hat from eps
        # ------------------------
        x0_hat = model.fast_denoise(x, t, None, noise=eps)

        # ------------------------
        # guidance on x0_hat
        # ------------------------
        if guidance_scale > 0:
            if use_schedule:
                w = guidance_scale * (1 - i / model.timesteps)
            else:
                w = guidance_scale

            x0_guided = x0_hat.detach()

            for _ in range(inner_steps):
                x0_guided = x0_guided.detach().requires_grad_(True)

                x_recon = decoder(x0_guided)
                loss = guidance_fn(x_recon, x_obs, mask, smooth_weight=smooth_weight)

                grad = torch.autograd.grad(loss, x0_guided)[0]

                if grad_normalize:
                    grad = grad / (grad.norm(dim=(1, 2), keepdim=True) + 1e-8)

                if grad_clip_value is not None:
                    grad = torch.clamp(grad, -grad_clip_value, grad_clip_value)

                x0_guided = (x0_guided - w * grad).detach()
        else:
            x0_guided = x0_hat

        # ------------------------
        # recompute eps from guided x0
        # THIS is the part you asked to keep
        # ------------------------
        if guidance_scale > 0:
            sqrt_alpha_bar_t = extract(model.sqrt_alphas_cumprod, t, x.shape)
            sqrt_one_minus_alpha_bar_t = extract(model.sqrt_one_minus_alphas_cumprod, t, x.shape)

            eps = (x - sqrt_alpha_bar_t * x0_guided) / (sqrt_one_minus_alpha_bar_t + 1e-8)

        # ------------------------
        # original DDPM update using guided eps
        # ------------------------
        betas_t = extract(model.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(model.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(model.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * eps / (sqrt_one_minus_alphas_cumprod_t + 1e-8)
        )

        if i > 0:
            posterior_variance_t = extract(model.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(torch.clamp(posterior_variance_t, min=1e-12)) * noise
        else:
            # either model_mean or x0_guided can be used here;
            # using x0_guided is usually more direct for your setup
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

    parser.add_argument("--block_size", type=int, default=40)

    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--inner_steps", type=int, default=1)
    parser.add_argument("--smooth_weight", type=float, default=0.0)
    parser.add_argument("--grad_clip_value", type=float, default=-1.0)
    parser.add_argument("--no_grad_normalize", action="store_true")
    parser.add_argument("--no_schedule", action="store_true")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # ------------------------
    # Load diffusion model
    # ------------------------
    ckpt_path = Path(
        f"../../../results/lightning_logs/version_{args.version}/checkpoints/last.ckpt"
    )

    print("Loading diffusion:", ckpt_path)
    model = TSDiff.load_from_checkpoint(ckpt_path).to(device).eval()

    num_channels = model.backbone.input_init[0].in_features

    # ------------------------
    # Load training data
    # ------------------------
    print("Loading train data:", args.train_data)
    train_data = np.load(args.train_data)

    idx = np.random.choice(len(train_data), size=args.train_subset, replace=False)
    train_bank = torch.from_numpy(train_data[idx]).float().to(device)
    print("Train subset shape:", train_bank.shape)

    # ------------------------
    # Load AE
    # ------------------------
    print("Loading AE:", args.decoder_path)

    ae = SiloTimeOnlyAE(
        channels=num_channels,
        seq_len=args.orig_seq_len,
        latent_steps=args.latent_steps,
    )

    ae.load_state_dict(torch.load(args.decoder_path, map_location=device))
    ae = ae.to(device).eval()

    def decoder(z):
        z = z.permute(0, 2, 1)
        x = ae.decode(z)
        x = x.permute(0, 2, 1)
        return x

    # ------------------------
    # Create imputation input
    # ------------------------
    full_data = train_data[:args.num_samples]
    mask_np = block_mask(full_data, block_size=args.block_size)
    x_obs_np = make_observed(full_data, mask_np)

    x_obs = torch.from_numpy(x_obs_np).float().to(device)
    mask = torch.from_numpy(mask_np).float().to(device)
    x_full = torch.from_numpy(full_data).float().to(device)

    # ------------------------
    # Sampling
    # ------------------------
    grad_clip_value = None if args.grad_clip_value < 0 else args.grad_clip_value

    samples = sample_guided(
        model=model,
        decoder=decoder,
        x_obs=x_obs,
        mask=mask,
        num_samples=args.num_samples,
        seq_len=args.latent_steps,
        num_channels=num_channels,
        device=device,
        guidance_scale=args.guidance_scale,
        inner_steps=args.inner_steps,
        smooth_weight=args.smooth_weight,
        grad_normalize=not args.no_grad_normalize,
        grad_clip_value=grad_clip_value,
        use_schedule=not args.no_schedule,
    )

    np.save("samples.npy", samples.detach().cpu().numpy())

    print("Done.")
    print("Latent sample shape:", samples.shape)

    # ------------------------
    # Decode and evaluate
    # ------------------------
    with torch.no_grad():
        decoded = decoder(samples)

    mse_missing = ((decoded - x_full)[mask == 0] ** 2).mean()
    mse_observed = ((decoded - x_full)[mask == 1] ** 2).mean()

    print("MSE missing:", mse_missing.item())
    print("MSE observed:", mse_observed.item())

    # ------------------------
    # Plot one example
    # ------------------------
    i = 0
    ch = 0

    plt.figure(figsize=(10, 4))
    plt.plot(x_obs[i, :, ch].cpu(), label="observed")
    plt.plot(x_full[i, :, ch].cpu(), "--", label="ground truth")
    plt.plot(decoded[i, :, ch].detach().cpu(), label="imputed")
    plt.legend()
    plt.title("Imputation check")
    plt.show()


if __name__ == "__main__":
    main()