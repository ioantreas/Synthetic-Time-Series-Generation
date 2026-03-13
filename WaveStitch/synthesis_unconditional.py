# sample_unconditional.py
import argparse
import os
import numpy as np
import torch

from training_utils import fetchModel, fetchDiffusionConfig


@torch.no_grad()
def sample_ddpm(model, diffusion_config, num_samples, seq_len, channels, device):
    """
    Unconditional DDPM sampling.
    Returns: (num_samples, seq_len, channels) torch tensor
    """
    Tdiff = diffusion_config["T"]
    alphas = diffusion_config["alphas"].to(device)         # (Tdiff,)
    alpha_bars = diffusion_config["alpha_bars"].to(device) # (Tdiff,)
    betas = diffusion_config["betas"].to(device)           # (Tdiff,)

    x = torch.randn(num_samples, seq_len, channels, device=device)

    for t in range(Tdiff - 1, -1, -1):
        t_in = torch.full((num_samples, 1), t, device=device, dtype=torch.long)

        # model returns (B, C, T) -> convert to (B, T, C)
        eps_hat = model(x, t_in).permute(0, 2, 1).contiguous()

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]

        # DDPM mean (epsilon-prediction form)
        mean = (1.0 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_hat)

        if t > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t) * z
        else:
            x = mean

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ckpt", type=str, required=True, help="path to model .pth")
    parser.add_argument("-backbone", type=str, default="s4")
    parser.add_argument("-timesteps", "-T", type=int, default=100)

    # model hparams must match training
    parser.add_argument("-num_res_layers", type=int, default=8)
    parser.add_argument("-res_channels", type=int, default=128)
    parser.add_argument("-skip_channels", type=int, default=128)
    parser.add_argument("-diff_step_embed_in", type=int, default=64)
    parser.add_argument("-diff_step_embed_mid", type=int, default=128)
    parser.add_argument("-diff_step_embed_out", type=int, default=128)
    parser.add_argument("-s4_lmax", type=int, default=256)
    parser.add_argument("-s4_dstate", type=int, default=64)
    parser.add_argument("-s4_dropout", type=float, default=0.0)
    parser.add_argument("-s4_bidirectional", type=bool, default=True)
    parser.add_argument("-s4_layernorm", type=bool, default=True)

    # sampling params
    parser.add_argument("-n", "--num_samples", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--channels", type=int, default=28)
    parser.add_argument("--out", type=str, default="synthetic_samples.npy")

    # diffusion schedule
    parser.add_argument("-beta_0", type=float, default=0.0001)
    parser.add_argument("-beta_T", type=float, default=0.02)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a dummy args-like object for fetchModel/fetchDiffusionConfig
    class A: pass
    a = A()
    for k, v in vars(args).items():
        setattr(a, k, v)

    # Instantiate + load
    model = fetchModel(args.channels, args.channels, a).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    diffusion_config = fetchDiffusionConfig(a)

    x = sample_ddpm(model, diffusion_config, args.num_samples, args.seq_len, args.channels, device)
    x_np = x.detach().cpu().numpy()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.save(args.out, x_np)
    print(f"[done] wrote {args.out} with shape {x_np.shape}")


if __name__ == "__main__":
    main()