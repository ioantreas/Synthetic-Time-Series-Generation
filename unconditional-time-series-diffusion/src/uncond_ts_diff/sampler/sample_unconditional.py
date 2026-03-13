import argparse
from pathlib import Path
import numpy as np
import torch

from uncond_ts_diff.model import TSDiff


@torch.no_grad()
def sample_unconditional(model, num_samples, seq_len, num_channels, device):
    model.eval()
    x = torch.randn(num_samples, seq_len, num_channels, device=device)

    for i in reversed(range(model.timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        x = model.p_sample(x, t, i, features=None)

    return x


def resolve_checkpoint(version, base_dir: str):
    base = Path("../../../results/lightning_logs") / f"version_{version}" / "checkpoints"
    if not base.exists():
        raise FileNotFoundError(f"No such version folder: {base}")

    ckpts = sorted(base.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {base}")

    for c in ckpts:
        if c.name == "last.ckpt":
            return c
    return ckpts[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True, help="Lightning version number")
    parser.add_argument("--results_dir", type=str, default="./results", help="Folder that contains lightning_logs/")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seq_len", type=int, default=168)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default=None, help="Optional output path for .npy")

    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ckpt_path = resolve_checkpoint(args.version, args.results_dir)

    # Default: save next to checkpoint
    out_path = Path(args.out) if args.out else (ckpt_path.parent / "synthetic_samples.npy")

    print(f"Using checkpoint: {ckpt_path}")
    print(f"Saving samples to: {out_path}")

    model = TSDiff.load_from_checkpoint(ckpt_path)
    model = model.to(device)

    # Infer channels from the trained model
    num_channels = getattr(model, "input_dim", None)
    if num_channels is None:
        # Fallback: try backbone first linear layer
        num_channels = model.backbone.input_init[0].in_features

    samples = sample_unconditional(
        model=model,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        num_channels=num_channels,
        device=device,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, samples.cpu().numpy())

    print("Done.")
    print("shape:", tuple(samples.shape), "mean:", samples.mean().item(), "std:", samples.std().item())


if __name__ == "__main__":
    main()



