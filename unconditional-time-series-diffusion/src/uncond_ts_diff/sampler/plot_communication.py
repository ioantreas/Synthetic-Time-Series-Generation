import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spinning", required=True,
                        help="Spinning Decoder summary CSV (the file with total_bytes_mean, etc.)")
    parser.add_argument("--vigil", required=True,
                        help="VIGIL communication CSV containing *_mb columns")
    parser.add_argument("--out_dir", default="communication_plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spinning = pd.read_csv(args.spinning)
    vigil = pd.read_csv(args.vigil)

    # ------------------------------------------------------------------
    # Read one representative row from each file.
    # Communication is constant across dataset/scenario in both files.
    # We also assert this, so the script fails loudly if that ever changes.
    # ------------------------------------------------------------------
    spin_total_bytes = spinning["total_bytes_mean"].to_numpy(dtype=float)
    spin_c2s_bytes = spinning["client_to_server_bytes_mean"].to_numpy(dtype=float)
    spin_s2c_bytes = spinning["server_to_client_bytes_mean"].to_numpy(dtype=float)

    if not np.allclose(spin_total_bytes, spin_total_bytes[0]):
        raise ValueError("Spinning total communication is not constant across rows.")
    if not np.allclose(spin_c2s_bytes, spin_c2s_bytes[0]):
        raise ValueError("Spinning client->server communication is not constant across rows.")
    if not np.allclose(spin_s2c_bytes, spin_s2c_bytes[0]):
        raise ValueError("Spinning server->client communication is not constant across rows.")

    for col in ["client_to_server_mb", "server_to_client_mb", "total_mb", "per_imputation_mb"]:
        vals = vigil[col].to_numpy(dtype=float)
        if not np.allclose(vals, vals[0]):
            raise ValueError(f"VIGIL {col} is not constant across rows.")

    vigil_c2s_mb = float(vigil["client_to_server_mb"].iloc[0])
    vigil_s2c_mb = float(vigil["server_to_client_mb"].iloc[0])
    vigil_total_mb = float(vigil["total_mb"].iloc[0])

    spin_total_mb = float(spin_total_bytes[0]) / 1e6

    # One VIGIL latent tensor is half of the initial upload:
    # anchor + confidence map = 2 latent-sized tensors.
    latent_tensor_mb = vigil_c2s_mb / 2.0

    # Spinning sends one latent-sized tensor in each direction per denoising step.
    # Therefore each denoising step transfers 2 * latent_tensor_mb.
    spin_mb_per_step = 2.0 * latent_tensor_mb

    # Infer total cumulative denoising steps from the measured Spinning total.
    total_steps_float = spin_total_mb / spin_mb_per_step
    total_steps = int(round(total_steps_float))

    if not np.isclose(total_steps_float, total_steps):
        raise ValueError(
            f"Could not infer an integer number of denoising steps: {total_steps_float}"
        )

    print(f"Latent tensor size: {latent_tensor_mb:.3f} MB")
    print(f"Spinning total: {spin_total_mb:.3f} MB")
    print(f"VIGIL total: {vigil_total_mb:.3f} MB")
    print(f"Inferred cumulative denoising steps: {total_steps}")
    print(f"Spinning one-way transfers: {2 * total_steps}")
    print("VIGIL one-way transfers: 2")

    # ------------------------------------------------------------------
    # Plot 1: cumulative communication volume
    # ------------------------------------------------------------------
    x = np.arange(total_steps + 1)
    spin_cumulative_mb = spin_mb_per_step * x

    # VIGIL:
    # x=0: one client->server upload containing anchor + confidence map.
    # During all denoising steps: no additional communication.
    # At completion: one server->client return containing all generated latents.
    vigil_x = np.array([0, total_steps, total_steps])
    vigil_y = np.array([vigil_c2s_mb, vigil_c2s_mb, vigil_total_mb])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, spin_cumulative_mb, label="Spinning Decoder", linewidth=2)
    ax.plot(vigil_x, vigil_y, label="VIGIL", linewidth=2, drawstyle="steps-post")
    ax.set_xlabel("Cumulative reverse-diffusion steps")
    ax.set_ylabel("Cumulative communication (MB)")
    ax.set_yscale("log")
    ax.set_xlim(0, total_steps)
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "communication_volume_over_sampling.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "communication_volume_over_sampling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: cumulative one-way communication events
    # ------------------------------------------------------------------
    spin_events = 2 * x

    # VIGIL has one upload before sampling and one download at the end.
    vigil_event_x = np.array([0, total_steps, total_steps])
    vigil_events = np.array([1, 1, 2])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, spin_events, label="Spinning Decoder", linewidth=2)
    ax.plot(vigil_event_x, vigil_events, label="VIGIL", linewidth=2, drawstyle="steps-post")
    ax.set_xlabel("Cumulative reverse-diffusion steps")
    ax.set_ylabel("Cumulative one-way communication events")
    ax.set_xlim(0, total_steps)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "communication_events_over_sampling.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "communication_events_over_sampling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
