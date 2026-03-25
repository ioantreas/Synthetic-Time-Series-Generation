import logging
import argparse
from pathlib import Path
import yaml
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff


# =========================================================
# Dataset
# =========================================================
class SequenceDataset(Dataset):
    def __init__(self, data):
        # data: (N, T, C)
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # returns (T, C)
        return self.data[idx]


# =========================================================
# Model creation
# =========================================================
def create_model(config, num_channels: int):

    diff_cfg = getattr(diffusion_configs, config["diffusion_config"])

    backbone_params = dict(diff_cfg["backbone_parameters"])
    backbone_params["input_dim"] = num_channels
    backbone_params["output_dim"] = num_channels

    model = TSDiff(
        backbone_parameters=backbone_params,
        timesteps=diff_cfg["timesteps"],
        diffusion_scheduler=diff_cfg["diffusion_scheduler"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        freq=config.get("freq", None),
        normalization=config.get("normalization", "none"),
        use_features=False,
        use_lags=False,
        lr=config.get("lr", 1e-3),
        init_skip=False,
    )

    return model


# =========================================================
# Main
# =========================================================
def main(config, log_dir):
    # --------------------
    # Load data
    # --------------------
    data = np.load(config["dataset"])  # expected (N, T, C)
    if data.ndim != 3:
        raise ValueError(f"Expected .npy with shape (N,T,C). Got {data.shape}")

    N, T, C = data.shape
    print("Loaded data:", data.shape)
    print(f"Using ALL channels: C={C}")

    dataset = SequenceDataset(data)

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=True,
    )

    # --------------------
    # Build model
    # --------------------
    model = create_model(config, num_channels=C)

    # --------------------
    # Trainer
    # --------------------
    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        every_n_epochs=config.get("save_every_n_epochs", 10),
    )

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        default_root_dir=log_dir,
        log_every_n_steps=10,
    )

    # --------------------
    # Train
    # --------------------
    trainer.fit(model, train_dataloaders=loader)

    print("Training finished.")
    print("Checkpoint saved to:", trainer.checkpoint_callback.last_model_path)


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./results")

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config=config, log_dir=args.out_dir)