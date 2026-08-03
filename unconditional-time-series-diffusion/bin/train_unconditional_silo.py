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
    # Load data (multi-silo support)
    # --------------------
    dataset_paths = config["dataset"]

    # allow single path OR list of paths
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    arrays = []
    for p in dataset_paths:
        arr = np.load(p)
        if arr.ndim != 3:
            raise ValueError(f"{p} is not (N,T,C): got {arr.shape}")
        arrays.append(arr)

    # check alignment
    N, T = arrays[0].shape[0], arrays[0].shape[1]
    for i, arr in enumerate(arrays):
        if arr.shape[0] != N or arr.shape[1] != T:
            raise ValueError(f"Dataset {i} not aligned: {arr.shape} vs {(N,T)}")

    # concatenate along channel dimension
    data = np.concatenate(arrays, axis=2)

    print("Loaded multi-silo data:")
    for i, arr in enumerate(arrays):
        print(f"  silo {i}: {arr.shape}")
    print("Final concatenated:", data.shape)

    # optional: log channel splits
    channel_splits = []
    start = 0
    for arr in arrays:
        c = arr.shape[2]
        channel_splits.append((start, start + c))
        start += c
    print("Channel splits:", channel_splits)

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
    C = data.shape[2]
    print(f"Using ALL concatenated channels: C={C}")

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

    main(config, args.out_dir)
