# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import argparse
from pathlib import Path

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.data = torch.from_numpy(np.load(path)).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def create_model(config):
    model = TSDiff(
        **getattr(diffusion_configs, config["diffusion_config"]),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization=config["normalization"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        lr=config["lr"],
        init_skip=config["init_skip"],
    )
    model.to(config["device"])
    return model


def main(config, log_dir):
    # Create model
    model = create_model(config)

    # Setup dataset and data loading
    train_dataset = SequenceDataset(config["dataset"])

    data_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        filename="tsdiff-{epoch:03d}-{train_loss:.3f}",
        save_weights_only=True,
    )

    callbacks = [checkpoint_callback, RichProgressBar()]

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=[int(config["device"].split(":")[-1])],
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=log_dir,
        gradient_clip_val=config.get("gradient_clip_val", None),
    )
    logger.info(f"Logging to {trainer.logger.log_dir}")
    trainer.fit(model, train_dataloaders=data_loader)
    logger.info("Training completed.")

    best_ckpt_path = Path(trainer.logger.log_dir) / "best_checkpoint.ckpt"

    if not best_ckpt_path.exists():
        torch.save(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
            best_ckpt_path,
        )
    logger.info(f"Loading {best_ckpt_path}.")
    best_state_dict = torch.load(best_ckpt_path)
    model.load_state_dict(best_state_dict, strict=True)

    with open(Path(trainer.logger.log_dir) / "config.yaml", "w") as fp:
        yaml.dump(config, fp)

if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./", help="Path to results dir"
    )

    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    main(config=config, log_dir=args.out_dir)
