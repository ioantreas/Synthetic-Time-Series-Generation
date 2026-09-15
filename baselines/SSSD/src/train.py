import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from utils.util import (
    find_max_epoch,
    print_size,
    training_loss,
    calc_diffusion_hyperparams,
    get_mask_mnr,
    get_mask_bm,
    get_mask_rm,
)

from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer


def make_batches(data, batch_size, shuffle=True):
    """Yield NumPy batches without introducing a DataLoader."""
    indices = np.arange(len(data))
    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        yield data[batch_indices]


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k,
          data_path,
          batch_size):

    # Load custom windows directly. Expected shape: (N, T, C).
    training_data = np.load(data_path).astype(np.float32)
    if training_data.ndim != 3:
        raise ValueError(
            f"Expected training data with shape (N, T, C), got {training_data.shape}"
        )

    _, sequence_length, num_channels = training_data.shape
    print(f"Loaded training data: {training_data.shape}", flush=True)

    # Match the model dimensions to the supplied dataset.
    model_config["in_channels"] = num_channels
    model_config["out_channels"] = num_channels
    if use_model == 2:
        model_config["s4_lmax"] = sequence_length

    local_path = "T{}_beta0{}_betaT{}".format(
        diffusion_config["T"],
        diffusion_config["beta_0"],
        diffusion_config["beta_T"],
    )

    # output_directory = os.path.join(output_directory, local_path)
    output_directory = os.path.join(output_directory, f"{masking}_k{missing_k}", local_path)
    os.makedirs(output_directory, exist_ok=True)
    print("output directory", output_directory, flush=True)

    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    if use_model == 0:
        net = DiffWaveImputer(**model_config).cuda()
    elif use_model == 1:
        net = SSSDSAImputer(**model_config).cuda()
    elif use_model == 2:
        net = SSSDS4Imputer(**model_config).cuda()
    else:
        raise ValueError(f"Unknown model index: {use_model}")

    print_size(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    if ckpt_iter == "max":
        ckpt_iter = find_max_epoch(output_directory)
    else:
        ckpt_iter = int(ckpt_iter)

    if ckpt_iter >= 0:
        try:
            model_path = os.path.join(output_directory, f"{ckpt_iter}.pkl")
            checkpoint = torch.load(model_path, map_location="cpu")
            net.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Successfully loaded model at iteration {ckpt_iter}")
        except Exception as exc:
            ckpt_iter = -1
            print(f"Could not load checkpoint; training from scratch. Reason: {exc}")
    else:
        ckpt_iter = -1
        print("No checkpoint requested; training from scratch.")

    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for batch_np in make_batches(training_data, batch_size, shuffle=True):
            batch = torch.from_numpy(batch_np).float().cuda()  # (B, T, C)

            # Preserve the authors' masking behavior: create one (T, C) mask
            # from the first sample and repeat it across the batch.
            if masking == "rm":
                transposed_mask = get_mask_rm(batch[0], missing_k)
            elif masking == "mnr":
                transposed_mask = get_mask_mnr(batch[0], missing_k)
            elif masking == "bm":
                transposed_mask = get_mask_bm(batch[0], missing_k)
            else:
                raise ValueError(f"Unknown masking mode: {masking}")

            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(batch.size(0), 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            batch = batch.permute(0, 2, 1)  # (B, C, T)

            assert batch.size() == mask.size() == loss_mask.size()

            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            loss = training_loss(
                net,
                nn.MSELoss(),
                X,
                diffusion_hyperparams,
                only_generate_missing=only_generate_missing,
            )
            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print(f"iteration: {n_iter}\tloss: {loss.item()}")

            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = f"{n_iter}.pkl"
                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(output_directory, checkpoint_name),
                )
                print(f"model at iteration {n_iter} is saved")

            n_iter += 1
            if n_iter >= n_iters + 1:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/config_SSSDS4.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_directory", type=str, default=None,
                        help="Optional override for train_config.output_directory")
    parser.add_argument("--ckpt_iter", default=None,
                        help='Optional checkpoint iteration or "max"')
    parser.add_argument("--masking", type=str, choices=["rm", "mnr", "bm"], default=None,
                        help="Optional override for train_config.masking")
    parser.add_argument("--missing_k", type=int, default=None,
                        help="Optional override for train_config.missing_k")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_config = config["train_config"]
    trainset_config = config["trainset_config"]
    if args.output_directory is not None:
        train_config["output_directory"] = args.output_directory
    if args.ckpt_iter is not None:
        train_config["ckpt_iter"] = args.ckpt_iter
    if args.masking is not None:
        train_config["masking"] = args.masking
    if args.missing_k is not None:
        train_config["missing_k"] = args.missing_k

    diffusion_config = config["diffusion_config"]
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    if train_config["use_model"] == 0:
        model_config = config["wavenet_config"]
    elif train_config["use_model"] == 1:
        model_config = config["sashimi_config"]
    elif train_config["use_model"] == 2:
        model_config = config["wavenet_config"]
    else:
        raise ValueError(f"Unknown model index: {train_config['use_model']}")

    train(
        **train_config,
        data_path=trainset_config["train_data_path"],
        batch_size=args.batch_size,
    )