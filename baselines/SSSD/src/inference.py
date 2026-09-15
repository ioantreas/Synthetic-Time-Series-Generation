import os
import argparse
import json
import numpy as np
import time
import torch
import random

from utils.util import find_max_epoch, print_size, sampling, calc_diffusion_hyperparams
from imputers.DiffWaveImputer import DiffWaveImputer
from imputers.SSSDSAImputer import SSSDSAImputer
from imputers.SSSDS4Imputer import SSSDS4Imputer


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def quantile_loss(target, forecast, q, eval_points):
    return 2.0 * np.sum(
        np.abs((forecast - target) * eval_points * ((target <= forecast) - q))
    )


def compute_crps(target, samples, eval_points):
    """
    Normalized CRPS used in CSDI-style time-series imputation evaluation.

    target:      (N, T, C)
    samples:     (N, S, T, C)
    eval_points: (N, T, C), 1 only at missing/evaluated entries
    """
    denom = np.sum(np.abs(target * eval_points))
    if denom == 0:
        return float("nan")

    crps = 0.0
    quantiles = np.arange(0.05, 1.0, 0.05)
    for q in quantiles:
        forecast = np.quantile(samples, q, axis=1)
        crps += quantile_loss(target, forecast, q, eval_points)

    return float(crps / len(quantiles) / denom)


def generate(output_directory, ckpt_path, ckpt_iter,
             use_model, only_generate_missing, data_path,
             mask_path, num_samples, batch_size, seed):

    data = np.load(data_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.float32)

    if data.ndim != 3:
        raise ValueError(f"Expected data shape (N, T, C), got {data.shape}")
    if mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match data shape {data.shape}")
    if not np.all(np.isin(mask, [0, 1])):
        raise ValueError("Mask must contain only 0 and 1, with 1=observed and 0=missing")

    num_windows, sequence_length, num_channels = data.shape
    print(f"Loaded test data: {data.shape}", flush=True)

    model_config["in_channels"] = num_channels
    model_config["out_channels"] = num_channels
    if use_model == 2:
        model_config["s4_lmax"] = sequence_length

    local_path = "T{}_beta0{}_betaT{}".format(
        diffusion_config["T"],
        diffusion_config["beta_0"],
        diffusion_config["beta_T"],
    )

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

    # Accept either the experiment root or the exact T... checkpoint directory.
    candidate_path = os.path.join(ckpt_path, local_path)
    if os.path.isdir(candidate_path):
        resolved_ckpt_path = candidate_path
    else:
        resolved_ckpt_path = ckpt_path

    if ckpt_iter == "max":
        ckpt_iter = find_max_epoch(resolved_ckpt_path)
    else:
        ckpt_iter = int(ckpt_iter)

    model_path = os.path.join(resolved_ckpt_path, f"{ckpt_iter}.pkl")
    checkpoint = torch.load(model_path, map_location="cpu")
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    print(f"Successfully loaded model at iteration {ckpt_iter}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_torch = torch.from_numpy(data).float().cuda().permute(0, 2, 1)
    mask_torch = torch.from_numpy(mask).float().cuda().permute(0, 2, 1)

    all_batch_predictions = []

    sync_cuda()
    inference_start = time.perf_counter()

    for start in range(0, num_windows, batch_size):
        end = min(start + batch_size, num_windows)

        batch = data_torch[start:end]
        batch_mask = mask_torch[start:end]

        stochastic_samples = []

        for sample_idx in range(num_samples):
            generated = sampling(
                net,
                batch.shape,
                diffusion_hyperparams,
                cond=batch,
                mask=batch_mask,
                only_generate_missing=only_generate_missing,
            )

            # Preserve observed values exactly.
            generated = generated * (1.0 - batch_mask) + batch * batch_mask

            # Keep the device-to-CPU transfer inside the timed region,
            # matching the timing protocol used by our method.
            stochastic_samples.append(generated.detach().cpu().numpy())

            print(
                f"batch {start}:{end}, sample {sample_idx + 1}/{num_samples}",
                flush=True,
            )

        # Do not stack here: CPU post-processing is excluded from timing.
        all_batch_predictions.append(stochastic_samples)

    sync_cuda()
    inference_time = time.perf_counter() - inference_start

    timing = {
        "inference_time_sec": inference_time,
        "inference_time_per_imputation_sec": inference_time / num_samples,
        "total_method_time_sec": inference_time,
    }

    # CPU-only aggregation is performed after timing.
    stacked_batches = []

    for stochastic_samples in all_batch_predictions:
        # list of S arrays [B,C,T] -> [B,S,C,T]
        stochastic_samples = np.stack(stochastic_samples, axis=1)

        # Common evaluation layout: [B,S,T,C]
        stochastic_samples = stochastic_samples.transpose(0, 1, 3, 2)

        stacked_batches.append(stochastic_samples)

    all_preds = np.concatenate(stacked_batches, axis=0)
    median_pred = np.median(all_preds, axis=1)

    eval_points = 1.0 - mask
    squared_error = ((median_pred - data) ** 2) * eval_points
    missing_count = np.sum(eval_points)
    mse = float(np.sum(squared_error) / missing_count) if missing_count > 0 else float("nan")
    crps = compute_crps(data, all_preds, eval_points)

    np.save(os.path.join(output_directory, "all_preds.npy"), all_preds)
    np.save(os.path.join(output_directory, "median_pred.npy"), median_pred)
    np.save(os.path.join(output_directory, "data.npy"), data)
    np.save(os.path.join(output_directory, "mask.npy"), mask)

    metrics = {
        "mse": mse,
        "crps": crps,
        "num_samples": int(num_samples),
        "prediction_shape": list(all_preds.shape),
        "inference_time_sec": timing["inference_time_sec"],
        "inference_time_per_imputation_sec": timing["inference_time_per_imputation_sec"],
        "total_method_time_sec": timing["total_method_time_sec"],
    }
    with open(os.path.join(output_directory, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config/config_SSSDS4.json")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint root or exact T... checkpoint directory")
    parser.add_argument("--ckpt_iter", default="max")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--output_directory", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train_config = config["train_config"]
    trainset_config = config["trainset_config"]
    diffusion_config = config["diffusion_config"]
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)

    data_path = args.data_path if args.data_path is not None else trainset_config["test_data_path"]
    mask_path = args.mask_path if args.mask_path is not None else trainset_config["test_mask_path"]
    output_directory = args.output_directory if args.output_directory is not None else train_config["output_directory"]

    if train_config["use_model"] == 0:
        model_config = config["wavenet_config"]
    elif train_config["use_model"] == 1:
        model_config = config["sashimi_config"]
    elif train_config["use_model"] == 2:
        model_config = config["wavenet_config"]
    else:
        raise ValueError(f"Unknown model index: {train_config['use_model']}")

    generate(output_directory=output_directory, ckpt_path=args.checkpoint, ckpt_iter=args.ckpt_iter,
             use_model=train_config["use_model"], only_generate_missing=train_config["only_generate_missing"],
             data_path=data_path, mask_path=mask_path, num_samples=args.num_samples, batch_size=args.batch_size,
             seed=args.seed)