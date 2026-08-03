import os
import torch
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--config_file', type=str, default=None, 
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)

    parser.add_argument("--data", type=str)
    parser.add_argument("--mask", type=str)

    parser.add_argument("--num_imputation_samples", type=int, default=10)

    parser.add_argument(
        "--eval_scenario",
        type=str,
        required=True,
        choices=["single_block", "forecast"],
        help="Evaluation scenario used to organise generated results.",
    )


    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, args.name)

    # Store evaluation results separately.
    args.result_dir = os.path.join(
        args.save_dir,
        args.eval_scenario,
        f"seed_{args.seed}",
    )

    return args

def quantile_loss(target, forecast, q):
    return 2 * np.sum(np.abs((forecast - target) * ((target <= forecast).astype(np.float32) - q)))

def compute_crps(all_preds, x_true, mask):
    missing = mask == 0
    quantiles = np.arange(0.05, 1.0, 0.05)
    target = x_true[missing]
    denom = np.sum(np.abs(target)) + 1e-8

    crps = 0.0
    for q in quantiles:
        q_pred = np.quantile(all_preds, q, axis=1)
        forecast = q_pred[missing]
        crps += quantile_loss(target, forecast, q) / denom

    return float(crps / len(quantiles))

def compute_metrics(all_preds, x_true, mask):
    x_pred = np.median(all_preds, axis=1)
    missing = mask == 0

    metrics = {
        "mse_missing": float(((x_pred - x_true)[missing] ** 2).mean()),
        "crps_missing": compute_crps(all_preds, x_true, mask),
    }

    return metrics, x_pred

def plot_example(x_true, x_pred, mask, out_dir, sample_idx=0, channel_idx=0):
    t = np.arange(x_true.shape[1])
    plt.figure(figsize=(10, 4))
    plt.plot(t, x_true[sample_idx, :, channel_idx], label="true")
    plt.plot(t, x_pred[sample_idx, :, channel_idx], label="estimate")
    plt.scatter(t[mask[sample_idx, :, channel_idx] == 1], x_true[sample_idx, mask[sample_idx, :, channel_idx] == 1, channel_idx], s=10, label="observed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "example_imputation.png"), dpi=200)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    
    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        test_dataloader_info = build_dataloader_cond(config, args)
    dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']
        all_samples = []

        for k in range(args.num_imputation_samples):
            print(f"Imputation {k + 1}/{args.num_imputation_samples}")

            samples, reals, masks = trainer.restore(
                dataloader,
                [dataset.window, dataset.var_num],
                coef,
                stepsize,
                sampling_steps,
            )

            if dataset.auto_norm:
                samples = unnormalize_to_zero_to_one(samples)

            all_samples.append(samples)

        all_samples = np.stack(all_samples, axis=1)
        median_pred = np.median(all_samples, axis=1)

        metrics, median_pred = compute_metrics(all_samples, reals, masks)

        np.save(os.path.join(args.result_dir, "all_preds.npy"), all_samples)
        np.save(os.path.join(args.result_dir, "median_pred.npy"), median_pred)
        np.save(os.path.join(args.result_dir, "data.npy"), reals)
        np.save(os.path.join(args.result_dir, "mask.npy"), masks)

        with open(os.path.join(args.result_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        plot_example(reals, median_pred, masks, args.result_dir)

        print(metrics)
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        samples = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)

if __name__ == '__main__':
    main()
