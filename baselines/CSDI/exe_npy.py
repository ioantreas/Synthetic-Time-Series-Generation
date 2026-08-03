import argparse
import random
import numpy as np

import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Physio
from dataset_npy import get_dataloader
from utils import train, evaluate


parser = argparse.ArgumentParser(description="CSDI generic NPY")

parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--train_data", type=str, required=True)
parser.add_argument("--test_data", type=str, required=True)

parser.add_argument("--missing_ratio", type=float, default=0.25)
parser.add_argument(
    "--mask_type",
    type=str,
    default="block",
    choices=["block", "random"],
)

parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)
parser.add_argument("--out_dir", type=str, default="./save_npy")
parser.add_argument("--num_examples", type=int, default=None, help="Randomly subsample this many sequences from train/test without replacement",)
parser.add_argument("--fixed_mask", type=str, default=None)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.missing_ratio

train_loader, valid_loader, test_loader, target_dim = get_dataloader(
    train_data=args.train_data,
    test_data=args.test_data,
    seed=args.seed,
    batch_size=config["train"]["batch_size"],
    missing_ratio=args.missing_ratio,
    mask_type=args.mask_type,
    fixed_mask=args.fixed_mask,

)

if args.num_examples is not None:
    import numpy as np
    from torch.utils.data import Subset, DataLoader

    rng = np.random.default_rng(args.seed)

    n = min(args.num_examples, len(test_loader.dataset))

    idx = rng.choice(
        len(test_loader.dataset),
        size=n,
        replace=False,
    )

    test_subset = Subset(
        test_loader.dataset,
        idx,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
    )

    print(f"Using {n} test examples")

print("Detected target_dim:", target_dim)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

foldername = os.path.join(
    args.out_dir,
    # f"csdi_{target_dim}features_{args.mask_type}_{args.missing_ratio}_{current_time}",
)

print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)

with open(os.path.join(foldername, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

with open(os.path.join(foldername, "args.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

model = CSDI_Physio(
    config,
    args.device,
    target_dim=target_dim,
).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(
        torch.load(
            os.path.join("./save", args.modelfolder, "model.pth"),
            map_location=args.device,
        )
    )

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=1,
    foldername=foldername,
)