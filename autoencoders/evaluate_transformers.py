import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
from statsmodels.tsa.stattools import acf
from torch.utils.data import (
    DataLoader,
    TensorDataset
)

# ============================================================
# Positional Encoding
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=4096):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# ============================================================
# Transformer Autoencoder
# ============================================================

class TransformerTimeAE(nn.Module):

    def __init__(
            self,
            channels,
            seq_len,
            latent_steps=16,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=256,
            dropout=0.1
    ):

        super().__init__()

        self.channels = channels
        self.seq_len = seq_len
        self.latent_steps = latent_steps
        self.d_model = d_model

        # ------------------------------------------------

        self.input_proj = nn.Linear(
            channels,
            d_model
        )

        self.pos_enc = PositionalEncoding(
            d_model,
            max_len=seq_len
        )

        # ------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ------------------------------------------------

        self.latent_queries = nn.Parameter(
            torch.randn(latent_steps, d_model)
        )

        self.cross_attn_enc = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True
        )

        # ------------------------------------------------

        self.cross_attn_dec = nn.MultiheadAttention(
            d_model,
            nhead,
            batch_first=True
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(
            d_model,
            channels
        )

    # ====================================================

    def encode(self, x):

        # (B,C,L) -> (B,L,C)

        x = x.permute(0, 2, 1)

        h = self.input_proj(x)

        h = self.pos_enc(h)

        h = self.encoder(h)

        B = h.shape[0]

        queries = self.latent_queries.unsqueeze(0).expand(
            B,
            -1,
            -1
        )

        z, _ = self.cross_attn_enc(
            queries,
            h,
            h
        )

        return z

    # ====================================================

    def decode(self, z):

        B = z.shape[0]

        seq_queries = torch.zeros(
            B,
            self.seq_len,
            self.d_model,
            device=z.device
        )

        seq_queries = self.pos_enc(seq_queries)

        h, _ = self.cross_attn_dec(
            seq_queries,
            z,
            z
        )

        h = self.decoder(h)

        xhat = self.output_proj(h)

        # (B,L,C) -> (B,C,L)

        xhat = xhat.permute(0, 2, 1)

        return xhat

    # ====================================================

    def forward(self, x):

        z = self.encode(x)

        xhat = self.decode(z)

        return xhat, z

# -------------------------------------------------
# Encode -> Decode
# -------------------------------------------------

def encode_decode(model, data, device, batch_size=64):

    dataset = TensorDataset(
        torch.from_numpy(data).float()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size
    )

    xs = []
    xhats = []
    latents = []

    model.eval()

    with torch.no_grad():

        for (x,) in loader:

            x = x.to(device)

            # (B,L,C) -> (B,C,L)
            x_in = x.permute(0, 2, 1)

            xhat, z = model(x_in)

            xs.append(
                x.cpu().numpy()
            )

            xhats.append(
                xhat.permute(0, 2, 1).cpu().numpy()
            )

            latents.append(
                z.cpu().numpy()
            )

    xs = np.concatenate(xs)

    xhats = np.concatenate(xhats)

    latents = np.concatenate(latents)

    return xs, xhats, latents


# -------------------------------------------------
# Generate from train latents
# -------------------------------------------------

def generate_from_train_latents(
        model,
        train_data,
        num_samples,
        device,
        batch_size=64
):

    dataset = TensorDataset(
        torch.from_numpy(train_data).float()
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size
    )

    latents = []

    model.eval()

    with torch.no_grad():

        for (x,) in loader:

            x = x.to(device)

            x_in = x.permute(0, 2, 1)

            _, z = model(x_in)

            latents.append(
                z.cpu().numpy()
            )

    latents = np.concatenate(latents)

    print("Train latent pool:", latents.shape)

    idx = np.random.choice(
        len(latents),
        num_samples,
        replace=True
    )

    z_sample = latents[idx]

    z_sample = torch.from_numpy(
        z_sample
    ).float().to(device)

    with torch.no_grad():

        xhat = model.decode(z_sample)

    xhat = xhat.permute(0, 2, 1).cpu().numpy()

    return xhat


# -------------------------------------------------
# Random real window sampling
# -------------------------------------------------

def sample_random_windows(data, num_samples):

    idx = np.random.choice(
        len(data),
        num_samples,
        replace=False
    )

    return data[idx]


# -------------------------------------------------
# Metrics
# -------------------------------------------------

def mse(real, synth):

    return ((real - synth) ** 2).mean()


def mse_per_channel(real, synth):

    return ((real - synth) ** 2).mean(axis=(0,1))


def acf_error(real, synth, nlags=40):

    errs = []

    C = real.shape[-1]

    for c in range(C):

        rseries = real[:,:,c].reshape(-1)

        sseries = synth[:,:,c].reshape(-1)

        if np.std(rseries) < 1e-8:
            continue

        if np.std(sseries) < 1e-8:
            continue

        r = acf(rseries, nlags=nlags)

        s = acf(sseries, nlags=nlags)

        errs.append(
            np.mean(np.abs(r - s))
        )

    if len(errs) == 0:
        return 0.0

    return np.mean(errs)


def corr_matrix(batch):

    X = batch.reshape(
        -1,
        batch.shape[-1]
    )

    std = np.std(X, axis=0)

    valid = std > 1e-8

    if np.sum(valid) < 2:
        return np.zeros((1,1))

    X = X[:, valid]

    return np.corrcoef(
        X,
        rowvar=False
    )


def corr_difference(real, synth):

    Cr = corr_matrix(real)

    Cs = corr_matrix(synth)

    m = min(
        Cr.shape[0],
        Cs.shape[0]
    )

    Cr = Cr[:m,:m]

    Cs = Cs[:m,:m]

    return np.mean(np.abs(Cr - Cs))


def fft_difference(real, synth):

    rf = np.mean(
        np.abs(
            np.fft.rfft(real, axis=1)
        ) ** 2,
        axis=(0,2)
    )

    sf = np.mean(
        np.abs(
            np.fft.rfft(synth, axis=1)
        ) ** 2,
        axis=(0,2)
    )

    return np.mean(np.abs(rf - sf))


def histogram_distance(real, synth, bins=100):

    r = real.flatten()

    s = synth.flatten()

    hr,_ = np.histogram(
        r,
        bins=bins,
        density=True
    )

    hs,_ = np.histogram(
        s,
        bins=bins,
        density=True
    )

    return np.mean(np.abs(hr - hs))


# -------------------------------------------------
# Plots
# -------------------------------------------------

def plot_sequences(real, synth, outdir, channel=0):

    if channel >= real.shape[-1]:

        print(
            f"Skipping sequence plot for channel {channel}"
        )

        return

    plt.figure(figsize=(10,4))

    plt.plot(
        real[0,:,channel],
        label="real"
    )

    plt.plot(
        synth[0,:,channel],
        label="generated"
    )

    plt.legend()

    plt.title(
        f"Sequence comparison (channel {channel})"
    )

    plt.savefig(
        outdir / f"sequence_{channel}.png"
    )

    plt.close()


def plot_fft(real, synth, outdir):

    rf = np.mean(
        np.abs(
            np.fft.rfft(real,axis=1)
        ) ** 2,
        axis=(0,2)
    )

    sf = np.mean(
        np.abs(
            np.fft.rfft(synth,axis=1)
        ) ** 2,
        axis=(0,2)
    )

    plt.figure(figsize=(8,4))

    plt.plot(rf,label="real")

    plt.plot(sf,label="generated")

    plt.legend()

    plt.title("Average power spectrum")

    plt.savefig(outdir/"fft.png")

    plt.close()


def plot_histogram(real, synth, outdir):

    plt.figure(figsize=(8,4))

    plt.hist(
        real.flatten(),
        bins=100,
        density=True,
        alpha=0.5,
        label="real"
    )

    plt.hist(
        synth.flatten(),
        bins=100,
        density=True,
        alpha=0.5,
        label="generated"
    )

    plt.legend()

    plt.title("Value distribution")

    plt.savefig(outdir/"histogram.png")

    plt.close()


def plot_corr(real, synth, outdir):

    Cr = corr_matrix(real)

    Cs = corr_matrix(synth)

    for M,name in [

        (Cr,"real"),

        (Cs,"generated"),

        (Cs-Cr,"diff")

    ]:

        plt.figure(figsize=(6,5))

        plt.imshow(
            M,
            cmap="coolwarm",
            vmin=-1,
            vmax=1
        )

        plt.colorbar()

        plt.title(name)

        plt.savefig(
            outdir/f"corr_{name}.png"
        )

        plt.close()


# -------------------------------------------------
# Evaluation
# -------------------------------------------------

def evaluate(real, synth, outdir):

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    total_mse = mse(real,synth)

    per_channel = mse_per_channel(real,synth)

    acf_err = acf_error(real,synth)

    corr_err = corr_difference(real,synth)

    fft_err = fft_difference(real,synth)

    hist_err = histogram_distance(real,synth)

    metrics = {

        "reconstruction_mse":
            float(total_mse),

        "acf_difference":
            float(acf_err),

        "correlation_difference":
            float(corr_err),

        "fft_difference":
            float(fft_err),

        "histogram_distance":
            float(hist_err),

        "mse_per_channel":
            per_channel.tolist(),

        "num_samples":
            int(real.shape[0])
    }

    with open(outdir/"metrics.json","w") as f:

        json.dump(metrics,f,indent=2)

    print("\n===== METRICS =====")

    for k,v in metrics.items():

        if k != "mse_per_channel":

            print(k,":",v)

    for c in range(min(real.shape[-1], 5)):

        plot_sequences(
            real,
            synth,
            outdir,
            c
        )

    plot_fft(real,synth,outdir)

    plot_histogram(real,synth,outdir)

    plot_corr(real,synth,outdir)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_dataset",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--latent_steps",
        type=int,
        default=16
    )

    parser.add_argument(
        "--d_model",
        type=int,
        default=128
    )

    parser.add_argument(
        "--nhead",
        type=int,
        default=8
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--feature_idx",
        type=int,
        nargs="+",
        default=None,
        help="Only evaluate selected feature indices"
    )

    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # -------------------------------------------------

    train = np.load(
        args.train_dataset
    ).astype(np.float32)

    test = np.load(
        args.test_dataset
    ).astype(np.float32)

    if args.feature_idx is not None:

        print(
            "Using feature indices:",
            args.feature_idx
        )

        train = train[:,:,args.feature_idx]

        test = test[:,:,args.feature_idx]

    print("Train:",train.shape)

    print("Test:",test.shape)

    N,L,C = test.shape

    # -------------------------------------------------

    model = TransformerTimeAE(

        channels=C,

        seq_len=L,

        latent_steps=args.latent_steps,

        d_model=args.d_model,

        nhead=args.nhead,

        num_layers=args.num_layers,

        # dim_feedforward=args.d_model * 2
    )

    model.load_state_dict(
        torch.load(
            args.model,
            map_location=device
        )
    )

    model.to(device)

    # -------------------------------------------------

    out_root = Path(args.output)

    out_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # test -> test
    # -------------------------------------------------

    print(
        "\nRunning test -> test reconstruction"
    )

    real, recon, _ = encode_decode(

        model,

        test[:args.num_samples],

        device,

        args.batch_size
    )

    evaluate(
        real,
        recon,
        out_root/"test_to_test"
    )

    # -------------------------------------------------
    # train -> test
    # -------------------------------------------------

    print(
        "\nRunning train -> test baseline"
    )

    synth = generate_from_train_latents(

        model,

        train,

        args.num_samples,

        device
    )

    real = sample_random_windows(
        test,
        args.num_samples
    )

    evaluate(
        real,
        synth,
        out_root/"train_to_test"
    )

    # -------------------------------------------------
    # train -> train
    # -------------------------------------------------

    print(
        "\nRunning train -> train baseline"
    )

    synth = generate_from_train_latents(

        model,

        train,

        args.num_samples,

        device
    )

    real = sample_random_windows(
        train,
        args.num_samples
    )

    evaluate(
        real,
        synth,
        out_root/"train_to_train"
    )

    print("\nFinished")


if __name__ == "__main__":

    main()