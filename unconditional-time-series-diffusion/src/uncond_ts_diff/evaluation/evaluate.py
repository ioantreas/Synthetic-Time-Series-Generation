import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# =====================================
# Feature extractor (multivariate)
# =====================================

def extract_features(X):
    N, T, C = X.shape
    eps = 1e-8
    feats = []

    mean = X.mean(axis=1)
    std = X.std(axis=1)
    minv = X.min(axis=1)
    maxv = X.max(axis=1)

    feats += [mean, std, minv, maxv]

    lags = [1, 2, 6, 12, 24]
    acfs = []

    for lag in lags:
        if lag >= T:
            acfs.append(np.zeros((N, C)))
        else:
            x1 = X[:, lag:]
            x2 = X[:, :-lag]
            num = ((x1 - x1.mean(1, keepdims=True)) *
                   (x2 - x2.mean(1, keepdims=True))).mean(1)
            den = x1.std(1) * x2.std(1) + eps
            acfs.append(num / den)

    feats.append(np.concatenate(acfs, axis=1))

    ps = np.abs(np.fft.rfft(X, axis=1)) ** 2
    feats.append(ps.mean(axis=1))
    feats.append(ps.std(axis=1))

    corr_mean = np.zeros((N, 1))
    corr_abs = np.zeros((N, 1))

    for i in range(N):
        if C == 1:
            continue
        c = np.corrcoef(X[i], rowvar=False)
        c = np.nan_to_num(c)
        off = c[~np.eye(C, dtype=bool)]
        corr_mean[i] = off.mean()
        corr_abs[i] = np.abs(off).mean()

    feats += [corr_mean, corr_abs]

    return np.concatenate(feats, axis=1)


# =====================================
# MMD
# =====================================

def rbf_mmd(X, Y, gamma=None):
    if gamma is None:
        Z = np.vstack([X, Y])
        idx = np.random.choice(len(Z), size=min(1000, len(Z)), replace=False)
        Zs = Z[idx]
        d = ((Zs[:, None] - Zs[None]) ** 2).sum(-1)
        med = np.median(d[d > 0])
        gamma = 1.0 / (2 * med + 1e-8)

    def k(A, B):
        d = ((A[:, None] - B[None]) ** 2).sum(-1)
        return np.exp(-gamma * d)

    return float(np.sqrt(np.maximum(
        k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean(), 0
    )))


# =====================================
# Plots
# =====================================

def save_sequences(real, synth, outdir, n=3, channel=0):
    plt.figure(figsize=(12,4))
    for i in range(n):
        plt.plot(real[i,:,channel])
    plt.title(f"Real samples (channel {channel})")
    plt.savefig(outdir/"real_sequences.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12,4))
    for i in range(n):
        plt.plot(synth[i,:,channel])
    plt.title(f"Synthetic samples (channel {channel})")
    plt.savefig(outdir/"synthetic_sequences.png", dpi=150)
    plt.close()


def save_hist(real, synth, outdir):
    plt.figure(figsize=(8,4))
    plt.hist(real.flatten(), bins=100, density=True, alpha=0.5, label="Real")
    plt.hist(synth.flatten(), bins=100, density=True, alpha=0.5, label="Synth")
    plt.legend()
    plt.title("Value distribution")
    plt.savefig(outdir/"hist.png", dpi=150)
    plt.close()


def save_acf_plot(real, synth, outdir):
    r = acf(real.reshape(-1), nlags=60)
    s = acf(synth.reshape(-1), nlags=60)

    plt.figure(figsize=(8,4))
    plt.plot(r, label="Real")
    plt.plot(s, label="Synth")
    plt.legend()
    plt.title("Autocorrelation")
    plt.savefig(outdir/"acf.png", dpi=150)
    plt.close()

def save_plot_acf_multivariate(real, synth, outdir, nlags=60, max_feats=6):
    """
    real, synth: (N, T, C)
    plots first max_feats features
    """

    C = real.shape[-1]
    F = min(C, max_feats)

    plt.figure(figsize=(10, 6))

    for c in range(F):
        r = real[:, :, c].reshape(-1)
        s = synth[:, :, c].reshape(-1)

        acf_r = acf(r, nlags=nlags)
        acf_s = acf(s, nlags=nlags)

        plt.subplot(F, 1, c + 1)
        plt.plot(acf_r, label="Real")
        plt.plot(acf_s, label="Synth")
        plt.title(f"Feature {c}")
        if c == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig(outdir / "acf_multivariate.png", dpi=150)
    plt.close()


def save_fft(real, synth, outdir):
    rf = np.mean(np.abs(np.fft.rfft(real, axis=1))**2, axis=0)
    sf = np.mean(np.abs(np.fft.rfft(synth, axis=1))**2, axis=0)

    plt.figure(figsize=(8,4))
    plt.plot(rf, label="Real")
    plt.plot(sf, label="Synth")
    plt.legend()
    plt.title("Power spectrum")
    plt.savefig(outdir/"fft.png", dpi=150)
    plt.close()


import numpy as np

def safe_corr_matrix(batch, eps=1e-8):
    """
    batch: (N, T, C)
    Returns mean correlation matrix across windows, ignoring undefined entries.
    """
    N, T, C = batch.shape
    sumM = np.zeros((C, C), dtype=np.float64)
    cntM = np.zeros((C, C), dtype=np.float64)

    for seq in batch:
        std = seq.std(axis=0)
        mask = std > eps
        idx = np.where(mask)[0]
        if len(idx) < 2:
            # still count diag for any non-constant single channel if you want
            if len(idx) == 1:
                i = idx[0]
                sumM[i, i] += 1.0
                cntM[i, i] += 1.0
            continue

        c = np.corrcoef(seq[:, idx], rowvar=False)
        c = np.nan_to_num(c)

        # accumulate only for observed entries
        for a, ia in enumerate(idx):
            for b, ib in enumerate(idx):
                sumM[ia, ib] += c[a, b]
                cntM[ia, ib] += 1.0

    M = np.zeros((C, C), dtype=np.float64)
    mask = cntM > 0
    M[mask] = sumM[mask] / cntM[mask]

    # Optional: enforce diagonal = 1 where we ever observed variance
    diag_obs = np.diag(cntM) > 0
    M[np.arange(C), np.arange(C)] = np.where(diag_obs, 1.0, 0.0)

    return M

def save_corr_maps(real, synth, outdir):
    Cr = safe_corr_matrix(real)
    Cs = safe_corr_matrix(synth)

    vmax = 1.0

    for M, name in [(Cr,"real"), (Cs,"synth"), (Cs-Cr,"diff")]:
        plt.figure(figsize=(6,5))
        plt.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.title(f"Correlation {name}")
        plt.tight_layout()
        plt.savefig(outdir/f"corr_{name}.png", dpi=150)
        plt.close()

def save_full_timeseries(real_windows, synth_windows, outdir, channel=0):
    """
    Plot long real vs synthetic trajectories.

    real_windows: (N, T, C)
    synth_windows: (N, T, C)
    """

    # Reconstruct long sequences by concatenating windows
    real_long = real_windows.reshape(-1, real_windows.shape[-1])
    synth_long = synth_windows.reshape(-1, synth_windows.shape[-1])

    # Match lengths
    T = min(len(real_long), len(synth_long))

    real_long = real_long[:T]
    synth_long = synth_long[:T]

    plt.figure(figsize=(14,6))

    plt.subplot(2,1,1)
    plt.plot(real_long[:,channel])
    plt.title(f"Real long trajectory (channel {channel})")

    plt.subplot(2,1,2)
    plt.plot(synth_long[:,channel])
    plt.title(f"Synthetic long trajectory (channel {channel})")

    plt.tight_layout()
    plt.savefig(outdir/"long_timeseries.png", dpi=150)
    plt.close()

def save_long_timeseries_zoom(real, synth, outdir, channel=0, length=1000):
    real_long = real.reshape(-1, real.shape[-1])
    synth_long = synth.reshape(-1, synth.shape[-1])

    T = min(len(real_long), len(synth_long))

    start = np.random.randint(0, T - length)

    real_seg = real_long[start:start+length, channel]
    synth_seg = synth_long[start:start+length, channel]

    plt.figure(figsize=(12,5))

    plt.subplot(2,1,1)
    plt.plot(real_seg)
    plt.title("Real time series (zoomed segment)")

    plt.subplot(2,1,2)
    plt.plot(synth_seg)
    plt.title("Synthetic time series (zoomed segment)")

    plt.tight_layout()
    plt.savefig(outdir / "long_timeseries_zoom.png", dpi=150)
    plt.close()

def save_overlay_timeseries(real, synth, outdir, channel=0, length=1000):

    real_long = real.reshape(-1, real.shape[-1])
    synth_long = synth.reshape(-1, synth.shape[-1])

    T = min(len(real_long), len(synth_long))
    start = np.random.randint(0, T - length)

    real_seg = real_long[start:start+length, channel]
    synth_seg = synth_long[start:start+length, channel]

    plt.figure(figsize=(12,4))
    plt.plot(real_seg, label="Real", alpha=0.8)
    plt.plot(synth_seg, label="Synthetic", alpha=0.8)

    plt.legend()
    plt.title("Real vs Synthetic time series segment")
    plt.tight_layout()
    plt.savefig(outdir / "overlay_timeseries.png", dpi=150)
    plt.close()

def save_multichannel_timeseries_zoom(real, synth, outdir, channels=(0,1,2), length=1000):
    """
    Plot zoomed long trajectories for multiple channels.
    """

    real_long = real.reshape(-1, real.shape[-1])
    synth_long = synth.reshape(-1, synth.shape[-1])

    T = min(len(real_long), len(synth_long))
    start = np.random.randint(0, T - length)

    real_seg = real_long[start:start+length]
    synth_seg = synth_long[start:start+length]

    num_channels = len(channels)

    plt.figure(figsize=(12, 3*num_channels*2))

    # Real
    for i, c in enumerate(channels):
        plt.subplot(num_channels*2, 1, i+1)
        plt.plot(real_seg[:, c])
        plt.title(f"Real channel {c}")

    # Synthetic
    for i, c in enumerate(channels):
        plt.subplot(num_channels*2, 1, num_channels + i + 1)
        plt.plot(synth_seg[:, c])
        plt.title(f"Synthetic channel {c}")

    plt.tight_layout()
    plt.savefig(outdir / "multichannel_timeseries_zoom.png", dpi=150)
    plt.close()

# =====================================
# Main
# =====================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)

    real = np.load(args.dataset)

    synth_path = (
            Path("../../../results/lightning_logs")
            / f"version_{args.version}"
            / "checkpoints"
            / "synthetic_samples.npy"
    )

    synth = np.load(synth_path)

    if synth.ndim == 4:
        synth = synth[...,0]

    n = min(len(real), len(synth), 1500)

    print(f"\n\n\n\n {n} \n\n\n\n")

    idx_real = np.random.choice(len(real), size=n, replace=False)
    real = real[idx_real]

    idx_synth = np.random.choice(len(synth), size=n, replace=False)
    synth = synth[idx_synth]

    feats_r = extract_features(real)
    feats_s = extract_features(synth)

    X = np.vstack([feats_r, feats_s])
    y = np.hstack([np.zeros(len(feats_r)), np.ones(len(feats_s))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])

    mmd = rbf_mmd(feats_r, feats_s)

    outdir = synth_path.parent / "eval"
    outdir.mkdir(exist_ok=True)

    metrics = {
        "auc": float(auc),
        "mmd": mmd,
        "real_std": float(real.std()),
        "synth_std": float(synth.std()),
        "n": int(n),
    }

    # -----------------------------
    # Per-channel AUC
    # -----------------------------
    auc_per_channel = []

    for c in range(real.shape[-1]):
        feats_r_c = extract_features(real[:,:,c:c+1])
        feats_s_c = extract_features(synth[:,:,c:c+1])

        Xc = np.vstack([feats_r_c, feats_s_c])
        yc = np.hstack([np.zeros(len(feats_r_c)), np.ones(len(feats_s_c))])

        Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
            Xc, yc, test_size=0.3, stratify=yc
        )

        sc_c = StandardScaler()
        Xtr_c = sc_c.fit_transform(Xtr_c)
        Xte_c = sc_c.transform(Xte_c)

        clf_c = LogisticRegression(max_iter=2000)
        clf_c.fit(Xtr_c, ytr_c)

        auc_c = roc_auc_score(yte_c, clf_c.predict_proba(Xte_c)[:,1])
        auc_per_channel.append(auc_c)

    metrics["auc_per_channel_min"] = float(np.min(auc_per_channel))
    metrics["auc_per_channel_mean"] = float(np.mean(auc_per_channel))
    metrics["auc_per_channel_max"] = float(np.max(auc_per_channel))

    print("Real mean std per channel:", real.std(axis=(0,1)))
    print("Synth mean std per channel:", synth.std(axis=(0,1)))
    Cr = safe_corr_matrix(real)
    Cs = safe_corr_matrix(synth)
    print("Correlation difference magnitude:", np.mean(np.abs(Cr - Cs)))

    with open(outdir/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    save_sequences(real, synth, outdir, channel=args.channel)
    save_hist(real, synth, outdir)
    save_acf_plot(real, synth, outdir)
    save_plot_acf_multivariate(real, synth, outdir)
    save_fft(real, synth, outdir)
    save_corr_maps(real, synth, outdir)
    save_multichannel_timeseries_zoom(real, synth, outdir, channels=(0,1,2))
    # save_full_timeseries(real, synth, outdir, channel=args.channel)
    # save_long_timeseries_zoom(real, synth, outdir, channel=args.channel)
    # save_overlay_timeseries(real, synth, outdir, channel=args.channel)


    print(json.dumps(metrics, indent=2))
    print("Saved to:", outdir)


if __name__ == "__main__":
    main()