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
    plt.figure(figsize=(12, 4))
    for i in range(min(n, len(real))):
        plt.plot(real[i, :, channel])
    plt.title(f"Real samples (channel {channel})")
    plt.savefig(outdir / "real_sequences.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 4))
    for i in range(min(n, len(synth))):
        plt.plot(synth[i, :, channel])
    plt.title(f"Synthetic samples (channel {channel})")
    plt.savefig(outdir / "synthetic_sequences.png", dpi=150)
    plt.close()


def save_hist(real, synth, outdir):
    plt.figure(figsize=(8, 4))
    plt.hist(real.flatten(), bins=100, density=True, alpha=0.5, label="Real")
    plt.hist(synth.flatten(), bins=100, density=True, alpha=0.5, label="Synth")
    plt.legend()
    plt.title("Value distribution")
    plt.savefig(outdir / "hist.png", dpi=150)
    plt.close()


def save_acf_plot(real, synth, outdir):
    r = acf(real.reshape(-1), nlags=60)
    s = acf(synth.reshape(-1), nlags=60)

    plt.figure(figsize=(8, 4))
    plt.plot(r, label="Real")
    plt.plot(s, label="Synth")
    plt.legend()
    plt.title("Autocorrelation")
    plt.savefig(outdir / "acf.png", dpi=150)
    plt.close()


def save_plot_acf_multivariate(real, synth, outdir, nlags=60, max_feats=6):
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
    rf = np.mean(np.abs(np.fft.rfft(real, axis=1)) ** 2, axis=0)
    sf = np.mean(np.abs(np.fft.rfft(synth, axis=1)) ** 2, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(rf, label="Real")
    plt.plot(sf, label="Synth")
    plt.legend()
    plt.title("Power spectrum")
    plt.savefig(outdir / "fft.png", dpi=150)
    plt.close()


def save_corr_maps(real, synth, outdir):
    Cr = np.mean([np.corrcoef(r, rowvar=False) for r in real], axis=0)
    Cs = np.mean([np.corrcoef(s, rowvar=False) for s in synth], axis=0)

    vmax = max(abs(Cr).max(), abs(Cs).max())

    for M, name in [(Cr, "real"), (Cs, "synth"), (Cs - Cr, "diff")]:
        plt.figure(figsize=(6, 5))
        plt.imshow(M, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        plt.colorbar()
        plt.title(f"Correlation {name}")
        plt.tight_layout()
        plt.savefig(outdir / f"corr_{name}.png", dpi=150)
        plt.close()


# =====================================
# Helpers
# =====================================

def _load_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    # allow (N,T,C,1) -> (N,T,C)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"{path} must have shape (N,T,C) (or (N,T,C,1)), got {arr.shape}")
    return arr


# =====================================
# Main
# =====================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real", required=True, help="Path to real dataset .npy (N,T,C)")
    p.add_argument("--synth", required=True, help="Path to synthetic samples .npy (N,T,C) or (N,T,C,1)")
    p.add_argument("--outdir", default=None, help="Output directory for eval. Default: <synth_parent>/eval")
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)

    real_path = Path(args.real)
    synth_path = Path(args.synth)

    real = _load_npy(real_path)
    synth = _load_npy(synth_path)

    n = min(len(real), len(synth))
    real = real[:n]
    synth = synth[:n]

    feats_r = extract_features(real)
    feats_s = extract_features(synth)

    X = np.vstack([feats_r, feats_s])
    y = np.hstack([np.zeros(len(feats_r)), np.ones(len(feats_s))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=args.seed)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

    mmd = rbf_mmd(feats_r, feats_s)

    outdir = Path(args.outdir) if args.outdir is not None else (synth_path.parent / "eval")
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "auc": float(auc),
        "mmd": float(mmd),
        "real_std": float(real.std()),
        "synth_std": float(synth.std()),
        "n": int(n),
        "real_path": str(real_path),
        "synth_path": str(synth_path),
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_sequences(real, synth, outdir, channel=args.channel)
    save_hist(real, synth, outdir)
    save_acf_plot(real, synth, outdir)
    save_plot_acf_multivariate(real, synth, outdir)
    save_fft(real, synth, outdir)
    save_corr_maps(real, synth, outdir)

    print(json.dumps(metrics, indent=2))
    print("Saved to:", outdir)


if __name__ == "__main__":
    main()