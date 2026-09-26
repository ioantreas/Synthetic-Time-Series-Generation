import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


SCENARIOS = ["single_block", "blackout", "forecast", "random"]

SCENARIO_LABELS = {
    "single_block": "Block",
    "blackout": "Blackout",
    "forecast": "Forecast",
    "random": "Random",
}

DATASET_LABELS = {
    "appliances": "Appliances",
    "air_quality": "Air Quality",
    "har": "HAR",
    "metro": "Metro",
    "tep": "TEP",
}

# ============================================================
# FIXED SELECTIONS
# ============================================================

MAIN_SELECTION = {
    "dataset": "appliances",
    "sample": 121,
    "channel": 13,
}

APPENDIX_SELECTIONS = {
    "appliances":  {"sample": 121, "channel": 13},
    "air_quality": {"sample": 14,  "channel": 7},
    "har":         {"sample": 264, "channel": 0},
    "metro":       {"sample": 27,  "channel": 1},
    "tep":         {"sample": 133, "channel": 17},
}


# ============================================================
# LOADING
# ============================================================

def load_scenario(root, dataset, scenario, seed=1, guidance="variance"):
    d = root / dataset / f"seed_{seed}" / guidance / scenario

    truth_path = d / "test_data.npy"
    mask_path = d / "mask.npy"
    pred_path = d / "all_predictions.npy"

    for p in [truth_path, mask_path, pred_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    truth = np.load(truth_path)   # [B,T,C]
    mask = np.load(mask_path)     # [B,T,C]
    preds = np.load(pred_path)    # [B,K,T,C]

    return truth, mask, preds


def get_plot_data(root, dataset, scenario, sample_idx, channel_idx, seed=1):
    truth, mask, preds = load_scenario(
        root,
        dataset,
        scenario,
        seed=seed,
    )

    y_true = truth[sample_idx, :, channel_idx]
    y_mask = mask[sample_idx, :, channel_idx].astype(bool)

    # K stochastic imputations, each of length T
    y_preds = preds[sample_idx, :, :, channel_idx]

    median = np.median(y_preds, axis=0)
    q10 = np.quantile(y_preds, 0.10, axis=0)
    q90 = np.quantile(y_preds, 0.90, axis=0)

    return y_true, y_mask, median, q10, q90


# ============================================================
# PLOTTING HELPERS
# ============================================================

def shade_missing(ax, mask):
    missing = ~mask

    padded = np.pad(
        missing.astype(np.int8),
        (1, 1),
        constant_values=0,
    )

    changes = np.diff(padded)

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    for start, end in zip(starts, ends):
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color="red",
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )


def plot_one(
    ax,
    root,
    dataset,
    scenario,
    sample_idx,
    channel_idx,
    seed=1,
    title=None,
):
    truth, mask, median, q10, q90 = get_plot_data(
        root,
        dataset,
        scenario,
        sample_idx,
        channel_idx,
        seed=seed,
    )

    t = np.arange(len(truth))

    # Missing region
    shade_missing(ax, mask)

    # Predictive interval
    band = ax.fill_between(
        t,
        q10,
        q90,
        color="tab:blue",
        alpha=0.18,
        linewidth=0,
        label="10–90% interval",
        zorder=1,
    )

    # Ground truth
    gt_line, = ax.plot(
        t,
        truth,
        linestyle="--",
        linewidth=1.6,
        color="black",
        label="Ground truth",
        zorder=3,
    )

    # VIGIL prediction
    vigil_line, = ax.plot(
        t,
        median,
        linewidth=1.8,
        color="tab:blue",
        label="VIGIL median",
        zorder=4,
    )

    # Dummy patch purely for legend
    missing_patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="red",
        alpha=0.10,
        edgecolor="none",
        label="Missing region",
    )

    if title is not None:
        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
        )

    ax.grid(
        True,
        alpha=0.15,
        linewidth=0.6,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.tick_params(
        axis="both",
        labelsize=11,
    )

    # Bold tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    return [
        gt_line,
        vigil_line,
        band,
        missing_patch,
    ]


# ============================================================
# MAIN FIGURE
# ============================================================

def make_main_figure(root, output_dir, seed=1):
    dataset = MAIN_SELECTION["dataset"]
    sample_idx = MAIN_SELECTION["sample"]
    channel_idx = MAIN_SELECTION["channel"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.8),
    )

    handles = plot_one(
        axes[0],
        root,
        dataset,
        "single_block",
        sample_idx,
        channel_idx,
        seed=seed,
        title=None,
    )

    plot_one(
        axes[1],
        root,
        dataset,
        "forecast",
        sample_idx,
        channel_idx,
        seed=seed,
        title=None,
    )

    # Actual subplot area
    left = 0.15
    right = 0.97
    plot_center = (left + right) / 2

    fig.subplots_adjust(
        left=left,
        right=right,
        top=0.95,
        bottom=0.23,
        hspace=0.27,
    )

    # Shared y-axis label
    fig.supylabel(
        "Standardized value",
        fontsize=13,
        fontweight="bold",
        x=0.035,
    )

    # Centered relative to subplot area
    fig.supxlabel(
        "Time",
        fontsize=13,
        fontweight="bold",
        x=plot_center,
        y=0.125,
    )

    # Legend centered relative to subplot area
    legend = fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(plot_center, 0.025),
        frameon=False,
        fontsize=12,
        ncol=4,
        columnspacing=1.4,
        handlelength=2.0,
        handletextpad=0.6,
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.savefig(
        output_dir / "qualitative_main.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "qualitative_main.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# APPENDIX FIGURE
# ============================================================

def make_appendix_figure(root, output_dir, seed=1):
    datasets = list(APPENDIX_SELECTIONS.keys())

    fig, axes = plt.subplots(
        nrows=len(datasets),
        ncols=len(SCENARIOS),
        figsize=(14.5, 11.0),
    )

    legend_handles = None

    for row, dataset in enumerate(datasets):
        sample_idx = APPENDIX_SELECTIONS[dataset]["sample"]
        channel_idx = APPENDIX_SELECTIONS[dataset]["channel"]

        for col, scenario in enumerate(SCENARIOS):
            handles = plot_one(
                axes[row, col],
                root,
                dataset,
                scenario,
                sample_idx,
                channel_idx,
                seed=seed,
                title=SCENARIO_LABELS[scenario] if row == 0 else None,
            )

            if legend_handles is None:
                legend_handles = handles

        # Dataset label once per row
        axes[row, 0].text(
            -0.27,
            0.5,
            DATASET_LABELS[dataset],
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )

    # Actual subplot area
    left = 0.11
    right = 0.985
    plot_center = (left + right) / 2

    fig.subplots_adjust(
        left=left,
        right=right,
        top=0.955,
        bottom=0.14,
        wspace=0.23,
        hspace=0.34,
    )

    # Global y label
    fig.supylabel(
        "Standardized value",
        fontsize=14,
        fontweight="bold",
        x=0.025,
    )

    # Centered relative to subplot area
    fig.supxlabel(
        "Time",
        fontsize=14,
        fontweight="bold",
        x=plot_center,
        y=0.075,
    )

    # Legend centered relative to subplot area
    legend = fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc="lower center",
        bbox_to_anchor=(plot_center, 0.01),
        frameon=False,
        fontsize=12,
        ncol=4,
        columnspacing=1.8,
        handlelength=2.0,
        handletextpad=0.6,
    )

    for text in legend.get_texts():
        text.set_fontweight("bold")

    fig.savefig(
        output_dir / "qualitative_appendix.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir / "qualitative_appendix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("qualitative_figures"),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Using fixed selections only.\n")

    print("Main:")
    print(
        MAIN_SELECTION["dataset"],
        "sample =", MAIN_SELECTION["sample"],
        "channel =", MAIN_SELECTION["channel"],
    )

    print("\nAppendix:")
    for dataset, selection in APPENDIX_SELECTIONS.items():
        print(
            DATASET_LABELS[dataset],
            "sample =", selection["sample"],
            "channel =", selection["channel"],
        )

    make_main_figure(
        args.root,
        args.output_dir,
        seed=args.seed,
    )

    make_appendix_figure(
        args.root,
        args.output_dir,
        seed=args.seed,
    )

    print("\nSaved:")
    print(args.output_dir / "qualitative_main.pdf")
    print(args.output_dir / "qualitative_main.png")
    print(args.output_dir / "qualitative_appendix.pdf")
    print(args.output_dir / "qualitative_appendix.png")


if __name__ == "__main__":
    main()