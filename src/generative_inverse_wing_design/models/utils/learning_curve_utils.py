from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def collect_learning_curve_runs(
    root: str | Path,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[Path]]:
    # 1. Find all seed folders
    root_path = Path(root)
    seed_dirs = [d for d in root_path.glob("seed_*") if d.is_dir()]

    all_train_dfs: list[pd.DataFrame] = []
    all_val_dfs: list[pd.DataFrame] = []

    # 2. Collect all train/val learning curves from all seeds
    for seed_dir in seed_dirs:
        csv_files = glob.glob(os.path.join(seed_dir, "**", "*_learning_curve.csv"), recursive=True)
        train_files = [f for f in csv_files if "_train_learning_curve.csv" in f]

        for train_file in train_files:
            val_file = train_file.replace("_train_learning_curve.csv", "_val_learning_curve.csv")
            if not os.path.exists(val_file):
                continue

            # Group by epoch and average if there are multiple entries per epoch
            train_df = pd.read_csv(train_file).groupby("epoch").mean().reset_index()
            val_df = pd.read_csv(val_file).groupby("epoch").mean().reset_index()
            all_train_dfs.append(train_df)
            all_val_dfs.append(val_df)

    return all_train_dfs, all_val_dfs, seed_dirs


def aggregate_learning_curve_runs(
    all_train_dfs: list[pd.DataFrame],
    all_val_dfs: list[pd.DataFrame],
) -> tuple[list[float], list[str], dict[str, list[list[float]]], dict[str, list[list[float]]]]:
    if not all_train_dfs or not all_val_dfs:
        raise ValueError("No train/val learning curve pairs found.")

    # 3. Align by epoch (outer join), then aggregate
    all_epochs = sorted(
        set(np.concatenate([df["epoch"].values for df in all_train_dfs + all_val_dfs]))
    )
    metrics = [col for col in all_train_dfs[0].columns if col != "epoch"]

    train_metrics_by_epoch = {metric: [] for metric in metrics}
    val_metrics_by_epoch = {metric: [] for metric in metrics}

    for epoch in all_epochs:
        for metric in metrics:
            train_vals = [
                df.loc[df["epoch"] == epoch, metric].values[0]
                for df in all_train_dfs
                if epoch in df["epoch"].values
            ]
            val_vals = [
                df.loc[df["epoch"] == epoch, metric].values[0]
                for df in all_val_dfs
                if epoch in df["epoch"].values
            ]

            train_metrics_by_epoch[metric].append(train_vals if train_vals else [np.nan])
            val_metrics_by_epoch[metric].append(val_vals if val_vals else [np.nan])

    return all_epochs, metrics, train_metrics_by_epoch, val_metrics_by_epoch


def _directional_std(values: list[list[float]], means: list[float], direction: str) -> list[float]:
    if direction not in {"upper", "lower"}:
        raise ValueError("direction must be 'upper' or 'lower'.")

    if direction == "upper":
        return [np.nanstd([v for v in vals if v >= mean]) for vals, mean in zip(values, means)]
    return [np.nanstd([v for v in vals if v <= mean]) for vals, mean in zip(values, means)]


def plot_learning_curves_mean_std(
    root: str | Path,
    save_path: str | Path | None = None,
) -> plt.Figure:
    all_train_dfs, all_val_dfs, seed_dirs = collect_learning_curve_runs(root)
    print(
        f"Found {len(all_train_dfs)} train and {len(all_val_dfs)} val learning curves "
        f"from {len(seed_dirs)} seeds."
    )

    all_epochs, metrics, train_metrics_by_epoch, val_metrics_by_epoch = (
        aggregate_learning_curve_runs(
            all_train_dfs,
            all_val_dfs,
        )
    )

    # 4. Plot mean ± directional std for each metric
    n_metrics = len(metrics)
    fig, axs = plt.subplots(n_metrics, 1, figsize=(8, 3 * n_metrics))
    if n_metrics == 1:
        axs = [axs]

    for i, metric in enumerate(metrics):
        train_metrics = train_metrics_by_epoch[metric]
        val_metrics = val_metrics_by_epoch[metric]

        train_means = [np.nanmean(vals) for vals in train_metrics]
        val_means = [np.nanmean(vals) for vals in val_metrics]

        train_upper_stds = _directional_std(train_metrics, train_means, "upper")
        train_lower_stds = _directional_std(train_metrics, train_means, "lower")
        val_upper_stds = _directional_std(val_metrics, val_means, "upper")
        val_lower_stds = _directional_std(val_metrics, val_means, "lower")

        if "weight" not in metric.lower():
            axs[i].fill_between(
                all_epochs,
                np.array(val_means) - np.array(val_upper_stds),
                np.array(val_means) + np.array(val_lower_stds),
                color="tab:blue",
                alpha=0.2,
            )
            axs[i].fill_between(
                all_epochs,
                np.array(train_means) - np.array(train_upper_stds),
                np.array(train_means) + np.array(train_lower_stds),
                color="tab:orange",
                alpha=0.4,
            )

        axs[i].plot(all_epochs, val_means, label="val mean", color="tab:blue", linewidth=2)
        if "weight" not in metric.lower():
            axs[i].plot(
                all_epochs, train_means, label="train mean", color="tab:orange", linewidth=2
            )

        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("epoch")
        axs[i].set_facecolor("white")
        axs[i].set_xlim(0, max(all_epochs))
        axs[i].legend(facecolor="white")
        axs[i].grid(True, alpha=0.3, color="black", linewidth=0.5)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=600, format=Path(save_path).suffix.lstrip(".") or "svg")
    plt.show()
    return fig


def plot_preprocessing_learning_curves_curated(
    root: str | Path,
    save_path: str | Path | None = "learning_curves_preprocessing_vae_curated.svg",
) -> plt.Figure:
    all_train_dfs, all_val_dfs, seed_dirs = collect_learning_curve_runs(root)
    print(
        f"Found {len(all_train_dfs)} train and {len(all_val_dfs)} val learning curves "
        f"from {len(seed_dirs)} seeds."
    )

    all_epochs, _, train_metrics_by_epoch, val_metrics_by_epoch = aggregate_learning_curve_runs(
        all_train_dfs,
        all_val_dfs,
    )

    # 4. Plot curated two-panel figure for the paper
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    metric = "loss"
    train_metrics = train_metrics_by_epoch[metric]
    val_metrics = val_metrics_by_epoch[metric]

    train_means = [np.nanmean(vals) for vals in train_metrics]
    val_means = [np.nanmean(vals) for vals in val_metrics]
    train_upper_stds = _directional_std(train_metrics, train_means, "upper")
    train_lower_stds = _directional_std(train_metrics, train_means, "lower")
    val_upper_stds = _directional_std(val_metrics, val_means, "upper")
    val_lower_stds = _directional_std(val_metrics, val_means, "lower")

    axs[0].fill_between(
        all_epochs,
        np.array(val_means) - np.array(val_upper_stds),
        np.array(val_means) + np.array(val_lower_stds),
        color="tab:blue",
        alpha=0.2,
    )
    axs[0].fill_between(
        all_epochs,
        np.array(train_means) - np.array(train_upper_stds),
        np.array(train_means) + np.array(train_lower_stds),
        color="tab:orange",
        alpha=0.4,
    )

    axs[0].plot(all_epochs, val_means, label="val mean", color="tab:blue", linewidth=2)
    axs[0].plot(all_epochs, train_means, label="train mean", color="tab:orange", linewidth=2)
    axs[0].set_ylabel(metric, fontsize=12)
    axs[0].set_facecolor("white")
    axs[0].set_xlim(0, max(all_epochs))
    axs[0].set_ylim(0, max(max(train_means), max(val_means)) * 1.1)
    axs[0].legend(facecolor="white")
    axs[0].grid(True, alpha=0.3, color="black", linewidth=0.5)

    # weight schedules
    train_means_aero = [np.nanmean(vals) for vals in train_metrics_by_epoch["aero_weight"]]
    train_means_kl = [np.nanmean(vals) for vals in train_metrics_by_epoch["kl_weight"]]

    axs[1].plot(all_epochs, train_means_aero, label="aero_weight", color="tab:orange", linewidth=2)
    axs[1].plot(all_epochs, train_means_kl, label="kl_weight", color="tab:blue", linewidth=2)
    axs[1].set_ylabel("weights", fontsize=12)
    axs[1].set_xlabel("epoch", fontsize=12)
    axs[1].set_facecolor("white")
    axs[1].set_xlim(0, max(all_epochs))
    axs[1].set_ylim(0, 0.7)
    axs[1].legend(facecolor="white")
    axs[1].grid(True, alpha=0.3, color="black", linewidth=0.5)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=600, format=Path(save_path).suffix.lstrip(".") or "svg")
    plt.show()
    return fig


def plot_cvae_learning_curves_curated(
    root: str | Path,
    save_path: str | Path | None = "learning_curves_cvae_curated.svg",
) -> plt.Figure:
    all_train_dfs, all_val_dfs, seed_dirs = collect_learning_curve_runs(root)
    print(
        f"Found {len(all_train_dfs)} train and {len(all_val_dfs)} val learning curves "
        f"from {len(seed_dirs)} seeds."
    )

    all_epochs, _, train_metrics_by_epoch, val_metrics_by_epoch = aggregate_learning_curve_runs(
        all_train_dfs,
        all_val_dfs,
    )

    # 4. Plot curated two-panel figure for cVAE report plots
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    train_metrics_loss = train_metrics_by_epoch["loss"]
    val_metrics_loss = val_metrics_by_epoch["loss"]
    train_means_loss = [np.nanmean(vals) for vals in train_metrics_loss]
    val_means_loss = [np.nanmean(vals) for vals in val_metrics_loss]

    train_upper_stds_loss = _directional_std(train_metrics_loss, train_means_loss, "upper")
    train_lower_stds_loss = _directional_std(train_metrics_loss, train_means_loss, "lower")
    val_upper_stds_loss = _directional_std(val_metrics_loss, val_means_loss, "upper")
    val_lower_stds_loss = _directional_std(val_metrics_loss, val_means_loss, "lower")

    axs[0].fill_between(
        all_epochs,
        np.array(val_means_loss) - np.array(val_upper_stds_loss),
        np.array(val_means_loss) + np.array(val_lower_stds_loss),
        color="tab:blue",
        alpha=0.2,
    )
    axs[0].fill_between(
        all_epochs,
        np.array(train_means_loss) - np.array(train_upper_stds_loss),
        np.array(train_means_loss) + np.array(train_lower_stds_loss),
        color="tab:orange",
        alpha=0.4,
    )
    axs[0].plot(all_epochs, val_means_loss, label="val mean", color="tab:blue", linewidth=2)
    axs[0].plot(all_epochs, train_means_loss, label="train mean", color="tab:orange", linewidth=2)
    axs[0].set_ylabel("loss", fontsize=12)
    axs[0].tick_params(axis="y")
    axs[0].set_facecolor("white")
    axs[0].set_xlim(0, max(all_epochs))
    axs[0].grid(True, alpha=0.3, color="black", linewidth=0.5)

    lines, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(lines, labels, facecolor="white", loc="upper right")

    # Scale geometry weight to overlay with iou_weight in one axis
    train_means_geometry = [
        np.nanmean(vals) * 0.001 for vals in train_metrics_by_epoch["geometry_weight"]
    ]
    train_means_iou = [np.nanmean(vals) for vals in train_metrics_by_epoch["iou_weight"]]

    axs[1].plot(
        all_epochs,
        train_means_geometry,
        label="geometry_weight * 0.001",
        color="tab:blue",
        linewidth=2,
    )
    axs[1].plot(all_epochs, train_means_iou, label="iou_weight", color="tab:orange", linewidth=2)
    axs[1].set_ylabel("weights", fontsize=12)
    axs[1].set_xlabel("epoch", fontsize=12)
    axs[1].set_facecolor("white")
    axs[1].set_xlim(0, max(all_epochs))
    axs[1].set_ylim(0, max(max(train_means_geometry), max(train_means_iou)) * 1.1)
    axs[1].grid(True, alpha=0.3, color="black", linewidth=0.5)

    lines, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(lines, labels, facecolor="white", loc="lower right")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=600, format=Path(save_path).suffix.lstrip(".") or "svg")
    plt.show()
    return fig
