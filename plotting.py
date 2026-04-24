from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _set_up_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _resolve_output_path(output_path: str | Path) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_simple_regret(simple_regret, output_path: str | Path) -> str:
    plt = _set_up_matplotlib()

    simple_regret = np.asarray(simple_regret, dtype=float)
    eval_counts = np.arange(1, len(simple_regret) + 1)
    plot_regret = np.maximum(simple_regret, 1e-16)

    output_path = _resolve_output_path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eval_counts, plot_regret, marker="o", linewidth=1.8, markersize=3.5)
    ax.set_yscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Simple regret")
    ax.set_title("Simple Regret vs Number of Evaluations")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_regret_runs_with_mean(simple_regret_runs, output_path: str | Path, title: str = "Simple Regret Across Runs") -> str:
    plt = _set_up_matplotlib()

    simple_regret_runs = np.asarray(simple_regret_runs, dtype=float)
    eval_counts = np.arange(1, simple_regret_runs.shape[1] + 1)
    mean_regret = np.maximum(simple_regret_runs.mean(axis=0), 1e-16)
    plot_regret_runs = np.maximum(simple_regret_runs, 1e-16)

    output_path = _resolve_output_path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for run_regret in plot_regret_runs:
        ax.plot(eval_counts, run_regret, color="tab:blue", alpha=0.15, linewidth=0.9)
    ax.plot(eval_counts, mean_regret, color="tab:red", linewidth=2.2, label="mean regret")
    ax.set_yscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Simple regret")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_regret_mean_with_error_bars(
    simple_regret_runs,
    output_path: str | Path,
    title: str = "Mean Simple Regret",
    label: str = "mean regret",
    color: str = "tab:red",
    error_every: int = 5,
) -> str:
    plt = _set_up_matplotlib()

    simple_regret_runs = np.asarray(simple_regret_runs, dtype=float)
    eval_counts = np.arange(1, simple_regret_runs.shape[1] + 1)
    mean_regret = simple_regret_runs.mean(axis=0)
    std_error = simple_regret_runs.std(axis=0) / np.sqrt(simple_regret_runs.shape[0])
    plot_mean = np.maximum(mean_regret, 1e-16)
    error_idx = np.arange(0, len(eval_counts), max(1, int(error_every)))
    plot_err = std_error[error_idx]

    output_path = _resolve_output_path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(eval_counts, plot_mean, color=color, linewidth=2.2, label=label)
    ax.errorbar(
        eval_counts[error_idx],
        plot_mean[error_idx],
        yerr=plot_err,
        fmt="none",
        ecolor=color,
        elinewidth=1.0,
        capsize=2.0,
        alpha=0.85,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Simple regret")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_regret_std(simple_regret_runs, output_path: str | Path, title: str = "Std Deviation of Simple Regret") -> str:
    plt = _set_up_matplotlib()

    simple_regret_runs = np.asarray(simple_regret_runs, dtype=float)
    eval_counts = np.arange(1, simple_regret_runs.shape[1] + 1)
    std_regret = np.maximum(simple_regret_runs.std(axis=0), 1e-16)

    output_path = _resolve_output_path(output_path)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.plot(eval_counts, std_regret, color="tab:green", linewidth=2.2, label="std regret")
    ax.set_yscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Std dev")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_simple_regret_runs(simple_regret_runs, output_path: str | Path) -> str:
    return plot_regret_mean_with_error_bars(
        simple_regret_runs,
        output_path,
        title="Mean Simple Regret Across Runs",
        label="mean regret",
        color="tab:red",
    )


def plot_kernel_comparison_mean(
    regret_runs_by_kernel: dict[str, np.ndarray],
    output_path: str | Path,
    error_every: int = 5,
) -> str:
    plt = _set_up_matplotlib()
    output_path = _resolve_output_path(output_path)

    colors = {"se": "tab:blue", "matrix_se": "tab:red"}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for kernel_name, runs in regret_runs_by_kernel.items():
        runs = np.asarray(runs, dtype=float)
        eval_counts = np.arange(1, runs.shape[1] + 1)
        mean_regret = runs.mean(axis=0)
        std_error = runs.std(axis=0) / np.sqrt(runs.shape[0])
        plot_mean = np.maximum(mean_regret, 1e-16)
        error_idx = np.arange(0, len(eval_counts), max(1, int(error_every)))
        color = colors.get(kernel_name, None)
        ax.plot(eval_counts, plot_mean, linewidth=2.2, color=color, label=kernel_name)
        ax.errorbar(
            eval_counts[error_idx],
            plot_mean[error_idx],
            yerr=std_error[error_idx],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2.0,
            alpha=0.85,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Simple regret")
    ax.set_title("Mean Simple Regret: SE vs Matrix-SE")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def plot_kernel_comparison_std(regret_runs_by_kernel: dict[str, np.ndarray], output_path: str | Path) -> str:
    plt = _set_up_matplotlib()
    output_path = _resolve_output_path(output_path)

    colors = {"se": "tab:blue", "matrix_se": "tab:red"}
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for kernel_name, runs in regret_runs_by_kernel.items():
        runs = np.asarray(runs, dtype=float)
        eval_counts = np.arange(1, runs.shape[1] + 1)
        std_regret = np.maximum(runs.std(axis=0), 1e-16)
        ax.plot(eval_counts, std_regret, linewidth=2.2, color=colors.get(kernel_name, None), label=kernel_name)
    ax.set_yscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel("Std dev")
    ax.set_title("Std Deviation of Simple Regret: SE vs Matrix-SE")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)
