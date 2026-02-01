"""Plot ablation comparison results with std shading.

Generates one plot per experiment file, each with 3 subplots (mse_qpos, mse_mom, mse_energy).
Shows all 3 models × 2 conditions (unguided/guided) with fill_between for std.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Project root
project_root = Path(__file__).parent.parent

# ICLR-friendly style (from plot.py)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.figsize": (6, 2.8),
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})

# Model display names and colors
MODEL_CONFIG = {
    "original": {"label": "Original", "color": "#d62728"},       # red
    "global_cond": {"label": "Global Cond.", "color": "#1f77b4"}, # blue
    "torque_concat": {"label": "Torque Concat.", "color": "#2ca02c"},  # green
}

# Metrics to plot
METRICS = [
    ("mse_qpos", "Position"),
    ("mse_mom", "Momentum"),
    ("mse_energy", "Energy"),
]

# Experiment display names
EXP_TITLES = {
    "exp_a_training": "Training Lengths",
    "exp_a_sinusoidal": "Sinusoidal Forcing",
    "exp_a_gp": "GP Forcing",
    "exp_a_zero": "Zero Forcing",
    "exp_b_context_fractions": "Context Fractions",
}


def load_csv(results_dir: Path, model_name: str, exp_name: str):
    """Load a CSV file for a given model and experiment. Returns None if missing."""
    csv_path = results_dir / model_name / f"{exp_name}.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def get_x_column(exp_name: str) -> str:
    """Return the x-axis column name based on experiment type."""
    if exp_name == "exp_b_context_fractions":
        return "context_fraction"
    return "trajectory_length"


def get_x_label(exp_name: str) -> str:
    """Return x-axis label based on experiment type."""
    if exp_name == "exp_b_context_fractions":
        return "Context Fraction"
    return "Trajectory Length"


def plot_experiment(
    exp_name: str,
    model_names: list,
    results_dir: Path,
    output_dir: Path,
    fmt: str = "png",
    max_length: float = None,
):
    """Generate a single figure with 3 subplots for one experiment."""
    x_col = get_x_column(exp_name)
    x_label = get_x_label(exp_name)

    # Load data for all models
    data = {}
    for model in model_names:
        df = load_csv(results_dir, model, exp_name)
        if df is not None:
            if max_length is not None and x_col == "trajectory_length":
                df = df[df[x_col] <= max_length]
            data[model] = df

    if not data:
        print(f"  Skipping {exp_name}: no data found for any model.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for ax_idx, (metric_key, metric_title) in enumerate(METRICS):
        ax = axes[ax_idx]
        ax.set_title(f"({chr(ord('a') + ax_idx)}) {metric_title}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("MSE")

        for model in model_names:
            if model not in data:
                continue

            df = data[model]
            cfg = MODEL_CONFIG[model]
            x = df[x_col].values

            for condition, linestyle, marker in [
                ("unguided", "--", "o"),
                ("guided", "-", "s"),
            ]:
                mean_col = f"{condition}_{metric_key}_mean"
                std_col = f"{condition}_{metric_key}_std"

                if mean_col not in df.columns:
                    continue

                mean = df[mean_col].values
                std = df[std_col].values

                label = f"{cfg['label']} ({condition.capitalize()})"
                ax.plot(
                    x,
                    mean,
                    marker=marker,
                    linestyle=linestyle,
                    color=cfg["color"],
                    label=label,
                    markersize=4,
                    linewidth=1.5,
                )
                ax.fill_between(
                    x,
                    np.maximum(mean - std, 0),
                    mean + std,
                    alpha=0.12,
                    color=cfg["color"],
                )

        # Only show legend on the first subplot to avoid clutter
        if ax_idx == 0:
            ax.legend(loc="upper left", framealpha=0.9, fontsize=7)

    title = EXP_TITLES.get(exp_name, exp_name)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = output_dir / f"{exp_name}.{fmt}"
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ablation comparison results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(project_root / "output_ablation" / "results"),
        help="Directory containing per-model result CSVs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "output_ablation" / "plots"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--model_names",
        nargs="+",
        default=["original", "global_cond", "torque_concat"],
        help="Model folder names to compare",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format",
    )
    parser.add_argument(
        "--max_length",
        type=float,
        default=None,
        help="Maximum trajectory length to include (exp_a only)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover experiments from CSV files across all model folders
    exp_names = set()
    for model in args.model_names:
        model_dir = results_dir / model
        if model_dir.exists():
            for csv_file in model_dir.glob("*.csv"):
                exp_names.add(csv_file.stem)

    exp_names = sorted(exp_names)
    print(f"Found experiments: {exp_names}")
    print(f"Models: {args.model_names}")

    for exp_name in exp_names:
        print(f"\nPlotting {exp_name}...")
        plot_experiment(exp_name, args.model_names, results_dir, output_dir, args.format, args.max_length)

    print("\nDone.")


if __name__ == "__main__":
    main()
