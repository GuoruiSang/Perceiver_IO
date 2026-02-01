"""
Generate Markdown ablation tables from CSV result files.

Reads per-model CSV results and produces Markdown tables comparing
models across trajectory lengths / context fractions.

Usage:
    python scripts/generate_ablation_tables.py \
        --results_dir output_ablation/results \
        --output_dir output_ablation/tables
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import math
import pandas as pd
from typing import Dict, List, Optional


MODEL_DISPLAY = {
    "original": "Original",
    "global_cond": "Global Cond.",
    "torque_concat": "Torque Concat.",
}

METRIC_DISPLAY = {
    "mse_qpos": "MSE Position",
    "mse_mom": "MSE Momentum",
    "mse_energy": "MSE Energy",
}

EXPERIMENT_DISPLAY = {
    "exp_a_training": "Experiment A: Training Torques",
    "exp_a_sinusoidal": "Experiment A: Sinusoidal Torques",
    "exp_a_gp": "Experiment A: GP Torques",
    "exp_a_zero": "Experiment A: Zero Torques",
    "exp_b_context_fractions": "Experiment B: Context Fractions",
}


def discover_experiments(results_dir: Path, model_names: List[str]) -> List[str]:
    """Find all unique experiment file names across all models."""
    experiments = set()
    for model in model_names:
        model_dir = results_dir / model
        if model_dir.exists():
            for csv_file in model_dir.glob("*.csv"):
                experiments.add(csv_file.stem)
    return sorted(experiments)


def load_experiment_data(
    results_dir: Path,
    model_names: List[str],
    experiment_name: str,
) -> Dict[str, Optional[pd.DataFrame]]:
    """Load CSV files for all models for a given experiment."""
    data = {}
    for model in model_names:
        csv_path = results_dir / model / f"{experiment_name}.csv"
        if csv_path.exists():
            data[model] = pd.read_csv(csv_path)
        else:
            print(f"  Warning: Missing {csv_path}")
            data[model] = None
    return data


def format_mean_std(mean: float, std: float, sig_figs: int = 3) -> str:
    """Format mean +/- std with consistent significant figures."""
    if pd.isna(mean) or pd.isna(std):
        return "---"
    m_str = f"{mean:.{sig_figs}g}"
    s_str = f"{std:.{sig_figs}g}"
    return f"{m_str} ± {s_str}"


def highlight_top3_in_row(
    values: List[Optional[float]],
    formatted: List[str],
) -> List[str]:
    """Mark the top-3 lowest cells: bold (1st), underline (2nd), italic (3rd).

    Args:
        values: raw mean values per cell (None if missing).
        formatted: formatted strings per cell.

    Returns:
        formatted strings with 1st wrapped in **...**,
        2nd in <u>...</u>, and 3rd in *...*.
    """
    valid = [(i, v) for i, v in enumerate(values) if v is not None and not math.isnan(v)]
    if len(valid) < 2:
        return formatted

    sorted_valid = sorted(valid, key=lambda x: x[1])
    rank_map = {}
    if len(sorted_valid) >= 1:
        rank_map[sorted_valid[0][0]] = 1
    if len(sorted_valid) >= 2:
        rank_map[sorted_valid[1][0]] = 2
    if len(sorted_valid) >= 3:
        rank_map[sorted_valid[2][0]] = 3

    result = []
    for i, s in enumerate(formatted):
        rank = rank_map.get(i)
        if rank == 1 and s != "---":
            result.append(f"**{s}**")
        elif rank == 2 and s != "---":
            result.append(f"<u>{s}</u>")
        elif rank == 3 and s != "---":
            result.append(f"*{s}*")
        else:
            result.append(s)
    return result


def generate_metric_table(
    experiment_data: Dict[str, Optional[pd.DataFrame]],
    index_col: str,
    metric_key: str,
    model_names: List[str],
    sig_figs: int = 3,
    bold_best: bool = True,
    row_subset: Optional[List] = None,
    row_stride: Optional[int] = None,
) -> str:
    """Generate a markdown table for one metric across all models.

    Args:
        experiment_data: model_name -> DataFrame (or None).
        index_col: "trajectory_length" or "context_fraction".
        metric_key: e.g., "mse_qpos" (expanded to unguided/guided _mean/_std).
        model_names: ordered list of model names.
        sig_figs: significant figures for formatting.
        bold_best: whether to bold the best value per row.
        row_subset: if set, only include these index values.
        row_stride: if set, include every Nth row.

    Returns:
        Markdown table string.
    """
    # Collect all index values across models
    all_indices = set()
    for df in experiment_data.values():
        if df is not None:
            all_indices.update(df[index_col].tolist())
    all_indices = sorted(all_indices)

    # Apply row filtering
    if row_subset is not None:
        all_indices = [idx for idx in all_indices if idx in row_subset]
    elif row_stride is not None:
        all_indices = all_indices[::row_stride]

    # Build header
    index_label = "Length" if index_col == "trajectory_length" else "Ctx Frac"
    header_parts = [index_label]
    for model in model_names:
        display = MODEL_DISPLAY.get(model, model)
        header_parts.append(f"{display} (Ung.)")
        header_parts.append(f"{display} (Guid.)")

    header_line = "| " + " | ".join(header_parts) + " |"
    separator = "| " + " | ".join(["---"] * len(header_parts)) + " |"

    rows = [header_line, separator]

    for idx in all_indices:
        # Format index value
        if index_col == "trajectory_length":
            idx_str = str(int(idx))
        else:
            idx_str = f"{idx:.2f}"

        # Collect all values across models (interleaved: ung, guid, ung, guid, ...)
        all_values = []
        all_formatted = []

        for model in model_names:
            df = experiment_data[model]
            if df is None:
                all_values.extend([None, None])
                all_formatted.extend(["---", "---"])
                continue

            row_data = df[df[index_col] == idx]
            if row_data.empty:
                all_values.extend([None, None])
                all_formatted.extend(["---", "---"])
                continue

            row_data = row_data.iloc[0]

            ung_mean = row_data.get(f"unguided_{metric_key}_mean")
            ung_std = row_data.get(f"unguided_{metric_key}_std")
            guid_mean = row_data.get(f"guided_{metric_key}_mean")
            guid_std = row_data.get(f"guided_{metric_key}_std")

            all_values.append(ung_mean)
            all_formatted.append(format_mean_std(ung_mean, ung_std, sig_figs))
            all_values.append(guid_mean)
            all_formatted.append(format_mean_std(guid_mean, guid_std, sig_figs))

        if bold_best:
            all_formatted = highlight_top3_in_row(all_values, all_formatted)

        cells = [idx_str] + all_formatted

        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def generate_experiment_markdown(
    experiment_data: Dict[str, Optional[pd.DataFrame]],
    experiment_name: str,
    index_col: str,
    model_names: List[str],
    metrics: List[str],
    sig_figs: int = 3,
    bold_best: bool = True,
    row_subset: Optional[List] = None,
    row_stride: Optional[int] = None,
) -> str:
    """Generate full markdown for one experiment with sub-tables per metric."""
    exp_display = EXPERIMENT_DISPLAY.get(experiment_name, experiment_name)
    sections = [f"# {exp_display}\n"]

    for metric_key in metrics:
        metric_display = METRIC_DISPLAY.get(metric_key, metric_key)
        sections.append(f"## {metric_display}\n")
        table = generate_metric_table(
            experiment_data=experiment_data,
            index_col=index_col,
            metric_key=metric_key,
            model_names=model_names,
            sig_figs=sig_figs,
            bold_best=bold_best,
            row_subset=row_subset,
            row_stride=row_stride,
        )
        sections.append(table)
        sections.append("")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown ablation tables from CSV results")
    parser.add_argument("--results_dir", type=str, default="output_ablation/results")
    parser.add_argument("--output_dir", type=str, default="output_ablation/tables")
    parser.add_argument("--model_names", type=str, nargs="+",
                        default=["original", "global_cond", "torque_concat"])
    parser.add_argument("--metrics", type=str, nargs="+",
                        default=["mse_qpos", "mse_mom", "mse_energy"])
    parser.add_argument("--precision", type=int, default=3,
                        help="Significant figures for formatted values")
    parser.add_argument("--no_bold_best", action="store_true",
                        help="Disable bolding the best value per row")
    parser.add_argument("--row_subset", type=float, nargs="+", default=None,
                        help="Only include these index values (trajectory lengths or context fractions)")
    parser.add_argument("--row_stride", type=int, default=None,
                        help="Include every Nth row")
    args = parser.parse_args()

    results_dir = project_root / args.results_dir
    output_dir = project_root / args.output_dir

    print(f"Results dir: {results_dir}")
    print(f"Output dir:  {output_dir}")

    experiments = discover_experiments(results_dir, args.model_names)
    print(f"Found experiments: {experiments}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for exp_name in experiments:
        index_col = "context_fraction" if "exp_b" in exp_name else "trajectory_length"
        print(f"\nProcessing: {exp_name}")

        exp_data = load_experiment_data(results_dir, args.model_names, exp_name)

        md = generate_experiment_markdown(
            experiment_data=exp_data,
            experiment_name=exp_name,
            index_col=index_col,
            model_names=args.model_names,
            metrics=args.metrics,
            sig_figs=args.precision,
            bold_best=not args.no_bold_best,
            row_subset=args.row_subset,
            row_stride=args.row_stride,
        )

        output_path = output_dir / f"{exp_name}.md"
        output_path.write_text(md)
        print(f"  Saved: {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
