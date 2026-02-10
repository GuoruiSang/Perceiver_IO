#!/usr/bin/env python3
"""Build final Protocol-2 reports for Transformer-vs-DPF comparisons.

Outputs:
  - absolute_metrics_summary.csv
  - guidance_ratio_probability_summary.csv
  - policy-level plots for absolute / ratio / probability views
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12
SYSTEMS = ("2dof", "3dof")
METRICS = ("nrmse_q", "nrmse_p", "hamres")
METRIC_LABELS = {
    "nrmse_q": r"$\mathrm{NRMSE}_q$",
    "nrmse_p": r"$\mathrm{NRMSE}_p$",
    "hamres": r"$\mathrm{HamRes}$",
}
METHOD_COLORS = {
    "unguided_dpf": "#D7A1B0",
    "guided_dpf": "#A12C33",
    "unguided_diffusion": "#BFD7EA",
    "guided_diffusion": "#2F66B0",
    "hnn_rollout": "#D7B347",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dpf-2dof-nrmse",
        default=str(project_root / "output_ablation/default_full_sweep/metrics/nrmse_range_per_sample_Lle1000.csv"),
    )
    parser.add_argument(
        "--dpf-2dof-hamres",
        default=str(project_root / "output_ablation/default_full_sweep/metrics/metrics_per_sample.csv"),
    )
    parser.add_argument(
        "--dpf-3dof-nrmse",
        default=str(project_root / "output_ablation/default_full_sweep_3dof_latest_hnn999/metrics/nrmse_range_per_sample_Lle1000.csv"),
    )
    parser.add_argument(
        "--dpf-3dof-hamres",
        default=str(project_root / "output_ablation/default_full_sweep_3dof_latest_hnn999/metrics/metrics_per_sample.csv"),
    )
    parser.add_argument(
        "--diff-2dof",
        default=str(project_root / "output_ablation/protocol2_2dof_transformer_eval_with_hnn/metrics/metrics_per_sample_nrmse_hamres.csv"),
    )
    parser.add_argument(
        "--diff-3dof",
        default=str(project_root / "output_ablation/protocol2_3dof_transformer_eval_with_hnn/metrics/metrics_per_sample_nrmse_hamres.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(project_root / "output_ablation/protocol2_transformer_reports"),
    )
    parser.add_argument(
        "--plots-dir",
        default=str(project_root / "plots/protocol2_transformer_reports"),
    )
    return parser.parse_args()


def _summary_stats(arr: np.ndarray) -> dict:
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def load_dpf(system: str, nrmse_csv: Path, hamres_csv: Path) -> pd.DataFrame:
    if not nrmse_csv.exists():
        raise FileNotFoundError(f"Missing DPF NRMSE file: {nrmse_csv}")
    if not hamres_csv.exists():
        raise FileNotFoundError(f"Missing DPF HamRes file: {hamres_csv}")
    ndf = pd.read_csv(nrmse_csv)
    hdf = pd.read_csv(hamres_csv)
    ndf = ndf[(ndf["system"] == system) & (ndf["length"] <= 1000)].copy()
    hdf = hdf[(hdf["system"] == system) & (hdf["length"] <= 1000)].copy()
    keep_h = ["combo", "policy", "length", "seed", "sample_idx", "ung_hamres", "gui_hamres"]
    merged = pd.merge(
        ndf,
        hdf[keep_h],
        on=["combo", "policy", "length", "seed", "sample_idx"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def load_diff(system: str, diff_csv: Path) -> pd.DataFrame:
    if not diff_csv.exists():
        raise FileNotFoundError(f"Missing diffusion metrics file: {diff_csv}")
    df = pd.read_csv(diff_csv)
    df = df[(df["system"] == system) & (df["length"] <= 1000)].copy()
    return df


def build_absolute_summary(all_dpf: dict[str, pd.DataFrame], all_diff: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    method_map = {
        "unguided_dpf": ("dpf", "ung_{}"),
        "guided_dpf": ("dpf", "gui_{}"),
        "unguided_diffusion": ("diff", "ung_{}"),
        "guided_diffusion": ("diff", "gui_{}"),
        "hnn_rollout": ("diff", "hnn_{}"),
    }
    for system in SYSTEMS:
        dpf = all_dpf[system]
        diff = all_diff[system]
        for policy in sorted(dpf["policy"].unique()):
            lengths = sorted(dpf[dpf["policy"] == policy]["length"].unique())
            for length in lengths:
                for method, (family, col_tmpl) in method_map.items():
                    src = dpf if family == "dpf" else diff
                    sub = src[(src["policy"] == policy) & (src["length"] == length)]
                    if sub.empty:
                        continue
                    for metric in METRICS:
                        col = col_tmpl.format(metric)
                        if col not in sub.columns:
                            continue
                        vals = sub[col].to_numpy(dtype=float)
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue
                        stats = _summary_stats(vals)
                        rows.append(
                            {
                                "system": system,
                                "policy": policy,
                                "length": int(length),
                                "method": method,
                                "metric": metric,
                                **stats,
                            }
                        )
    return pd.DataFrame(rows)


def build_guidance_summary(all_dpf: dict[str, pd.DataFrame], all_diff: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for system in SYSTEMS:
        dpf = all_dpf[system]
        diff = all_diff[system]
        for policy in sorted(dpf["policy"].unique()):
            lengths = sorted(dpf[dpf["policy"] == policy]["length"].unique())
            for length in lengths:
                dpf_sub = dpf[(dpf["policy"] == policy) & (dpf["length"] == length)]
                diff_sub = diff[(diff["policy"] == policy) & (diff["length"] == length)]
                for family, sub in [("dpf", dpf_sub), ("diffusion", diff_sub)]:
                    if sub.empty:
                        continue
                    for metric in METRICS:
                        ung = sub[f"ung_{metric}"].to_numpy(dtype=float)
                        gui = sub[f"gui_{metric}"].to_numpy(dtype=float)
                        ratio = gui / np.maximum(ung, EPS)
                        improve = (gui < ung).astype(float)
                        ratio = ratio[np.isfinite(ratio)]
                        if ratio.size == 0:
                            continue
                        ratio_stats = _summary_stats(ratio)
                        rows.append(
                            {
                                "system": system,
                                "policy": policy,
                                "length": int(length),
                                "family": family,
                                "metric": metric,
                                "ratio_count": ratio_stats["count"],
                                "ratio_mean": ratio_stats["mean"],
                                "ratio_std": ratio_stats["std"],
                                "ratio_median": ratio_stats["median"],
                                "ratio_p25": ratio_stats["p25"],
                                "ratio_p75": ratio_stats["p75"],
                                "prob_improve": float(np.mean(improve)),
                            }
                        )
    return pd.DataFrame(rows)


def plot_policy_curves(abs_df: pd.DataFrame, guide_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    for system in sorted(abs_df["system"].unique()):
        for policy in sorted(abs_df[abs_df["system"] == system]["policy"].unique()):
            abs_sub = abs_df[(abs_df["system"] == system) & (abs_df["policy"] == policy)]
            grd_sub = guide_df[(guide_df["system"] == system) & (guide_df["policy"] == policy)]
            if abs_sub.empty or grd_sub.empty:
                continue

            lengths = sorted(abs_sub["length"].unique())

            # Absolute medians for five methods.
            fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
            for ax, metric in zip(axes, METRICS):
                for method, color in METHOD_COLORS.items():
                    mdf = abs_sub[(abs_sub["metric"] == metric) & (abs_sub["method"] == method)].sort_values("length")
                    if mdf.empty:
                        continue
                    ax.plot(mdf["length"], mdf["median"], marker="o", linewidth=2.0, color=color, label=method)
                ax.set_ylabel(METRIC_LABELS[metric])
                ax.set_title(f"{system.upper()} | torque policy={policy}")
            axes[-1].set_xlabel("Trajectory Length")
            axes[-1].set_xticks(lengths)
            axes[-1].set_xticklabels([str(x) for x in lengths], rotation=45, ha="right")
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), framealpha=0.95)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out_abs = plots_dir / f"protocol2_{system}_{policy}_absolute_median.png"
            fig.savefig(out_abs, dpi=250, bbox_inches="tight")
            plt.close(fig)

            # Ratio + probability for DPF vs diffusion.
            fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
            family_colors = {"dpf": "#A12C33", "diffusion": "#2F66B0"}
            for row, metric in enumerate(METRICS):
                ax_ratio = axes[row, 0]
                ax_prob = axes[row, 1]
                for fam, color in family_colors.items():
                    gdf = grd_sub[(grd_sub["metric"] == metric) & (grd_sub["family"] == fam)].sort_values("length")
                    if gdf.empty:
                        continue
                    ax_ratio.plot(gdf["length"], gdf["ratio_mean"], marker="o", linewidth=2.0, color=color, label=fam)
                    ax_prob.plot(gdf["length"], gdf["prob_improve"], marker="o", linewidth=2.0, color=color, label=fam)
                ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                ax_prob.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
                ax_ratio.set_ylabel(f"{METRIC_LABELS[metric]} ratio")
                ax_prob.set_ylabel(f"{METRIC_LABELS[metric]} P(improve)")
            axes[-1, 0].set_xlabel("Trajectory Length")
            axes[-1, 1].set_xlabel("Trajectory Length")
            axes[-1, 0].set_xticks(lengths)
            axes[-1, 1].set_xticks(lengths)
            axes[-1, 0].set_xticklabels([str(x) for x in lengths], rotation=45, ha="right")
            axes[-1, 1].set_xticklabels([str(x) for x in lengths], rotation=45, ha="right")
            axes[0, 0].set_title(f"{system.upper()} | {policy} | guided/unguided ratio")
            axes[0, 1].set_title(f"{system.upper()} | {policy} | probability of improvement")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.02), framealpha=0.95)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out_cmp = plots_dir / f"protocol2_{system}_{policy}_ratio_probability.png"
            fig.savefig(out_cmp, dpi=250, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    plots_dir = Path(args.plots_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    dpf_paths = {
        "2dof": (Path(args.dpf_2dof_nrmse).resolve(), Path(args.dpf_2dof_hamres).resolve()),
        "3dof": (Path(args.dpf_3dof_nrmse).resolve(), Path(args.dpf_3dof_hamres).resolve()),
    }
    diff_paths = {
        "2dof": Path(args.diff_2dof).resolve(),
        "3dof": Path(args.diff_3dof).resolve(),
    }

    all_dpf = {s: load_dpf(s, *dpf_paths[s]) for s in SYSTEMS}
    all_diff = {s: load_diff(s, diff_paths[s]) for s in SYSTEMS}

    abs_df = build_absolute_summary(all_dpf, all_diff)
    guide_df = build_guidance_summary(all_dpf, all_diff)

    abs_csv = out_dir / "absolute_metrics_summary.csv"
    guide_csv = out_dir / "guidance_ratio_probability_summary.csv"
    abs_df.to_csv(abs_csv, index=False)
    guide_df.to_csv(guide_csv, index=False)
    print(f"[saved] {abs_csv}")
    print(f"[saved] {guide_csv}")

    plot_policy_curves(abs_df, guide_df, plots_dir)
    print(f"[saved] plots under {plots_dir}")


if __name__ == "__main__":
    main()

