#!/usr/bin/env python3
"""Create seven-method boxplots per torque policy.

For each torque policy, this script saves two separate figures:
  - 2DoF figure
  - 3DoF figure

Each figure keeps the previous method-comparison layout:
  - row 1: RMSE_q
  - row 2: RMSE_p
  - row 3: HamRes

Compared methods:
  1) Unguided DPF
  2) Guided DPF (Gradient / one-step)
  3) Guided DPF (Resampling, m=4)
  4) HNN rollout
  5) Unguided Diffusion
  6) Guided Diffusion (Gradient / one-step)
  7) Guided Diffusion (Resampling, m=4)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.system_eval_utils import SYSTEM_CONFIGS, compute_rmse


ALL_POLICIES = ("sinusoidal", "gp", "zero", "spline")
MAX_LENGTH = 1000

METHODS = [
    # Row-1 (legend): Unguided DPF, Guided DPF (One-step), Guided DPF (Resampling, m=4), HNN rollout
    ("unguided_dpf", "Unguided DPF", "#D7A1B0"),
    ("guided_dpf_onestep", "Guided DPF (Gradient)", "#C24A5A"),
    ("guided_dpf_resampling_m4", "Guided DPF (Resampling, m=4)", "#7A1019"),
    ("hnn_rollout", "HNN rollout", "#D7B347"),
    # Row-2 (legend): Unguided Diffusion, Guided Diffusion (One-step), Guided Diffusion (Resampling, m=4)
    ("unguided_diffusion", "Unguided Diffusion", "#BFD7EA"),
    ("guided_diffusion_onestep", "Guided Diffusion (Gradient)", "#3C8BD8"),
    ("guided_diffusion_resampling_m4", "Guided Diffusion (Resampling, m=4)", "#1E4E8E"),
]

LEGEND_ROW1 = [
    "unguided_dpf",
    "guided_dpf_onestep",
    "guided_dpf_resampling_m4",
    "hnn_rollout",
]
LEGEND_ROW2 = [
    "unguided_diffusion",
    "guided_diffusion_onestep",
    "guided_diffusion_resampling_m4",
]

METRICS = (
    ("rmse_q", r"$\mathrm{RMSE}_q$"),
    ("rmse_p", r"$\mathrm{RMSE}_p$"),
    ("hamres", r"$\mathrm{HamRes}$"),
)
OUTPUT_SUFFIX = "_rmse"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    diff_onestep_2dof = project_root / "output/eval_runs/2dof_transformer_eval_with_hnn/metrics/metrics_per_sample_rmse_hamres.csv"
    diff_onestep_3dof = project_root / "output/eval_runs/3dof_transformer_eval_with_hnn/metrics/metrics_per_sample_rmse_hamres.csv"
    out_dir = project_root / "plots/method_comparison_7methods"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dpf-onestep-2dof-root",
        default=str(project_root / "output/eval_runs/default_full_sweep"),
    )
    parser.add_argument(
        "--dpf-onestep-3dof-root",
        default=str(project_root / "output/eval_runs/default_full_sweep_3dof_structhnn"),
    )
    parser.add_argument(
        "--dpf-onestep-2dof-hamres",
        default=str(project_root / "output/eval_runs/default_full_sweep/metrics/metrics_per_sample.csv"),
    )
    parser.add_argument(
        "--dpf-onestep-3dof-hamres",
        default=str(project_root / "output/eval_runs/default_full_sweep_3dof_structhnn/metrics/metrics_per_sample.csv"),
    )
    parser.add_argument(
        "--dpf-resampling-2dof-root",
        required=True,
    )
    parser.add_argument(
        "--dpf-resampling-3dof-root",
        required=True,
    )
    parser.add_argument(
        "--diff-onestep-2dof",
        default=str(diff_onestep_2dof),
    )
    parser.add_argument(
        "--diff-onestep-3dof",
        default=str(diff_onestep_3dof),
    )
    parser.add_argument(
        "--diff-resampling-2dof",
        required=True,
    )
    parser.add_argument(
        "--diff-resampling-3dof",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        default=str(out_dir),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(ALL_POLICIES),
        choices=list(ALL_POLICIES),
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["2dof", "3dof"],
        choices=["2dof", "3dof"],
    )
    return parser.parse_args()


def _ensure_exists(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")


def _is_complete_policy_cache(df: pd.DataFrame) -> bool:
    if "policy" not in df.columns:
        return False
    found = set(df["policy"].dropna().astype(str).unique().tolist())
    return found == set(ALL_POLICIES)


def build_resampling_dpf_rmse(system: str, root_dir: Path) -> Path:
    out_csv = root_dir / "metrics" / "rmse_range_per_sample_Lle1000.csv"
    if out_csv.exists():
        try:
            df_head = pd.read_csv(out_csv)
            cols = df_head.columns.tolist()
            if (
                "gui_rmse_q" in cols
                and "gui_rmse_p" in cols
                and _is_complete_policy_cache(df_head)
            ):
                return out_csv
        except Exception:
            pass

    cfg = SYSTEM_CONFIGS[system]
    qpos_dim = int(cfg["qpos_dim"])
    mj_model = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    traj_root = root_dir / "trajectories" / system
    if not traj_root.exists():
        raise FileNotFoundError(f"Missing trajectory directory: {traj_root}")

    rows: list[dict] = []
    files = sorted(traj_root.glob("*/*.pt"))
    for pt_path in tqdm(files, desc=f"DPF resampling metrics({system})"):
        bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
        length = int(bundle["length"])
        if length > MAX_LENGTH:
            continue

        combo = str(bundle["combo"])
        policy = str(bundle["policy"])
        seed = int(bundle["seed"])
        states = bundle["guided_states"]
        torques = bundle["guided_torques"]
        for sample_idx in range(states.shape[0]):
            grq, grp = compute_rmse(states[sample_idx], torques[sample_idx], mj_model, qpos_dim)
            rows.append(
                {
                    "combo": combo,
                    "system": system,
                    "policy": policy,
                    "length": length,
                    "seed": seed,
                    "sample_idx": sample_idx,
                    "gui_rmse_q": grq,
                    "gui_rmse_p": grp,
                }
            )

    if not rows:
        raise RuntimeError(f"No rows generated while computing metrics for {system} resampling DPF.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def build_dpf_onestep_rmse(system: str, run_root: Path) -> Path:
    out_csv = run_root / "metrics" / "rmse_range_per_sample_Lle1000.csv"
    if out_csv.exists():
        return out_csv

    cfg = SYSTEM_CONFIGS[system]
    qpos_dim = int(cfg["qpos_dim"])
    mj_model = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    traj_root = run_root / "trajectories" / system
    if not traj_root.exists():
        raise FileNotFoundError(f"Missing DPF one-step trajectories: {traj_root}")

    rows: list[dict] = []
    files = sorted(traj_root.glob("*/*.pt"))
    for pt_path in tqdm(files, desc=f"DPF one-step RMSE({system})"):
        bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
        length = int(bundle["length"])
        if length > MAX_LENGTH:
            continue

        combo = str(bundle["combo"])
        policy = str(bundle["policy"])
        seed = int(bundle["seed"])
        ung_states = bundle["unguided_states"]
        ung_torques = bundle["unguided_torques"]
        gui_states = bundle["guided_states"]
        gui_torques = bundle["guided_torques"]

        for sample_idx in range(ung_states.shape[0]):
            urq, urp = compute_rmse(ung_states[sample_idx], ung_torques[sample_idx], mj_model, qpos_dim)
            grq, grp = compute_rmse(gui_states[sample_idx], gui_torques[sample_idx], mj_model, qpos_dim)
            rows.append(
                {
                    "combo": combo,
                    "system": system,
                    "policy": policy,
                    "length": length,
                    "seed": seed,
                    "sample_idx": sample_idx,
                    "ung_rmse_q": urq,
                    "gui_rmse_q": grq,
                    "ung_rmse_p": urp,
                    "gui_rmse_p": grp,
                }
            )

    if not rows:
        raise RuntimeError(f"No rows generated while building one-step RMSE for {system}.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def load_dpf_onestep_rmse(system: str, run_root: Path, hamres_csv: Path) -> pd.DataFrame:
    _ensure_exists(run_root, f"DPF one-step root ({system})")
    _ensure_exists(hamres_csv, f"DPF one-step HamRes ({system})")
    rmse_csv = build_dpf_onestep_rmse(system, run_root)

    rdf = pd.read_csv(rmse_csv)
    hdf = pd.read_csv(hamres_csv)
    rdf = rdf[(rdf["system"] == system) & (rdf["length"] <= MAX_LENGTH)].copy()
    hdf = hdf[(hdf["system"] == system) & (hdf["length"] <= MAX_LENGTH)].copy()
    keep_h = ["combo", "policy", "length", "seed", "sample_idx", "ung_hamres", "gui_hamres"]
    return pd.merge(
        rdf,
        hdf[keep_h],
        on=["combo", "policy", "length", "seed", "sample_idx"],
        how="inner",
        validate="one_to_one",
    )


def load_dpf_resampling(system: str, root_dir: Path) -> pd.DataFrame:
    metrics_csv = root_dir / "metrics" / "metrics_per_sample.csv"
    _ensure_exists(metrics_csv, f"DPF resampling metrics ({system})")
    rmse_csv = build_resampling_dpf_rmse(system, root_dir)

    ndf = pd.read_csv(rmse_csv)
    hdf = pd.read_csv(metrics_csv)
    ndf = ndf[(ndf["system"] == system) & (ndf["length"] <= MAX_LENGTH)].copy()
    hdf = hdf[(hdf["system"] == system) & (hdf["length"] <= MAX_LENGTH)].copy()
    keep_h = ["combo", "policy", "length", "seed", "sample_idx", "gui_hamres"]
    return pd.merge(
        ndf,
        hdf[keep_h],
        on=["combo", "policy", "length", "seed", "sample_idx"],
        how="inner",
        validate="one_to_one",
    )


def load_diffusion(system: str, csv_path: Path) -> pd.DataFrame:
    _ensure_exists(csv_path, f"Diffusion metrics ({system})")
    df = pd.read_csv(csv_path)
    return df[(df["system"] == system) & (df["length"] <= MAX_LENGTH)].copy()


def _extract_values(
    method_key: str,
    metric_key: str,
    length: int,
    policy: str,
    dpf_onestep: pd.DataFrame,
    dpf_resampling: pd.DataFrame,
    diff_onestep: pd.DataFrame,
    diff_resampling: pd.DataFrame,
) -> np.ndarray:
    if method_key == "unguided_dpf":
        source = dpf_onestep
        col = f"ung_{metric_key}"
    elif method_key == "guided_dpf_resampling_m4":
        source = dpf_resampling
        col = f"gui_{metric_key}"
    elif method_key == "guided_dpf_onestep":
        source = dpf_onestep
        col = f"gui_{metric_key}"
    elif method_key == "unguided_diffusion":
        source = diff_onestep
        col = f"ung_{metric_key}"
    elif method_key == "guided_diffusion_resampling_m4":
        source = diff_resampling
        col = f"gui_{metric_key}"
    elif method_key == "guided_diffusion_onestep":
        source = diff_onestep
        col = f"gui_{metric_key}"
    elif method_key == "hnn_rollout":
        source = diff_onestep
        col = f"hnn_{metric_key}"
    else:
        raise ValueError(f"Unsupported method: {method_key}")

    sub = source[(source["policy"] == policy) & (source["length"] == length)]
    if col not in sub.columns or sub.empty:
        return np.array([np.nan], dtype=float)
    arr = sub[col].to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr if arr.size > 0 else np.array([np.nan], dtype=float)


def _compute_ylim_top_from_unguided_dpf_subset(
    metric_key: str,
    policy: str,
    dpf_onestep: pd.DataFrame,
    lengths_subset: list[int],
) -> float | None:
    if not lengths_subset:
        return None
    sub = dpf_onestep[
        (dpf_onestep["policy"] == policy)
        & (dpf_onestep["length"].isin(lengths_subset))
    ]
    col = f"ung_{metric_key}"
    if sub.empty or col not in sub.columns:
        return None
    vals = sub[col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    vmax = float(np.max(vals))
    if not np.isfinite(vmax) or vmax <= 0.0:
        return None
    return 1.2 * vmax


def make_policy_plot(
    system: str,
    policy: str,
    dpf_onestep: pd.DataFrame,
    dpf_resampling: pd.DataFrame,
    diff_onestep: pd.DataFrame,
    diff_resampling: pd.DataFrame,
    out_dir: Path,
) -> None:
    lengths = sorted(
        set(dpf_onestep[dpf_onestep["policy"] == policy]["length"].unique())
        & set(diff_onestep[diff_onestep["policy"] == policy]["length"].unique())
        & set(diff_resampling[diff_resampling["policy"] == policy]["length"].unique())
        & set(dpf_resampling[dpf_resampling["policy"] == policy]["length"].unique())
    )
    lengths = [int(x) for x in lengths if int(x) <= MAX_LENGTH]
    if not lengths:
        return

    early_lengths = [l for l in lengths if l <= 100]
    mid_lengths = [l for l in lengths if 150 <= l <= 500]
    late_lengths = [l for l in lengths if l > 500]
    length_groups = [("L ≤ 100", early_lengths), ("150 ≤ L ≤ 500", mid_lengths), ("L > 500", late_lengths)]

    left_units = max(len(early_lengths), 1)
    mid_units = max(len(mid_lengths), 1)
    right_units = max(len(late_lengths), 1)
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(38, 17),
        sharex=False,
        sharey=False,
        gridspec_kw={"width_ratios": [left_units, mid_units, right_units]},
    )
    width = 0.09
    center = (len(METHODS) - 1) / 2.0
    offsets = [(idx - center) * width for idx in range(len(METHODS))]

    for row_idx, (metric_key, metric_label) in enumerate(METRICS):
        for col_idx, (col_title, col_lengths) in enumerate(length_groups):
            ax = axes[row_idx, col_idx]
            if not col_lengths:
                ax.axis("off")
                continue

            base = np.arange(len(col_lengths), dtype=float)
            for (method_key, _label, color), offset in zip(METHODS, offsets):
                vals = []
                for length in col_lengths:
                    arr = _extract_values(
                        method_key=method_key,
                        metric_key=metric_key,
                        length=length,
                        policy=policy,
                        dpf_onestep=dpf_onestep,
                        dpf_resampling=dpf_resampling,
                        diff_onestep=diff_onestep,
                        diff_resampling=diff_resampling,
                    )
                    vals.append(arr)

                bp = ax.boxplot(
                    vals,
                    positions=base + offset,
                    widths=width * 0.9,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=2.1),
                    whiskerprops=dict(color=color, linewidth=1.7),
                    capprops=dict(color=color, linewidth=1.7),
                    boxprops=dict(facecolor=color, alpha=0.9, edgecolor=color, linewidth=1.9),
                )
                for box in bp["boxes"]:
                    box.set_facecolor(color)
                    box.set_alpha(0.9)

            if col_idx == 0:
                ax.set_ylabel(metric_label)
            if row_idx == 0:
                ax.set_title(col_title, pad=8)

            ax.set_xticks(base)
            ax.set_xticklabels([str(x) for x in col_lengths], rotation=45, ha="right")
            if row_idx == len(METRICS) - 1:
                ax.set_xlabel("Trajectory Length")

    fig.suptitle(f"{system.upper()} | torque policy={policy}", y=0.955, fontsize=36)

    style_map = {k: (label, color) for k, label, color in METHODS}
    row1_handles = [mpatches.Patch(color=style_map[k][1], label=style_map[k][0]) for k in LEGEND_ROW1]
    row2_handles = [mpatches.Patch(color=style_map[k][1], label=style_map[k][0]) for k in LEGEND_ROW2]
    empty_handle = mpatches.Patch(facecolor="none", edgecolor="none", label=" ")
    row2_padded = row2_handles + [empty_handle]
    # Matplotlib fills legend entries column-major for ncol>1.
    # Interleave top/bottom rows so displayed rows match requested order.
    legend_handles = []
    for top_h, bottom_h in zip(row1_handles, row2_padded):
        legend_handles.extend([top_h, bottom_h])

    if system == "2dof" and policy == "sinusoidal":
        fig.legend(
            handles=legend_handles,
            loc="upper left",
            ncol=4,
            framealpha=0.95,
            bbox_to_anchor=(0.015, 1.01, 0.97, 0.08),
            mode="expand",
            alignment="left",
            borderaxespad=0.0,
            columnspacing=1.8,
            handletextpad=0.7,
        )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"boxplot_{system}_{policy}_7methods{OUTPUT_SUFFIX}.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[saved] {out_png}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = [p for p in args.policies]
    systems = [s for s in args.systems]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 28,
            "axes.labelsize": 32,
            "axes.titlesize": 34,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "legend.fontsize": 28,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    dpf_onestep: dict[str, pd.DataFrame] = {}
    dpf_resampling: dict[str, pd.DataFrame] = {}
    diff_onestep: dict[str, pd.DataFrame] = {}
    diff_resampling: dict[str, pd.DataFrame] = {}

    for system in systems:
        if system == "2dof":
            onestep_root = Path(args.dpf_onestep_2dof_root).resolve()
            onestep_hamres = Path(args.dpf_onestep_2dof_hamres).resolve()
            resampling_root = Path(args.dpf_resampling_2dof_root).resolve()
            diff_one_csv = Path(args.diff_onestep_2dof).resolve()
            diff_res_csv = Path(args.diff_resampling_2dof).resolve()
        else:
            onestep_root = Path(args.dpf_onestep_3dof_root).resolve()
            onestep_hamres = Path(args.dpf_onestep_3dof_hamres).resolve()
            resampling_root = Path(args.dpf_resampling_3dof_root).resolve()
            diff_one_csv = Path(args.diff_onestep_3dof).resolve()
            diff_res_csv = Path(args.diff_resampling_3dof).resolve()

        dpf_onestep[system] = load_dpf_onestep_rmse(system, onestep_root, onestep_hamres)
        dpf_resampling[system] = load_dpf_resampling(system, resampling_root)
        diff_onestep[system] = load_diffusion(system, diff_one_csv)
        diff_resampling[system] = load_diffusion(system, diff_res_csv)

    for policy in policies:
        for system in systems:
            make_policy_plot(
                system=system,
                policy=policy,
                dpf_onestep=dpf_onestep[system],
                dpf_resampling=dpf_resampling[system],
                diff_onestep=diff_onestep[system],
                diff_resampling=diff_resampling[system],
                out_dir=out_dir,
            )

    # Convenience exports for manuscript organization.
    main_dir = out_dir / "main_paper"
    appendix_dir = out_dir / "appendix"
    main_dir.mkdir(parents=True, exist_ok=True)
    appendix_dir.mkdir(parents=True, exist_ok=True)
    for system in systems:
        src = out_dir / f"boxplot_{system}_sinusoidal_7methods{OUTPUT_SUFFIX}.png"
        if src.exists():
            dst = main_dir / src.name
            dst.write_bytes(src.read_bytes())
    for policy in ("gp", "zero", "spline"):
        for system in systems:
            src = out_dir / f"boxplot_{system}_{policy}_7methods{OUTPUT_SUFFIX}.png"
            if src.exists():
                dst = appendix_dir / src.name
                dst.write_bytes(src.read_bytes())

    print(f"[done] outputs under: {out_dir}")


if __name__ == "__main__":
    main()
