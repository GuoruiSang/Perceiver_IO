#!/usr/bin/env python3
"""Protocol-2 comparison plots for 2DoF:

For each policy, create a separate figure with boxplots across lengths (50..1000),
comparing 5 methods:
  1) Unguided DPF
  2) Guided DPF
  3) Unguided Diffusion (fixed-length model, generate L=1000 then crop to L)
  4) Guided Diffusion (fixed-length model, generate L=1000 then crop to L)
  5) HNN rollout (free rollout; initialized from unguided diffusion x0)

Metrics: NRMSE_q(range), NRMSE_p(range), HamRes.
"""
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'scripts'))

import scripts.quick_eval as qe
from scripts.compute_ablation_2dof_with_smoothing import (
    SYSTEM_CONFIGS,
    BATCH_SIZE_UNGUIDED,
    BATCH_SIZE_GUIDED,
    HAMRES_SMOOTH_SIGMA,
    HAMRES_PSEUDO_HUBER_DELTA,
    HAMRES_MIN_SCALE_Q,
    HAMRES_MIN_SCALE_P,
    load_torques,
    compute_hamres,
)
from src.models.utils import reconstruct_traj_with_momentum


# ---- Run config ----
SYSTEM = "2dof"
POLICIES = ["sinusoidal", "gp", "zero", "spline"]
LENGTHS = list(range(50, 1001, 50))
NUM_SAMPLES = 100
SEED = 10
MIN_SCALE_Q = 1e-3
MIN_SCALE_P = 1e-3

def _env_path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    return Path(v) if v else default


FIXED_DIFFUSION_CKPT = _env_path(
    "FIXED_DIFFUSION_CKPT",
    (
    project_root
    / "checkpoints"
    / "2dof"
    / "trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0008.ckpt"
    ),
)

OUT_ROOT = _env_path("OUT_ROOT", project_root / "output_ablation" / "protocol2_2dof_fixed_diffusion_eval_with_hnn")
METRICS_ROOT = OUT_ROOT / "metrics"
PLOTS_ROOT = _env_path("PLOTS_ROOT", project_root / "plots")
METRICS_ROOT.mkdir(parents=True, exist_ok=True)
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

DIFF_PER_SAMPLE_CSV = METRICS_ROOT / "metrics_per_sample_nrmse_hamres.csv"
DIFF_DONE_JSON = OUT_ROOT / "completed_combos.json"
DIFF_MANIFEST_JSON = OUT_ROOT / "run_manifest.json"

DPF_NRMSE_CSV = _env_path(
    "DPF_NRMSE_CSV",
    (
    project_root
    / "output_ablation"
    / "default_full_sweep"
    / "metrics"
    / "nrmse_range_per_sample_Lle1000.csv"
    ),
)
DPF_HAMRES_CSV = _env_path(
    "DPF_HAMRES_CSV",
    (
    project_root
    / "output_ablation"
    / "default_full_sweep"
    / "metrics"
    / "metrics_per_sample.csv"
    ),
)


def run_hnn_rollout_batch(hnn, init_states, torques):
    """Run free HNN rollout from provided initial states.

    Args:
        init_states: [N, T, D], using timestep 0 as x0
        torques: [N, T, torque_dim]
    Returns:
        hnn_states: [N, T, D]
    """
    q0 = init_states[:, 0, : init_states.shape[-1] // 2]
    p0 = init_states[:, 0, init_states.shape[-1] // 2 :]
    tau_seq = torques.permute(1, 0, 2)  # [T, N, torque_dim]
    with torch.no_grad():
        p_traj, q_traj, _ = hnn.integrate_trajectory(
            p0=p0,
            q0=q0,
            tau_seq=tau_seq,
            dt=qe.DT if hasattr(qe, "DT") else 0.0002,
            num_steps=torques.shape[1] - 1,
        )
    q = q_traj.permute(1, 0, 2)  # [N, T, qdim]
    p = p_traj.permute(1, 0, 2)  # [N, T, qdim]
    return torch.cat([q, p], dim=-1)


def combo_id(system, policy, length, seed, num_samples):
    return f"{system}_{policy}_L{length}_N{num_samples}_seed{seed}"


def append_csv(path, fieldnames, row):
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def resolve_default_guidance(system, cfg):
    preset_key = qe.GUIDANCE_PRESET.lower()
    if preset_key == "auto":
        preset_key = "robust_r9_comboa" if system == "2dof" else "one_step_best"

    if preset_key == "custom":
        return {
            "guidance_method": qe.GUIDANCE_METHOD,
            "optimize_target": qe.OPTIMIZE_TARGET,
            "guidance_energy_mode": qe.GUIDANCE_ENERGY_MODE,
            "guidance_steps": qe.GUIDANCE_STEPS if qe.GUIDANCE_STEPS is not None else cfg["guidance_steps"],
            "guidance_lr": qe.GUIDANCE_LR if qe.GUIDANCE_LR is not None else cfg["guidance_lr"],
            "guidance_after_steps": qe.GUIDANCE_AFTER if qe.GUIDANCE_AFTER is not None else cfg["guidance_after_steps"],
            "guidance_before_steps": qe.GUIDANCE_BEFORE,
            "guidance_trust_lambda": qe.GUIDANCE_TRUST_LAMBDA,
            "guidance_hamres_smooth_sigma": qe.GUIDANCE_HAMRES_SMOOTH_SIGMA,
            "guidance_hamres_delta": qe.GUIDANCE_HAMRES_DELTA,
            "guidance_hamres_min_scale_q": qe.GUIDANCE_HAMRES_MIN_SCALE_Q,
            "guidance_hamres_min_scale_p": qe.GUIDANCE_HAMRES_MIN_SCALE_P,
        }

    p = qe.GUIDANCE_PRESETS[preset_key]
    return {
        "guidance_method": p["guidance_method"],
        "optimize_target": p["optimize_target"],
        "guidance_energy_mode": p["guidance_energy_mode"],
        "guidance_steps": p["guidance_steps"],
        "guidance_lr": p["guidance_lr"],
        "guidance_after_steps": p["guidance_after"],
        "guidance_before_steps": p["guidance_before"],
        "guidance_trust_lambda": p["guidance_trust_lambda"],
        "guidance_hamres_smooth_sigma": p["guidance_hamres_smooth_sigma"],
        "guidance_hamres_delta": p["guidance_hamres_delta"],
        "guidance_hamres_min_scale_q": p["guidance_hamres_min_scale_q"],
        "guidance_hamres_min_scale_p": p["guidance_hamres_min_scale_p"],
    }


def compute_nrmse_range_and_hamres(
    state: torch.Tensor,
    tau: torch.Tensor,
    qpos_dim: int,
    mj_model,
    hnn,
    var_dq,
    var_dp,
    hamres_kwargs,
):
    qpos = state[:, :qpos_dim].cpu().numpy()
    mom = state[:, qpos_dim:].cpu().numpy()
    tau_np = tau.cpu().numpy()
    t_len = qpos.shape[0]

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)
    M = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M, data.qM)
    initial_qvel = np.linalg.solve(M, mom[0])

    recon = reconstruct_traj_with_momentum(
        mj_model, t_len, qe.SIM_DT if hasattr(qe, "SIM_DT") else 0.0001, qpos[0], initial_qvel, tau_np, data_dt=qe.DT if hasattr(qe, "DT") else 0.0002
    )
    gt_qpos = recon["seq_qpos"]
    gt_mom = recon["seq_mom"]

    t_min = min(len(qpos) - 1, len(gt_qpos))
    if t_min <= 0:
        nrmse_q = np.nan
        nrmse_p = np.nan
    else:
        e_q = qpos[1 : t_min + 1] - gt_qpos[:t_min]
        e_p = mom[1 : t_min + 1] - gt_mom[:t_min]
        rmse_q = np.sqrt((e_q**2).mean(axis=0))
        rmse_p = np.sqrt((e_p**2).mean(axis=0))
        range_q = gt_qpos[:t_min].max(axis=0) - gt_qpos[:t_min].min(axis=0)
        range_p = gt_mom[:t_min].max(axis=0) - gt_mom[:t_min].min(axis=0)
        scale_q = np.maximum(range_q, MIN_SCALE_Q)
        scale_p = np.maximum(range_p, MIN_SCALE_P)
        nrmse_q = float((rmse_q / scale_q).mean())
        nrmse_p = float((rmse_p / scale_p).mean())

    hr = compute_hamres(
        state[:, :qpos_dim],
        state[:, qpos_dim:],
        tau,
        hnn,
        var_dq,
        var_dp,
        **hamres_kwargs,
    )
    return nrmse_q, nrmse_p, float(hr)


def run_diffusion_eval_if_needed():
    expected_combos = len(POLICIES)
    done = set()
    if DIFF_DONE_JSON.exists():
        done = set(json.loads(DIFF_DONE_JSON.read_text()))

    if len(done) >= expected_combos and DIFF_PER_SAMPLE_CSV.exists():
        print(f"[Skip] diffusion eval already complete ({len(done)}/{expected_combos} policies).")
        return

    cfg = dict(SYSTEM_CONFIGS[SYSTEM])
    cfg["dpf_ckpt"] = FIXED_DIFFUSION_CKPT
    if not cfg["dpf_ckpt"].exists():
        raise FileNotFoundError(f"Fixed diffusion checkpoint not found: {cfg['dpf_ckpt']}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qe.DEVICE = str(device)
    qe.set_seed(SEED)
    print(f"[Diffusion Eval] device={device}")
    print(f"[Diffusion Eval] checkpoint={cfg['dpf_ckpt']}")

    dpf, hnn, var_dq, var_dp, mj_model = qe.load_models(cfg, device)
    qpos_dim = cfg["qpos_dim"]
    guidance = resolve_default_guidance(SYSTEM, cfg)

    shared_kwargs = {}
    if qe.NUM_DIFFUSION_STEPS is not None:
        shared_kwargs["num_diffusion_steps"] = qe.NUM_DIFFUSION_STEPS

    ung_kwargs = dict(hnn=None, guidance_steps=0, smooth_sigma=qe.SMOOTH_SIGMA, **shared_kwargs)
    gui_kwargs = dict(
        hnn=hnn,
        guidance_method=guidance["guidance_method"],
        guidance_steps=guidance["guidance_steps"],
        guidance_lr=guidance["guidance_lr"],
        guidance_after_steps=guidance["guidance_after_steps"],
        guidance_before_steps=guidance["guidance_before_steps"],
        optimize_target=guidance["optimize_target"],
        guidance_energy_mode=guidance["guidance_energy_mode"],
        guidance_hamres_smooth_sigma=guidance["guidance_hamres_smooth_sigma"],
        guidance_hamres_delta=guidance["guidance_hamres_delta"],
        guidance_hamres_min_scale_q=guidance["guidance_hamres_min_scale_q"],
        guidance_hamres_min_scale_p=guidance["guidance_hamres_min_scale_p"],
        guidance_trust_lambda=guidance["guidance_trust_lambda"],
        smooth_sigma=qe.SMOOTH_SIGMA,
        smooth_guidance_only=qe.SMOOTH_GUIDANCE_ONLY,
        smooth_last_step_only=qe.SMOOTH_LAST_STEP_ONLY,
        alpha_q=qe.ALPHA_Q,
        alpha_p=qe.ALPHA_P,
        langevin_step_size=qe.LANGEVIN_STEP_SIZE,
        langevin_noise_scale=qe.LANGEVIN_NOISE_SCALE,
        chunk_length=qe.CHUNK_LENGTH,
        **shared_kwargs,
    )

    hamres_kwargs = {
        "smooth_sigma": HAMRES_SMOOTH_SIGMA if qe.HAMRES_SMOOTH_SIGMA_CFG is None else qe.HAMRES_SMOOTH_SIGMA_CFG,
        "delta": HAMRES_PSEUDO_HUBER_DELTA if qe.HAMRES_DELTA_CFG is None else qe.HAMRES_DELTA_CFG,
        "min_scale_q": HAMRES_MIN_SCALE_Q if qe.HAMRES_MIN_SCALE_Q_CFG is None else qe.HAMRES_MIN_SCALE_Q_CFG,
        "min_scale_p": HAMRES_MIN_SCALE_P if qe.HAMRES_MIN_SCALE_P_CFG is None else qe.HAMRES_MIN_SCALE_P_CFG,
    }

    manifest = {
        "system": SYSTEM,
        "policies": POLICIES,
        "lengths": LENGTHS,
        "num_samples": NUM_SAMPLES,
        "seed": SEED,
        "checkpoint": str(cfg["dpf_ckpt"]),
        "guidance_preset": qe.GUIDANCE_PRESET,
        "guidance_energy_mode": guidance["guidance_energy_mode"],
        "num_diffusion_steps": qe.NUM_DIFFUSION_STEPS,
        "context_fraction": 0.5,
        "protocol": "protocol2_generate_L1000_then_crop_to_L",
        "generation_length": 1000,
    }
    DIFF_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))

    fields = [
        "combo",
        "system",
        "policy",
        "length",
        "num_samples",
        "seed",
        "sample_idx",
        "ung_nrmse_q",
        "gui_nrmse_q",
        "d_nrmse_q",
        "ung_nrmse_p",
        "gui_nrmse_p",
        "d_nrmse_p",
        "ung_hamres",
        "gui_hamres",
        "d_hamres",
        "hnn_nrmse_q",
        "hnn_nrmse_p",
        "hnn_hamres",
    ]

    gen_length = max(LENGTHS)
    for policy in POLICIES:
        if policy in done:
            print(f"[Skip] policy={policy}")
            continue

        print(f"[Run] policy={policy} (generate L={gen_length}, then crop)")
        torques_full = load_torques(policy, NUM_SAMPLES, gen_length, device, cfg)
        qe.set_seed(SEED)
        noise_full = torch.randn(NUM_SAMPLES, gen_length, dpf.state_dim, device=device)

        ung_states_full, ung_torques_full = qe.run_batch(
            dpf, torques_full, noise_full, gen_length, BATCH_SIZE_UNGUIDED, **ung_kwargs
        )
        gui_states_full, gui_torques_full = qe.run_batch(
            dpf, torques_full, noise_full, gen_length, BATCH_SIZE_GUIDED, **gui_kwargs
        )
        hnn_states_full = run_hnn_rollout_batch(hnn, ung_states_full, torques_full)

        for length in LENGTHS:
            cid = combo_id(SYSTEM, policy, length, SEED, NUM_SAMPLES)
            ung_states = ung_states_full[:, :length]
            ung_torques = ung_torques_full[:, :length]
            gui_states = gui_states_full[:, :length]
            gui_torques = gui_torques_full[:, :length]
            hnn_states = hnn_states_full[:, :length]
            torques = torques_full[:, :length]

            for i in tqdm(range(NUM_SAMPLES), desc=f"metrics_{cid}"):
                uq, up, uh = compute_nrmse_range_and_hamres(
                    ung_states[i], ung_torques[i], qpos_dim, mj_model, hnn, var_dq, var_dp, hamres_kwargs
                )
                gq, gp, gh = compute_nrmse_range_and_hamres(
                    gui_states[i], gui_torques[i], qpos_dim, mj_model, hnn, var_dq, var_dp, hamres_kwargs
                )
                hq, hp, hh = compute_nrmse_range_and_hamres(
                    hnn_states[i], torques[i], qpos_dim, mj_model, hnn, var_dq, var_dp, hamres_kwargs
                )
                append_csv(
                    DIFF_PER_SAMPLE_CSV,
                    fields,
                    {
                        "combo": cid,
                        "system": SYSTEM,
                        "policy": policy,
                        "length": length,
                        "num_samples": NUM_SAMPLES,
                        "seed": SEED,
                        "sample_idx": i,
                        "ung_nrmse_q": uq,
                        "gui_nrmse_q": gq,
                        "d_nrmse_q": gq - uq,
                        "ung_nrmse_p": up,
                        "gui_nrmse_p": gp,
                        "d_nrmse_p": gp - up,
                        "ung_hamres": uh,
                        "gui_hamres": gh,
                        "d_hamres": gh - uh,
                        "hnn_nrmse_q": hq,
                        "hnn_nrmse_p": hp,
                        "hnn_hamres": hh,
                    },
                )

        done.add(policy)
        DIFF_DONE_JSON.write_text(json.dumps(sorted(done), indent=2))
        print(f"[Done] policy={policy}")


def load_dpf_table():
    ndf = pd.read_csv(DPF_NRMSE_CSV)
    hdf = pd.read_csv(DPF_HAMRES_CSV)

    ndf = ndf[(ndf["system"] == "2dof") & (ndf["length"] <= 1000)].copy()
    hdf = hdf[(hdf["system"] == "2dof") & (hdf["length"] <= 1000)].copy()

    keep_h = ["combo", "policy", "length", "seed", "sample_idx", "ung_hamres", "gui_hamres"]
    merged = pd.merge(
        ndf,
        hdf[keep_h],
        on=["combo", "policy", "length", "seed", "sample_idx"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def load_diffusion_table():
    if not DIFF_PER_SAMPLE_CSV.exists():
        raise FileNotFoundError(f"Missing diffusion metrics file: {DIFF_PER_SAMPLE_CSV}")
    df = pd.read_csv(DIFF_PER_SAMPLE_CSV)
    df = df[(df["system"] == "2dof") & (df["length"] <= 1000)].copy()
    return df


def make_policy_boxplots():
    dpf = load_dpf_table()
    diff = load_diffusion_table()
    lengths = sorted([l for l in dpf["length"].unique().tolist() if l <= 1000])

    methods = [
        ("unguided_dpf", "#D7A1B0"),  # dusty rose
        ("guided_dpf", "#A12C33"),    # deep crimson
        ("unguided_diffusion", "#BFD7EA"),  # light blue
        ("guided_diffusion", "#2F66B0"),    # blue
        ("hnn_rollout", "#D7B347"),   # muted gold
    ]
    metrics = [
        ("nrmse_q", r"$\mathrm{NRMSE}_q$"),
        ("nrmse_p", r"$\mathrm{NRMSE}_p$"),
        ("hamres", r"$\mathrm{HamRes}$"),
    ]

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 20,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    for policy in POLICIES:
        fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)

        sub_dpf = dpf[dpf["policy"] == policy]
        sub_diff = diff[diff["policy"] == policy]
        base = np.arange(len(lengths))
        width = 0.15
        offsets = [-2 * width, -1 * width, 0.0, 1 * width, 2 * width]

        for ax, (mkey, mlabel) in zip(axes, metrics):
            for (method_name, color), offset in zip(methods, offsets):
                vals = []
                for L in lengths:
                    if method_name == "unguided_dpf":
                        arr = sub_dpf[sub_dpf["length"] == L][f"ung_{mkey}"].to_numpy()
                    elif method_name == "guided_dpf":
                        arr = sub_dpf[sub_dpf["length"] == L][f"gui_{mkey}"].to_numpy()
                    elif method_name == "unguided_diffusion":
                        arr = sub_diff[sub_diff["length"] == L][f"ung_{mkey}"].to_numpy()
                    elif method_name == "guided_diffusion":
                        arr = sub_diff[sub_diff["length"] == L][f"gui_{mkey}"].to_numpy()
                    elif method_name == "hnn_rollout":
                        arr = sub_diff[sub_diff["length"] == L][f"hnn_{mkey}"].to_numpy()
                    else:
                        raise ValueError(method_name)
                    vals.append(arr)

                bp = ax.boxplot(
                    vals,
                    positions=base + offset,
                    widths=width * 0.95,
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="black", linewidth=2.4),
                    whiskerprops=dict(color=color, linewidth=2.0),
                    capprops=dict(color=color, linewidth=2.0),
                    boxprops=dict(facecolor=color, alpha=0.9, edgecolor=color, linewidth=2.2),
                )
                for b in bp["boxes"]:
                    b.set_facecolor(color)
                    b.set_alpha(0.9)

            ax.set_ylabel(mlabel)
            ax.set_title(f"2DoF | torque policy={policy}")

        axes[-1].set_xticks(base)
        tick_labels = [str(L) for L in lengths]
        axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right")
        axes[-1].set_xlabel("Trajectory Length")

        label_map = {
            "unguided_dpf": "Unguided DPF",
            "guided_dpf": "Guided DPF",
            "unguided_diffusion": "Unguided Diffusion",
            "guided_diffusion": "Guided Diffusion",
            "hnn_rollout": "HNN Rollout",
        }
        legend_handles = [mpatches.Patch(color=c, label=label_map[n]) for n, c in methods]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=5,
            framealpha=0.95,
            bbox_to_anchor=(0.5, 1.01),
        )
        for ax in axes:
            ax.tick_params(axis="both", which="major", labelsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = PLOTS_ROOT / f"boxplot_2dof_{policy}_dpf_vs_diffusion_protocol2.png"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"[Plot] saved {out}")


def main():
    run_diffusion_eval_if_needed()
    make_policy_boxplots()


if __name__ == "__main__":
    main()
