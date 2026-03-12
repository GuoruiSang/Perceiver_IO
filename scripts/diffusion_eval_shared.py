#!/usr/bin/env python3
"""Shared evaluation/plot pipeline for 2DoF and 3DoF method comparison."""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
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
sys.path.insert(0, str(project_root / "scripts"))

import scripts.guidance_eval_config as gc
import scripts.guidance_sampling_utils as gs
from scripts.system_eval_utils import (
    BATCH_SIZE_GUIDED,
    BATCH_SIZE_UNGUIDED,
    DT,
    HAMRES_MIN_SCALE_P,
    HAMRES_MIN_SCALE_Q,
    HAMRES_PSEUDO_HUBER_DELTA,
    HAMRES_SMOOTH_SIGMA,
    SIM_DT,
    SYSTEM_CONFIGS,
    compute_hamres,
    load_torques,
)
from src.models.utils import reconstruct_traj_with_momentum


@dataclass(frozen=True)
class SystemEvalConfig:
    system: str
    fixed_diffusion_ckpt_default: Path
    out_root_default: Path
    dpf_out_root_default: Path
    plot_filename_template: str
    plot_title_template: str
    title_each_axis: bool = False
    best_config_json_default: Path | None = None
    best_config_model_key_default: str = "fixed_diffusion"


def build_system_eval_config(system: str, project_root_override: Path | None = None) -> SystemEvalConfig:
    root = project_root if project_root_override is None else project_root_override
    if system == "2dof":
        return SystemEvalConfig(
            system="2dof",
            fixed_diffusion_ckpt_default=root
            / "checkpoints"
            / "2dof"
            / "transformer_diffusion"
            / "trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0009.ckpt",
            out_root_default=root / "output" / "eval_runs" / "2dof_transformer_eval_with_hnn",
            dpf_out_root_default=root / "output" / "eval_runs" / "default_full_sweep",
            plot_filename_template="boxplot_2dof_{policy}_dpf_vs_diffusion.png",
            plot_title_template="2DoF | torque policy={policy}",
            title_each_axis=False,
            best_config_json_default=None,
            best_config_model_key_default="fixed_diffusion",
        )
    if system == "3dof":
        return SystemEvalConfig(
            system="3dof",
            fixed_diffusion_ckpt_default=root
            / "checkpoints"
            / "3dof"
            / "transformer_diffusion"
            / "trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions_backbone-transformer:epoch=2999_val_loss:val_loss=0.0008.ckpt",
            out_root_default=root / "output" / "eval_runs" / "3dof_transformer_eval_with_hnn",
            dpf_out_root_default=root / "output" / "eval_runs" / "default_full_sweep_3dof_structhnn",
            plot_filename_template="boxplot_3dof_{policy}_dpf_vs_diffusion.png",
            plot_title_template="3DoF | torque policy={policy}",
            title_each_axis=True,
            best_config_json_default=None,
        )
    raise ValueError(f"Unsupported system={system!r}. Expected '2dof' or '3dof'.")


def _env_path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    return Path(v) if v else default


def run_system_eval(cfg_static: SystemEvalConfig) -> None:
    policies = [
        p.strip()
        for p in os.getenv("POLICIES_CSV", "sinusoidal,gp,zero,spline").split(",")
        if p.strip()
    ]
    lengths_csv = os.getenv("EVAL_LENGTHS_CSV", "").strip()
    if lengths_csv:
        lengths = sorted({int(v.strip()) for v in lengths_csv.split(",") if v.strip()})
    else:
        lengths = list(range(50, 1001, 50))
    num_samples = int(os.getenv("EVAL_NUM_SAMPLES", "100").strip() or "100")
    seed = int(os.getenv("EVAL_SEED", "10").strip() or "10")
    min_scale_q = 1e-3
    min_scale_p = 1e-3

    out_root = _env_path("OUT_ROOT", cfg_static.out_root_default)
    metrics_root = out_root / "metrics"
    plots_root = _env_path("PLOTS_ROOT", project_root / "plots")
    metrics_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    fixed_diffusion_ckpt = _env_path("FIXED_DIFFUSION_CKPT", cfg_static.fixed_diffusion_ckpt_default)
    hnn_ckpt_override = os.getenv("HNN_CKPT_3DOF", "").strip()
    diff_per_sample_csv = _env_path("DIFF_PER_SAMPLE_CSV", metrics_root / "metrics_per_sample_rmse_hamres.csv")
    diff_done_json = out_root / "completed_combos.json"
    diff_manifest_json = out_root / "run_manifest.json"
    dpf_out_root = _env_path("DPF_OUT_ROOT", cfg_static.dpf_out_root_default)
    dpf_hamres_csv = dpf_out_root / "metrics" / "metrics_per_sample.csv"

    # Optional override for one-step gradient step sizes.
    alpha_q_env = os.getenv("GUIDANCE_ALPHA_Q", "").strip()
    alpha_p_env = os.getenv("GUIDANCE_ALPHA_P", "").strip()
    if alpha_q_env:
        gc.ALPHA_Q = float(alpha_q_env)
    if alpha_p_env:
        gc.ALPHA_P = float(alpha_p_env)
    print(f"[Guidance] alpha_q={gc.ALPHA_Q} alpha_p={gc.ALPHA_P}")

    best_config_json = None
    best_config_model_key = cfg_static.best_config_model_key_default
    if cfg_static.best_config_json_default is not None:
        best_config_json = _env_path("BEST_CONFIG_JSON", cfg_static.best_config_json_default)
        best_config_model_key = os.getenv("BEST_CONFIG_MODEL_KEY", best_config_model_key)

    def run_hnn_rollout_batch(hnn, init_states, torques):
        q0 = init_states[:, 0, : init_states.shape[-1] // 2]
        p0 = init_states[:, 0, init_states.shape[-1] // 2 :]
        tau_seq = torques.permute(1, 0, 2)
        with torch.no_grad():
            p_traj, q_traj, _ = hnn.integrate_trajectory(
                p0=p0,
                q0=q0,
                tau_seq=tau_seq,
                dt=DT,
                num_steps=torques.shape[1] - 1,
            )
        q = q_traj.permute(1, 0, 2)
        p = p_traj.permute(1, 0, 2)
        return torch.cat([q, p], dim=-1)

    def combo_id(policy, length):
        return f"{cfg_static.system}_{policy}_L{length}_N{num_samples}_seed{seed}"

    def append_csv(path, fieldnames, row):
        new_file = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                w.writeheader()
            w.writerow(row)

    def maybe_override_guidance_with_best(guidance: dict) -> tuple[dict, str]:
        if best_config_json is None or not best_config_json.exists():
            return guidance, "guidance_eval_config_preset"

        try:
            payload = json.loads(best_config_json.read_text())
        except Exception as e:
            print(f"[Guidance] failed to read {best_config_json}: {e}")
            return guidance, "guidance_eval_config_preset"

        record = payload.get(best_config_model_key)
        if not record or not isinstance(record, dict):
            print(
                f"[Guidance] no key '{best_config_model_key}' in {best_config_json}; "
                "falling back to guidance preset config."
            )
            return guidance, "guidance_eval_config_preset"

        params = record.get("params", {}) or {}
        if not params:
            print(
                f"[Guidance] missing 'params' for key '{best_config_model_key}' in {best_config_json}; "
                "falling back to guidance preset config."
            )
            return guidance, "guidance_eval_config_preset"

        out = dict(guidance)
        out["guidance_method"] = params.get("strategy", "strategy2")
        out["optimize_target"] = "both"
        strategy = out["guidance_method"]
        default_mode = "robust_hamres" if strategy == "strategy1" else "one_step"
        out["guidance_energy_mode"] = params.get("mode", default_mode)
        out["guidance_num_candidates"] = int(params.get("num_candidates", out.get("guidance_num_candidates", 16)))
        out["guidance_trust_lambda"] = float(params.get("trust", out["guidance_trust_lambda"]))
        out["guidance_hamres_smooth_sigma"] = float(params.get("ham_sigma", out["guidance_hamres_smooth_sigma"]))
        out["guidance_hamres_delta"] = float(params.get("ham_delta", out["guidance_hamres_delta"]))
        out["guidance_hamres_min_scale_q"] = 1e-3
        out["guidance_hamres_min_scale_p"] = 1e-3

        gc.ALPHA_Q = float(params.get("alpha_q", gc.ALPHA_Q))
        gc.ALPHA_P = float(params.get("alpha_p", gc.ALPHA_P))
        print(
            f"[Guidance] override from {best_config_json} key={best_config_model_key}: "
            f"{record.get('candidate_id', '<unknown>')}"
        )
        return out, "best_configs_json"

    def compute_rmse_qp(state, tau, qpos_dim, mj_model):
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
            mj_model,
            t_len,
            SIM_DT,
            qpos[0],
            initial_qvel,
            tau_np,
            data_dt=DT,
            trajectory_alignment="pre_step",
        )
        gt_qpos = recon["seq_qpos"]
        gt_mom = recon["seq_mom"]

        t_min = min(len(qpos), len(gt_qpos))
        if t_min <= 0:
            return np.nan, np.nan

        e_q = qpos[:t_min] - gt_qpos[:t_min]
        e_p = mom[:t_min] - gt_mom[:t_min]
        rmse_q = np.sqrt((e_q**2).mean(axis=0))
        rmse_p = np.sqrt((e_p**2).mean(axis=0))
        return float(rmse_q.mean()), float(rmse_p.mean())

    def compute_rmse_and_hamres(state, tau, qpos_dim, mj_model, hnn, var_dq, var_dp, hamres_kwargs):
        rmse_q_mean, rmse_p_mean = compute_rmse_qp(state, tau, qpos_dim, mj_model)

        hr = compute_hamres(
            state[:, :qpos_dim],
            state[:, qpos_dim:],
            tau,
            hnn,
            var_dq,
            var_dp,
            **hamres_kwargs,
        )
        return rmse_q_mean, rmse_p_mean, float(hr)

    def build_dpf_rmse_csv() -> Path:
        out_csv = dpf_out_root / "metrics" / "rmse_range_per_sample_Lle1000.csv"
        if out_csv.exists():
            try:
                df_head = pd.read_csv(out_csv)
                cols = df_head.columns.tolist()
                if (
                    "ung_rmse_q" in cols
                    and "gui_rmse_q" in cols
                    and "ung_rmse_p" in cols
                    and "gui_rmse_p" in cols
                ):
                    return out_csv
            except Exception:
                pass

        cfg = SYSTEM_CONFIGS[cfg_static.system]
        qpos_dim = int(cfg["qpos_dim"])
        mj_model = mujoco.MjModel.from_xml_path(cfg["xml_path"])
        traj_root = dpf_out_root / "trajectories" / cfg_static.system
        if not traj_root.exists():
            raise FileNotFoundError(f"Missing DPF trajectory directory: {traj_root}")

        rows: list[dict] = []
        files = sorted(traj_root.glob("*/*.pt"))
        for pt_path in tqdm(files, desc=f"DPF one-step RMSE ({cfg_static.system})"):
            bundle = torch.load(pt_path, map_location="cpu", weights_only=False)
            length = int(bundle["length"])
            if length > 1000:
                continue

            combo = str(bundle["combo"])
            policy = str(bundle["policy"])
            seed = int(bundle["seed"])
            ung_states = bundle["unguided_states"]
            ung_torques = bundle["unguided_torques"]
            gui_states = bundle["guided_states"]
            gui_torques = bundle["guided_torques"]

            for sample_idx in range(ung_states.shape[0]):
                urq, urp = compute_rmse_qp(ung_states[sample_idx], ung_torques[sample_idx], qpos_dim, mj_model)
                grq, grp = compute_rmse_qp(gui_states[sample_idx], gui_torques[sample_idx], qpos_dim, mj_model)
                rows.append(
                    {
                        "combo": combo,
                        "system": cfg_static.system,
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
            raise RuntimeError(f"No DPF RMSE rows were generated for {cfg_static.system}.")

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        return out_csv

    def run_diffusion_eval_if_needed():
        expected_combos = len(policies)
        done = set()
        if diff_done_json.exists():
            done = set(json.loads(diff_done_json.read_text()))

        if len(done) >= expected_combos and diff_per_sample_csv.exists():
            print(f"[Skip] diffusion eval already complete ({len(done)}/{expected_combos} policies).")
            return

        cfg = dict(SYSTEM_CONFIGS[cfg_static.system])
        cfg["dpf_ckpt"] = fixed_diffusion_ckpt
        if not cfg["dpf_ckpt"].exists():
            raise FileNotFoundError(f"Fixed diffusion checkpoint not found: {cfg['dpf_ckpt']}")
        if cfg_static.system == "3dof" and hnn_ckpt_override:
            cfg["hnn_ckpt"] = Path(hnn_ckpt_override).expanduser().resolve()
            if not cfg["hnn_ckpt"].exists():
                raise FileNotFoundError(f"Override HNN checkpoint not found: {cfg['hnn_ckpt']}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gs.set_seed(seed)
        print(f"[Diffusion Eval] device={device}")
        print(f"[Diffusion Eval] checkpoint={cfg['dpf_ckpt']}")
        print(f"[Diffusion Eval] hnn_ckpt={cfg['hnn_ckpt']}")

        dpf, hnn, var_dq, var_dp, mj_model = gs.load_models(cfg, device)
        qpos_dim = cfg["qpos_dim"]
        guidance, _ = gc.resolve_guidance_config(
            system=cfg_static.system,
            auto_map={"2dof": "robust_r9_comboa", "3dof": "one_step_best"},
        )
        guidance, guidance_source = maybe_override_guidance_with_best(guidance)

        ung_kwargs = gc.build_unguided_kwargs()
        gui_kwargs = gc.build_guided_kwargs(hnn, guidance)
        hamres_kwargs = gc.build_hamres_eval_kwargs()

        manifest = {
            "system": cfg_static.system,
            "policies": policies,
            "lengths": lengths,
            "num_samples": num_samples,
            "seed": seed,
            "checkpoint": str(cfg["dpf_ckpt"]),
            "hnn_checkpoint": str(cfg["hnn_ckpt"]),
            "guidance_preset": gc.GUIDANCE_PRESET,
            "guidance_source": guidance_source,
            "best_config_json": str(best_config_json) if (best_config_json and best_config_json.exists()) else "",
            "best_config_model_key": best_config_model_key if best_config_json else "",
            "guidance_method": guidance["guidance_method"],
            "guidance_num_candidates": guidance["guidance_num_candidates"],
            "comparison_label": "resampling" if guidance["guidance_method"] == "strategy1" else "guided",
            "guidance_energy_mode": guidance["guidance_energy_mode"],
            "alpha_q": float(gc.ALPHA_Q),
            "alpha_p": float(gc.ALPHA_P),
            "num_diffusion_steps": gc.NUM_DIFFUSION_STEPS,
            "context_fraction": 0.5,
            "protocol": "generate_L1000_then_crop_to_L",
            "generation_length": 1000,
        }
        diff_manifest_json.write_text(json.dumps(manifest, indent=2))

        fields = [
            "combo",
            "system",
            "policy",
            "length",
            "num_samples",
            "seed",
            "sample_idx",
            "ung_rmse_q",
            "gui_rmse_q",
            "d_rmse_q",
            "ung_rmse_p",
            "gui_rmse_p",
            "d_rmse_p",
            "hnn_rmse_q",
            "hnn_rmse_p",
            "ung_hamres",
            "gui_hamres",
            "d_hamres",
            "hnn_hamres",
        ]

        gen_length = max(lengths)
        for policy in policies:
            if policy in done:
                print(f"[Skip] policy={policy}")
                continue

            print(f"[Run] policy={policy} (generate L={gen_length}, then crop)")
            torques_full = load_torques(policy, num_samples, gen_length, device, cfg)
            gs.set_seed(seed)
            noise_full = torch.randn(num_samples, gen_length, dpf.state_dim, device=device)

            ung_states_full, ung_torques_full = gs.run_batch(
                dpf, torques_full, noise_full, gen_length, BATCH_SIZE_UNGUIDED, **ung_kwargs
            )
            gui_states_full, gui_torques_full = gs.run_batch(
                dpf, torques_full, noise_full, gen_length, BATCH_SIZE_GUIDED, **gui_kwargs
            )
            hnn_states_full = run_hnn_rollout_batch(hnn, ung_states_full, torques_full)

            for length in lengths:
                cid = combo_id(policy, length)
                ung_states = ung_states_full[:, :length]
                ung_torques = ung_torques_full[:, :length]
                gui_states = gui_states_full[:, :length]
                gui_torques = gui_torques_full[:, :length]
                hnn_states = hnn_states_full[:, :length]
                torques = torques_full[:, :length]

                for i in tqdm(range(num_samples), desc=f"metrics_{cid}"):
                    urq, urp, uh = compute_rmse_and_hamres(
                        ung_states[i], ung_torques[i], qpos_dim, mj_model, hnn, var_dq, var_dp, hamres_kwargs
                    )
                    grq, grp, gh = compute_rmse_and_hamres(
                        gui_states[i], gui_torques[i], qpos_dim, mj_model, hnn, var_dq, var_dp, hamres_kwargs
                    )
                    hrq, hrp, hh = compute_rmse_and_hamres(
                        hnn_states[i], torques[i], qpos_dim, mj_model, hnn, var_dq, var_dp, hamres_kwargs
                    )
                    append_csv(
                        diff_per_sample_csv,
                        fields,
                        {
                            "combo": cid,
                            "system": cfg_static.system,
                            "policy": policy,
                            "length": length,
                            "num_samples": num_samples,
                            "seed": seed,
                            "sample_idx": i,
                            "ung_rmse_q": urq,
                            "gui_rmse_q": grq,
                            "d_rmse_q": grq - urq,
                            "ung_rmse_p": urp,
                            "gui_rmse_p": grp,
                            "d_rmse_p": grp - urp,
                            "hnn_rmse_q": hrq,
                            "hnn_rmse_p": hrp,
                            "ung_hamres": uh,
                            "gui_hamres": gh,
                            "d_hamres": gh - uh,
                            "hnn_hamres": hh,
                        },
                    )

            done.add(policy)
            diff_done_json.write_text(json.dumps(sorted(done), indent=2))
            print(f"[Done] policy={policy}")

    def load_dpf_table():
        rdf = pd.read_csv(build_dpf_rmse_csv())
        hdf = pd.read_csv(dpf_hamres_csv)
        rdf = rdf[(rdf["system"] == cfg_static.system) & (rdf["length"] <= 1000)].copy()
        hdf = hdf[(hdf["system"] == cfg_static.system) & (hdf["length"] <= 1000)].copy()
        keep_h = ["combo", "policy", "length", "seed", "sample_idx", "ung_hamres", "gui_hamres"]
        return pd.merge(
            rdf,
            hdf[keep_h],
            on=["combo", "policy", "length", "seed", "sample_idx"],
            how="inner",
            validate="one_to_one",
        )

    def load_diffusion_table():
        if not diff_per_sample_csv.exists():
            raise FileNotFoundError(f"Missing diffusion metrics file: {diff_per_sample_csv}")
        df = pd.read_csv(diff_per_sample_csv)
        return df[(df["system"] == cfg_static.system) & (df["length"] <= 1000)].copy()

    def make_policy_boxplots():
        dpf = load_dpf_table()
        diff = load_diffusion_table()
        valid_lengths = sorted([l for l in dpf["length"].unique().tolist() if l <= 1000])
        comparison_prefix = "Guided"
        if diff_manifest_json.exists():
            try:
                m = json.loads(diff_manifest_json.read_text())
                if m.get("guidance_method") == "strategy1":
                    comparison_prefix = "Resampling"
            except Exception:
                pass

        methods = [
            ("unguided_dpf", "#D7A1B0"),
            ("guided_dpf", "#A12C33"),
            ("unguided_diffusion", "#BFD7EA"),
            ("guided_diffusion", "#2F66B0"),
            ("hnn_rollout", "#D7B347"),
        ]
        metrics = [
            ("rmse_q", r"$\mathrm{RMSE}_q$"),
            ("rmse_p", r"$\mathrm{RMSE}_p$"),
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

        for policy in policies:
            fig, axes = plt.subplots(3, 1, figsize=(20, 14), sharex=True)
            sub_dpf = dpf[dpf["policy"] == policy]
            sub_diff = diff[diff["policy"] == policy]
            base = np.arange(len(valid_lengths))
            width = 0.15
            offsets = [-2 * width, -1 * width, 0.0, 1 * width, 2 * width]

            for ax, (mkey, mlabel) in zip(axes, metrics):
                for (method_name, color), offset in zip(methods, offsets):
                    vals = []
                    for L in valid_lengths:
                        if method_name == "unguided_dpf":
                            arr = sub_dpf[sub_dpf["length"] == L][f"ung_{mkey}"].to_numpy()
                        elif method_name == "guided_dpf":
                            arr = sub_dpf[sub_dpf["length"] == L][f"gui_{mkey}"].to_numpy()
                        elif method_name == "unguided_diffusion":
                            arr = sub_diff[sub_diff["length"] == L][f"ung_{mkey}"].to_numpy()
                        elif method_name == "guided_diffusion":
                            arr = sub_diff[sub_diff["length"] == L][f"gui_{mkey}"].to_numpy()
                        else:
                            arr = sub_diff[sub_diff["length"] == L][f"hnn_{mkey}"].to_numpy()
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
                if cfg_static.title_each_axis:
                    ax.set_title(cfg_static.plot_title_template.format(policy=policy))

            if not cfg_static.title_each_axis:
                axes[0].set_title(cfg_static.plot_title_template.format(policy=policy))

            axes[-1].set_xticks(base)
            axes[-1].set_xticklabels([str(L) for L in valid_lengths], rotation=45, ha="right")
            axes[-1].set_xlabel("Trajectory Length")

            label_map = {
                "unguided_dpf": "Unguided DPF",
                "guided_dpf": f"{comparison_prefix} DPF",
                "unguided_diffusion": "Unguided Diffusion",
                "guided_diffusion": f"{comparison_prefix} Diffusion",
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
            out = plots_root / cfg_static.plot_filename_template.format(policy=policy)
            fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            print(f"[Plot] saved {out}")

    run_diffusion_eval_if_needed()
    make_policy_boxplots()
