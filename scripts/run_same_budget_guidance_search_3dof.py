#!/usr/bin/env python3
"""Same-budget guidance search for 3DoF on DPF vs fixed-length diffusion.

Goal:
  - Use the same guidance search space and same evaluation budget.
  - Pick best config for each model family (DPF, fixed diffusion) by NRMSE_q/p.

Output:
  - output_ablation/same_budget_guidance_search_3dof/search_detail.csv
  - output_ablation/same_budget_guidance_search_3dof/summary_by_candidate.csv
  - output_ablation/same_budget_guidance_search_3dof/best_configs.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import scripts.quick_eval as qe
from scripts.compute_ablation_2dof_with_smoothing import (
    SYSTEM_CONFIGS,
    BATCH_SIZE_GUIDED,
    BATCH_SIZE_UNGUIDED,
    load_torques,
)
from src.models.utils import reconstruct_traj_with_momentum


EPS = 1e-12
MIN_SCALE_Q = 1e-3
MIN_SCALE_P = 1e-3


@dataclass
class ModelSpec:
    name: str
    dpf_ckpt: Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cfg_id(cfg: dict) -> str:
    keys = [
        "mode",
        "steps",
        "after",
        "before",
        "alpha_q",
        "alpha_p",
        "trust",
        "ham_sigma",
        "ham_delta",
    ]
    out = []
    for k in keys:
        if k in cfg:
            out.append(f"{k}={cfg[k]}")
    return "|".join(out)


def build_candidate_pool() -> list[dict]:
    cands: list[dict] = []

    # One-step family.
    for steps in [30, 50]:
        for after in [10, 15]:
            for alpha in [0.005, 0.01]:
                for trust in [0.0, 1e-3]:
                    cands.append(
                        dict(
                            mode="one_step",
                            steps=steps,
                            after=after,
                            before=20,
                            alpha_q=alpha,
                            alpha_p=alpha,
                            trust=trust,
                        )
                    )

    # Robust HamRes family.
    for steps in [30, 50]:
        for after in [10, 15]:
            for alpha in [0.005, 0.01]:
                for trust in [0.0, 1e-3]:
                    for ham_sigma in [0.5, 1.0]:
                        for ham_delta in [1.0, 2.0]:
                            cands.append(
                                dict(
                                    mode="robust_hamres",
                                    steps=steps,
                                    after=after,
                                    before=20,
                                    alpha_q=alpha,
                                    alpha_p=alpha,
                                    trust=trust,
                                    ham_sigma=ham_sigma,
                                    ham_delta=ham_delta,
                                )
                            )
    return cands


def load_model(cfg: dict, device: torch.device):
    dpf, hnn, _, _, mj_model = qe.load_models(cfg, device)
    return dpf, hnn, mj_model, cfg["qpos_dim"]


def guidance_kwargs(c: dict, hnn, num_diffusion_steps: int | None):
    kw = dict(
        hnn=hnn,
        guidance_method="normalized_sgd",
        guidance_steps=int(c["steps"]),
        guidance_lr=0.01,
        guidance_after_steps=int(c["after"]),
        guidance_before_steps=int(c["before"]),
        optimize_target="both",
        guidance_energy_mode=c["mode"],
        guidance_hamres_smooth_sigma=float(c.get("ham_sigma", 1.0)),
        guidance_hamres_delta=float(c.get("ham_delta", 1.0)),
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
        guidance_trust_lambda=float(c.get("trust", 0.0)),
        smooth_sigma=0,
        smooth_guidance_only=False,
        smooth_last_step_only=False,
        alpha_q=float(c["alpha_q"]),
        alpha_p=float(c["alpha_p"]),
    )
    if num_diffusion_steps is not None:
        kw["num_diffusion_steps"] = int(num_diffusion_steps)
    return kw


def unguided_kwargs(num_diffusion_steps: int | None):
    kw = dict(hnn=None, guidance_steps=0, smooth_sigma=0)
    if num_diffusion_steps is not None:
        kw["num_diffusion_steps"] = int(num_diffusion_steps)
    return kw


def compute_nrmse_qp(state, torque, qpos_dim: int, mj_model):
    qpos = state[:, :qpos_dim].cpu().numpy()
    mom = state[:, qpos_dim:].cpu().numpy()
    tau = torque.cpu().numpy()
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
        qe.SIM_DT if hasattr(qe, "SIM_DT") else 0.0001,
        qpos[0],
        initial_qvel,
        tau,
        data_dt=qe.DT if hasattr(qe, "DT") else 0.0002,
    )
    gt_qpos = recon["seq_qpos"]
    gt_mom = recon["seq_mom"]

    t_min = min(len(qpos) - 1, len(gt_qpos))
    if t_min <= 0:
        return np.nan, np.nan

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
    return nrmse_q, nrmse_p


def evaluate_setting(
    dpf,
    hnn,
    mj_model,
    qpos_dim: int,
    policy: str,
    length: int,
    num_samples: int,
    seed: int,
    cfg: dict,
    candidates: list[dict],
    num_diffusion_steps: int | None,
):
    set_seed(seed)
    torques = load_torques(policy, num_samples, length, qe.DEVICE, cfg)
    noise = torch.randn(num_samples, length, dpf.state_dim, device=qe.DEVICE)

    ung_states, ung_torques = qe.run_batch(
        dpf, torques, noise, length, BATCH_SIZE_UNGUIDED, **unguided_kwargs(num_diffusion_steps)
    )

    ung_q = np.empty(num_samples, dtype=float)
    ung_p = np.empty(num_samples, dtype=float)
    for i in range(num_samples):
        uq, up = compute_nrmse_qp(ung_states[i], ung_torques[i], qpos_dim, mj_model)
        ung_q[i] = uq
        ung_p[i] = up

    per_candidate_rows = []
    for c in candidates:
        gui_states, gui_torques = qe.run_batch(
            dpf,
            torques,
            noise,
            length,
            BATCH_SIZE_GUIDED,
            **guidance_kwargs(c, hnn, num_diffusion_steps),
        )
        gui_q = np.empty(num_samples, dtype=float)
        gui_p = np.empty(num_samples, dtype=float)
        for i in range(num_samples):
            gq, gp = compute_nrmse_qp(gui_states[i], gui_torques[i], qpos_dim, mj_model)
            gui_q[i] = gq
            gui_p[i] = gp

        med_ung_q = float(np.median(ung_q))
        med_ung_p = float(np.median(ung_p))
        med_gui_q = float(np.median(gui_q))
        med_gui_p = float(np.median(gui_p))
        ratio_q = med_gui_q / max(med_ung_q, EPS)
        ratio_p = med_gui_p / max(med_ung_p, EPS)
        prob_q = float(np.mean(gui_q < ung_q))
        prob_p = float(np.mean(gui_p < ung_p))

        per_candidate_rows.append(
            dict(
                candidate_id=cfg_id(c),
                policy=policy,
                length=length,
                seed=seed,
                med_ung_nrmse_q=med_ung_q,
                med_gui_nrmse_q=med_gui_q,
                med_ung_nrmse_p=med_ung_p,
                med_gui_nrmse_p=med_gui_p,
                ratio_q=ratio_q,
                ratio_p=ratio_p,
                prob_q=prob_q,
                prob_p=prob_p,
            )
        )
    return per_candidate_rows


def pick_best(summary_rows: list[dict]) -> dict | None:
    feasible = [
        r
        for r in summary_rows
        if r["median_ratio_q"] < 1.0
        and r["median_ratio_p"] < 1.0
        and r["mean_prob_q"] >= 0.55
        and r["mean_prob_p"] >= 0.55
    ]
    pool = feasible if feasible else summary_rows
    if not pool:
        return None

    # Lower is better.
    def score(r):
        gmean = math.sqrt(max(r["median_ratio_q"], EPS) * max(r["median_ratio_p"], EPS))
        stability_penalty = 0.2 * ((1.0 - r["mean_prob_q"]) + (1.0 - r["mean_prob_p"]))
        return gmean + stability_penalty

    return sorted(pool, key=score)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="3dof", choices=["3dof"])
    parser.add_argument("--policies", default="sinusoidal,gp,zero,spline")
    parser.add_argument("--lengths", default="100,300,500,700,900,1000")
    parser.add_argument("--seeds", default="10,11,12")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--num-candidates", type=int, default=24)
    parser.add_argument("--num-diff-steps", type=int, default=20)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument(
        "--fixed-diff-ckpt",
        default="checkpoints/trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&FixedTrajLength1000&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0008.ckpt",
    )
    parser.add_argument(
        "--out-dir",
        default="output_ablation/same_budget_guidance_search_3dof",
    )
    args = parser.parse_args()

    policies = [x.strip() for x in args.policies.split(",") if x.strip()]
    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    qe.DEVICE = str(device)
    print(f"[search] device={device}")

    cfg_base = dict(SYSTEM_CONFIGS["3dof"])
    spec_dpf = ModelSpec(name="dpf", dpf_ckpt=Path(cfg_base["dpf_ckpt"]))
    spec_fix = ModelSpec(name="fixed_diffusion", dpf_ckpt=(project_root / args.fixed_diff_ckpt).resolve())

    all_cands = build_candidate_pool()
    rng = random.Random(args.search_seed)
    rng.shuffle(all_cands)
    candidates = all_cands[: min(args.num_candidates, len(all_cands))]
    print(f"[search] candidates={len(candidates)} (from pool={len(all_cands)})")

    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = out_dir / "search_detail.csv"
    summary_csv = out_dir / "summary_by_candidate.csv"
    best_json = out_dir / "best_configs.json"
    manifest_json = out_dir / "manifest.json"

    manifest = dict(
        system="3dof",
        policies=policies,
        lengths=lengths,
        seeds=seeds,
        num_samples=args.num_samples,
        num_candidates=len(candidates),
        num_diff_steps=args.num_diff_steps,
        search_seed=args.search_seed,
        candidates=[dict(id=cfg_id(c), **c) for c in candidates],
        models={
            "dpf": str(spec_dpf.dpf_ckpt),
            "fixed_diffusion": str(spec_fix.dpf_ckpt),
        },
    )
    manifest_json.write_text(json.dumps(manifest, indent=2))

    detail_fields = [
        "model",
        "candidate_id",
        "policy",
        "length",
        "seed",
        "med_ung_nrmse_q",
        "med_gui_nrmse_q",
        "med_ung_nrmse_p",
        "med_gui_nrmse_p",
        "ratio_q",
        "ratio_p",
        "prob_q",
        "prob_p",
    ]
    summary_fields = [
        "model",
        "candidate_id",
        "median_ratio_q",
        "median_ratio_p",
        "mean_prob_q",
        "mean_prob_p",
        "num_settings",
    ]

    with detail_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=detail_fields)
        w.writeheader()
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()

    best_out = {}
    for spec in [spec_dpf, spec_fix]:
        print(f"\n[search] model={spec.name} ckpt={spec.dpf_ckpt}")
        cfg = dict(cfg_base)
        cfg["dpf_ckpt"] = spec.dpf_ckpt
        dpf, hnn, mj_model, qpos_dim = load_model(cfg, device)

        detail_rows = []
        setting_grid = [(p, l, s) for p in policies for l in lengths for s in seeds]
        for policy, length, seed in tqdm(setting_grid, desc=f"{spec.name}_settings"):
            rows = evaluate_setting(
                dpf=dpf,
                hnn=hnn,
                mj_model=mj_model,
                qpos_dim=qpos_dim,
                policy=policy,
                length=length,
                num_samples=args.num_samples,
                seed=seed,
                cfg=cfg,
                candidates=candidates,
                num_diffusion_steps=args.num_diff_steps,
            )
            for r in rows:
                r["model"] = spec.name
                detail_rows.append(r)

        with detail_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=detail_fields)
            w.writerows(detail_rows)

        grouped = defaultdict(list)
        for r in detail_rows:
            grouped[r["candidate_id"]].append(r)

        summary_rows = []
        for cid, rows in grouped.items():
            ratio_q = np.array([x["ratio_q"] for x in rows], dtype=float)
            ratio_p = np.array([x["ratio_p"] for x in rows], dtype=float)
            prob_q = np.array([x["prob_q"] for x in rows], dtype=float)
            prob_p = np.array([x["prob_p"] for x in rows], dtype=float)
            summary_rows.append(
                dict(
                    model=spec.name,
                    candidate_id=cid,
                    median_ratio_q=float(np.median(ratio_q)),
                    median_ratio_p=float(np.median(ratio_p)),
                    mean_prob_q=float(np.mean(prob_q)),
                    mean_prob_p=float(np.mean(prob_p)),
                    num_settings=len(rows),
                )
            )

        with summary_csv.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_fields)
            w.writerows(summary_rows)

        best = pick_best(summary_rows)
        if best is None:
            best_out[spec.name] = None
            continue
        cand = next(c for c in candidates if cfg_id(c) == best["candidate_id"])
        best_out[spec.name] = {
            "candidate_id": best["candidate_id"],
            "params": cand,
            "metrics": best,
        }
        print(f"[search] best[{spec.name}]={best['candidate_id']}")

    best_json.write_text(json.dumps(best_out, indent=2))
    print(f"\n[done] detail={detail_csv}")
    print(f"[done] summary={summary_csv}")
    print(f"[done] best={best_json}")


if __name__ == "__main__":
    main()
