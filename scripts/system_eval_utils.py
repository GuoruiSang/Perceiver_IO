"""Shared system/evaluation helpers for the active experiment path."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import h5py
import mujoco
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.models.HNN import HNNWrapper
from src.models.trajectory_dpf import TrajectoryDPF
from src.models.utils import EMA, reconstruct_traj_with_momentum


DT = 0.0002
SIM_DT = 0.0001
BATCH_SIZE_UNGUIDED = 200
BATCH_SIZE_GUIDED = 50

HAMRES_SMOOTH_SIGMA = 1.0
HAMRES_PSEUDO_HUBER_DELTA = 1.0
HAMRES_MIN_SCALE_Q = 1e-3
HAMRES_MIN_SCALE_P = 1e-3

TORQUE_PATHS_BY_SYSTEM = {
    "2dof": {
        "sinusoidal": project_root / "data" / "2dof" / "sinusoidal" / "sinusoidal_torques_2000_L1500.h5",
        "gp": project_root / "data" / "2dof" / "gp" / "gp_torques_1000_L1500_intermediate_v1.h5",
        "spline": project_root / "data" / "2dof" / "spline" / "spline_torques_1000_L1500_intermediate_v1.h5",
        "zero": None,
    },
    "3dof": {
        "sinusoidal": project_root / "data" / "3dof" / "sinusoidal" / "sinusoidal_torques_1000_L1500.h5",
        "gp": project_root / "data" / "3dof" / "gp" / "gp_torques_1000_L1500.h5",
        "spline": project_root / "data" / "3dof" / "spline" / "spline_torques_1000_L1500.h5",
        "zero": None,
    },
}

SYSTEM_CONFIGS = {
    "2dof": {
        "system_name": "2dof",
        "dpf_ckpt": project_root
        / "checkpoints"
        / "2dof"
        / "dpf"
        / "trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0008.ckpt",
        "hnn_ckpt": project_root
        / "checkpoints"
        / "2dof"
        / "hnn"
        / "StructuredHNN-2DOF-epoch-epoch=999.ckpt",
        "xml_path": str(project_root / "configs" / "rigid_arm_hinge_2dof.xml"),
        "qpos_dim": 2,
        "torque_dim": 2,
    },
    "3dof": {
        "system_name": "3dof",
        "dpf_ckpt": project_root
        / "checkpoints"
        / "3dof"
        / "dpf"
        / "trajectory_dpf_StateOnlyAdaLN_x0Stabilized&AbsoluteTimeEncoding&VariableTrajLength&UniformContext&EncoderNone&DecoderAttentions:epoch=2999_val_loss:val_loss=0.0010.ckpt",
        "hnn_ckpt": project_root
        / "checkpoints"
        / "3dof"
        / "hnn"
        / "StructuredHNN-dim256-traj40000-epoch-epoch=999.ckpt",
        "xml_path": str(project_root / "configs" / "rigid_arm_hinge.xml"),
        "qpos_dim": 3,
        "torque_dim": 3,
    },
}


def load_dpf_with_ema(ckpt_path, device):
    dpf = TrajectoryDPF.load_from_checkpoint(str(ckpt_path), map_location=device, strict=False)
    dpf.eval().to(device)

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if "ema_shadow" in checkpoint:
        dpf.ema = EMA(dpf.model, decay=0.9995)
        for name, tensor in checkpoint["ema_shadow"].items():
            if name in dpf.ema.shadow:
                dpf.ema.shadow[name] = tensor.to(device=device, dtype=dpf.ema.shadow[name].dtype)
    del checkpoint
    return dpf


def load_hnn_with_stats(ckpt_path, device):
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    hnn_kwargs = {}
    if "model_type" not in hparams:
        hnn_kwargs["model_type"] = "separable"
    del checkpoint

    hnn = HNNWrapper.load_from_checkpoint(str(ckpt_path), map_location=device, strict=False, **hnn_kwargs)
    hnn.eval().to(device)
    var_dq = hnn.qvel_var.mean().to(device)
    var_dp = hnn.mom_dot_var.mean().to(device)
    return hnn, var_dq, var_dp


def load_system_models(cfg, device):
    dpf = load_dpf_with_ema(cfg["dpf_ckpt"], device)
    hnn, var_dq, var_dp = load_hnn_with_stats(cfg["hnn_ckpt"], device)
    mj_model = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    mj_model.opt.timestep = SIM_DT
    return dpf, hnn, var_dq, var_dp, mj_model


def load_torques(policy, num_samples, max_length, device, cfg):
    torque_dim = cfg["torque_dim"]
    if policy == "zero":
        return torch.zeros(num_samples, max_length, torque_dim, device=device)
    system_name = cfg.get("system_name")
    if system_name not in TORQUE_PATHS_BY_SYSTEM:
        raise KeyError(f"Unknown system_name in cfg: {system_name}")
    path = TORQUE_PATHS_BY_SYSTEM[system_name][policy]
    with h5py.File(path, "r") as f:
        torques = f["torques"][:num_samples, :max_length, :torque_dim]
    return torch.tensor(torques, dtype=torch.float32, device=device)


def _gaussian_smooth_1d(seq: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return seq
    radius = max(1, int(round(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=seq.device, dtype=seq.dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()
    seq_nct = seq.transpose(0, 1).unsqueeze(0)
    seq_pad = F.pad(seq_nct, (radius, radius), mode="reflect")
    weight = kernel.view(1, 1, -1).repeat(seq.shape[-1], 1, 1)
    smoothed = F.conv1d(seq_pad, weight, groups=seq.shape[-1])
    return smoothed.squeeze(0).transpose(0, 1)


def _pseudo_huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    return (delta**2) * (torch.sqrt(1.0 + (x / delta) ** 2) - 1.0)


def _dim_scale_from_var(var_like, dim: int, device, dtype):
    if var_like is None:
        return None
    var_t = torch.as_tensor(var_like, device=device, dtype=dtype).flatten()
    if var_t.numel() == 1:
        return torch.sqrt(torch.clamp(var_t.repeat(dim), min=0.0))
    if var_t.numel() >= dim:
        return torch.sqrt(torch.clamp(var_t[:dim], min=0.0))
    return None


def compute_hamres(
    qpos,
    mom,
    torque,
    hnn,
    var_dq,
    var_dp,
    dt=DT,
    smooth_sigma=HAMRES_SMOOTH_SIGMA,
    delta=HAMRES_PSEUDO_HUBER_DELTA,
    min_scale_q=HAMRES_MIN_SCALE_Q,
    min_scale_p=HAMRES_MIN_SCALE_P,
):
    t_len = qpos.shape[0]
    if t_len < 3:
        return float("nan")

    q_use = _gaussian_smooth_1d(qpos, smooth_sigma)
    p_use = _gaussian_smooth_1d(mom, smooth_sigma)

    qdot = (q_use[2:] - q_use[:-2]) / (2 * dt)
    pdot = (p_use[2:] - p_use[:-2]) / (2 * dt)
    q_mid, p_mid, tau_mid = q_use[1:-1], p_use[1:-1], torque[1:-1]
    with torch.inference_mode(False):
        with torch.enable_grad():
            p_grad = p_mid.detach().clone().requires_grad_(True)
            q_grad = q_mid.detach().clone().requires_grad_(True)
            h_val = hnn(p_grad, q_grad)
            dH_dp, dH_dq = torch.autograd.grad(h_val.sum(), (p_grad, q_grad), create_graph=False)

    r_q = qdot - dH_dp
    r_p = pdot - (-dH_dq + tau_mid)

    q_dim, p_dim = r_q.shape[-1], r_p.shape[-1]
    scale_q = _dim_scale_from_var(var_dq, q_dim, r_q.device, r_q.dtype)
    scale_p = _dim_scale_from_var(var_dp, p_dim, r_p.device, r_p.dtype)
    if scale_q is None:
        scale_q = torch.sqrt(torch.clamp(qdot.var(dim=0, unbiased=False), min=0.0))
    if scale_p is None:
        scale_p = torch.sqrt(torch.clamp(pdot.var(dim=0, unbiased=False), min=0.0))
    scale_q = torch.clamp(scale_q, min=min_scale_q)
    scale_p = torch.clamp(scale_p, min=min_scale_p)

    r_q_norm = r_q / scale_q.unsqueeze(0)
    r_p_norm = r_p / scale_p.unsqueeze(0)
    per_t = _pseudo_huber(r_q_norm, delta).mean(dim=-1) + _pseudo_huber(r_p_norm, delta).mean(dim=-1)
    return per_t.median().item()


def compute_nmse(state, torque, mj_model, qpos_dim, dt=DT, sim_dt=SIM_DT):
    qpos = state[:, :qpos_dim].cpu().numpy()
    mom = state[:, qpos_dim:].cpu().numpy()
    torque_np = torque.cpu().numpy()
    t_len = qpos.shape[0]

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)
    m_mat = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, m_mat, data.qM)
    initial_qvel = np.linalg.solve(m_mat, mom[0])

    recon = reconstruct_traj_with_momentum(
        mj_model,
        t_len,
        sim_dt,
        qpos[0],
        initial_qvel,
        torque_np,
        data_dt=dt,
        trajectory_alignment="pre_step",
    )
    gt_qpos = recon["seq_qpos"]
    gt_mom = recon["seq_mom"]

    t_min = min(len(qpos), len(gt_qpos))
    mse_q_per_dim = ((qpos[:t_min] - gt_qpos[:t_min]) ** 2).mean(axis=0)
    mse_p_per_dim = ((mom[:t_min] - gt_mom[:t_min]) ** 2).mean(axis=0)
    var_q_per_dim = np.var(gt_qpos[:t_min], axis=0)
    var_p_per_dim = np.var(gt_mom[:t_min], axis=0)
    nmse_q_per_dim = np.where(var_q_per_dim > 1e-12, mse_q_per_dim / var_q_per_dim, 0.0)
    nmse_p_per_dim = np.where(var_p_per_dim > 1e-12, mse_p_per_dim / var_p_per_dim, 0.0)
    return float(nmse_q_per_dim.mean()), float(nmse_p_per_dim.mean()), nmse_q_per_dim, nmse_p_per_dim


def compute_rmse(state, torque, mj_model, qpos_dim, dt=DT, sim_dt=SIM_DT):
    qpos = state[:, :qpos_dim].cpu().numpy()
    mom = state[:, qpos_dim:].cpu().numpy()
    torque_np = torque.cpu().numpy()
    t_len = qpos.shape[0]

    data = mujoco.MjData(mj_model)
    data.qpos[:] = qpos[0]
    data.qvel[:] = 0
    mujoco.mj_forward(mj_model, data)
    m_mat = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, m_mat, data.qM)
    initial_qvel = np.linalg.solve(m_mat, mom[0])

    recon = reconstruct_traj_with_momentum(
        mj_model,
        t_len,
        sim_dt,
        qpos[0],
        initial_qvel,
        torque_np,
        data_dt=dt,
        trajectory_alignment="pre_step",
    )
    gt_qpos = recon["seq_qpos"]
    gt_mom = recon["seq_mom"]

    t_min = min(len(qpos), len(gt_qpos))
    rmse_q_per_dim = np.sqrt(((qpos[:t_min] - gt_qpos[:t_min]) ** 2).mean(axis=0))
    rmse_p_per_dim = np.sqrt(((mom[:t_min] - gt_mom[:t_min]) ** 2).mean(axis=0))
    return float(rmse_q_per_dim.mean()), float(rmse_p_per_dim.mean()), rmse_q_per_dim, rmse_p_per_dim


def stats_dict(prefix, values):
    if not values:
        return {f"{prefix}_{k}": np.nan for k in ["mean", "std", "p25", "median", "p95", "p99"]}
    arr = np.array(values)
    return {
        f"{prefix}_mean": np.mean(arr),
        f"{prefix}_std": np.std(arr),
        f"{prefix}_p25": np.percentile(arr, 25),
        f"{prefix}_median": np.median(arr),
        f"{prefix}_p95": np.percentile(arr, 95),
        f"{prefix}_p99": np.percentile(arr, 99),
    }


def generate_batch(dpf, num_samples, length, batch_torques, batch_size, guidance_kwargs, initial_noise=None):
    all_states, all_torques = [], []
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        bt = batch_torques[batch_start:batch_end]
        bn = initial_noise[batch_start:batch_end] if initial_noise is not None else None
        state, torque_out = dpf.sample_trajectories(
            num_samples=bt.shape[0],
            trajectory_length=length,
            context_fraction=0.5,
            use_ema=True,
            torque=bt,
            initial_noise=bn,
            **guidance_kwargs,
        )
        all_states.append(state)
        all_torques.append(torque_out)
    return torch.cat(all_states, dim=0), torch.cat(all_torques, dim=0)


def compute_metrics_for_samples(
    states,
    torques_out,
    num_samples,
    mj_model,
    hnn,
    var_dq,
    var_dp,
    qpos_dim,
    desc="",
    hamres_kwargs=None,
):
    if hamres_kwargs is None:
        hamres_kwargs = {}

    nmse_q_list, nmse_p_list, hamres_list = [], [], []
    nmse_q_per_dim_list, nmse_p_per_dim_list = [], []
    for i in tqdm(range(num_samples), desc=desc):
        state = states[i]
        tau = torques_out[i]
        nmse_q, nmse_p, nmse_q_pd, nmse_p_pd = compute_nmse(state, tau, mj_model, qpos_dim)
        qpos = state[:, :qpos_dim]
        mom = state[:, qpos_dim:]
        hr = compute_hamres(qpos, mom, tau, hnn, var_dq, var_dp, **hamres_kwargs)
        nmse_q_list.append(nmse_q)
        nmse_p_list.append(nmse_p)
        nmse_q_per_dim_list.append(nmse_q_pd)
        nmse_p_per_dim_list.append(nmse_p_pd)
        if not np.isnan(hr):
            hamres_list.append(hr)
    return nmse_q_list, nmse_p_list, hamres_list, nmse_q_per_dim_list, nmse_p_per_dim_list


def compute_rmse_for_samples(
    states,
    torques_out,
    num_samples,
    mj_model,
    hnn,
    var_dq,
    var_dp,
    qpos_dim,
    desc="",
    hamres_kwargs=None,
):
    if hamres_kwargs is None:
        hamres_kwargs = {}

    rmse_q_list, rmse_p_list, hamres_list = [], [], []
    rmse_q_per_dim_list, rmse_p_per_dim_list = [], []
    for i in tqdm(range(num_samples), desc=desc):
        state = states[i]
        tau = torques_out[i]
        rmse_q, rmse_p, rmse_q_pd, rmse_p_pd = compute_rmse(state, tau, mj_model, qpos_dim)
        qpos = state[:, :qpos_dim]
        mom = state[:, qpos_dim:]
        hr = compute_hamres(qpos, mom, tau, hnn, var_dq, var_dp, **hamres_kwargs)
        rmse_q_list.append(rmse_q)
        rmse_p_list.append(rmse_p)
        rmse_q_per_dim_list.append(rmse_q_pd)
        rmse_p_per_dim_list.append(rmse_p_pd)
        if not np.isnan(hr):
            hamres_list.append(hr)
    return rmse_q_list, rmse_p_list, hamres_list, rmse_q_per_dim_list, rmse_p_per_dim_list


def resolve_results_output_path(system, policy, sigmas, sweep_mode):
    if sweep_mode:
        out_dir = project_root / "output" / "results" / f"{system}_smoothing_sweep"
        filename = f"sweep_{policy}.csv"
    else:
        out_dir = project_root / "output" / "results" / f"{system}_smoothed"
        filename = f"metrics_{policy}_sigma{sigmas[0]}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename
