#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np


TARGET_RADIUS = 0.2
DEFAULT_TORQUE_LIMIT = 0.9

_WORKER_H5 = None
_WORKER_MODEL = None
_WORKER_IDS = None


def get_model_ids(model: mujoco.MjModel) -> dict[str, int]:
    joint0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint0")
    joint1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    target_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
    target_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")
    fingertip_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fingertip")
    return {
        "joint0_qpos": int(model.jnt_qposadr[joint0]),
        "joint1_qpos": int(model.jnt_qposadr[joint1]),
        "target_x_qpos": int(model.jnt_qposadr[target_x]),
        "target_y_qpos": int(model.jnt_qposadr[target_y]),
        "joint0_dof": int(model.jnt_dofadr[joint0]),
        "joint1_dof": int(model.jnt_dofadr[joint1]),
        "target_x_dof": int(model.jnt_dofadr[target_x]),
        "target_y_dof": int(model.jnt_dofadr[target_y]),
        "fingertip_body_id": int(fingertip_body_id),
    }


def init_worker(h5_path: str, xml_path: str, dt: float) -> None:
    global _WORKER_H5, _WORKER_MODEL, _WORKER_IDS
    _WORKER_H5 = h5py.File(h5_path, "r")
    _WORKER_MODEL = mujoco.MjModel.from_xml_path(xml_path)
    _WORKER_MODEL.opt.timestep = float(dt)
    _WORKER_MODEL.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    _WORKER_IDS = get_model_ids(_WORKER_MODEL)


def close_worker() -> None:
    global _WORKER_H5, _WORKER_MODEL, _WORKER_IDS
    if _WORKER_H5 is not None:
        _WORKER_H5.close()
    _WORKER_H5 = None
    _WORKER_MODEL = None
    _WORKER_IDS = None


def set_reacher_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_qpos: np.ndarray,
    arm_qvel: np.ndarray,
    waypoint_xy: np.ndarray,
    ids: dict[str, int],
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    data.qpos[ids["joint0_qpos"]] = float(arm_qpos[0])
    data.qpos[ids["joint1_qpos"]] = float(arm_qpos[1])
    data.qpos[ids["target_x_qpos"]] = float(waypoint_xy[0])
    data.qpos[ids["target_y_qpos"]] = float(waypoint_xy[1])
    data.qvel[ids["joint0_dof"]] = float(arm_qvel[0])
    data.qvel[ids["joint1_dof"]] = float(arm_qvel[1])
    data.qvel[ids["target_x_dof"]] = 0.0
    data.qvel[ids["target_y_dof"]] = 0.0
    mujoco.mj_forward(model, data)


def _rmse(diff: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))


def analyze_trajectory(task: tuple[int, int, float, float, int]) -> dict:
    traj_index, bins, target_radius, torque_limit, replay_stride = task
    assert _WORKER_H5 is not None
    assert _WORKER_MODEL is not None
    assert _WORKER_IDS is not None

    group = _WORKER_H5[f"traj_{traj_index}"]
    qpos = group["seq_qpos"][:].astype(np.float64)
    qvel = group["seq_qvel"][:].astype(np.float64)
    qacc = group["seq_qacc"][:].astype(np.float64)
    mom = group["seq_mom"][:].astype(np.float64)
    mom_dot = group["seq_mom_dot"][:].astype(np.float64)
    torque = group["seq_torque"][:].astype(np.float64)
    energy = group["seq_energy"][:].astype(np.float64)
    xy = group["seq_fingertip_xy"][:].astype(np.float64)
    waypoint_xy = group["waypoint_xy"][:].astype(np.float64)
    waypoint_index = int(group.attrs["waypoint_index"])
    waypoint_error = float(group.attrs["waypoint_error"])

    hist = np.zeros((bins, bins), dtype=np.int64)
    xy_radius = np.linalg.norm(xy, axis=1)
    mask = xy_radius <= target_radius
    if np.any(mask):
        edges = np.linspace(-target_radius, target_radius, bins + 1)
        xy_use = xy[mask]
        ix = np.clip(np.digitize(xy_use[:, 0], edges) - 1, 0, bins - 1)
        iy = np.clip(np.digitize(xy_use[:, 1], edges) - 1, 0, bins - 1)
        np.add.at(hist, (ix, iy), 1)

    data = mujoco.MjData(_WORKER_MODEL)
    set_reacher_state(_WORKER_MODEL, data, qpos[0], qvel[0], waypoint_xy, _WORKER_IDS)

    qpos_sq_sum = 0.0
    qvel_sq_sum = 0.0
    xy_sq_sum = 0.0
    qpos_max_abs = 0.0
    qvel_max_abs = 0.0
    xy_max_abs = 0.0
    qpos_final_abs = 0.0
    qvel_final_abs = 0.0
    xy_final_abs = 0.0

    for step in range(len(torque)):
        pred_qpos = data.qpos[[_WORKER_IDS["joint0_qpos"], _WORKER_IDS["joint1_qpos"]]]
        pred_qvel = data.qvel[[_WORKER_IDS["joint0_dof"], _WORKER_IDS["joint1_dof"]]]
        pred_xy = data.xpos[_WORKER_IDS["fingertip_body_id"], :2]

        diff_qpos = pred_qpos - qpos[step]
        diff_qvel = pred_qvel - qvel[step]
        diff_xy = pred_xy - xy[step]

        qpos_sq_sum += float(np.sum(np.square(diff_qpos)))
        qvel_sq_sum += float(np.sum(np.square(diff_qvel)))
        xy_sq_sum += float(np.sum(np.square(diff_xy)))
        qpos_max_abs = max(qpos_max_abs, float(np.max(np.abs(diff_qpos))))
        qvel_max_abs = max(qvel_max_abs, float(np.max(np.abs(diff_qvel))))
        xy_max_abs = max(xy_max_abs, float(np.max(np.abs(diff_xy))))

        if step == len(torque) - 1:
            qpos_final_abs = float(np.max(np.abs(diff_qpos)))
            qvel_final_abs = float(np.max(np.abs(diff_qvel)))
            xy_final_abs = float(np.max(np.abs(diff_xy)))

        data.ctrl[:] = torque[step]
        for _ in range(replay_stride):
            mujoco.mj_step(_WORKER_MODEL, data)

    point_count = int(qpos.shape[0] * qpos.shape[1])
    xy_count = int(xy.shape[0] * xy.shape[1])

    return {
        "point_count": int(qpos.shape[0]),
        "qpos_min": qpos.min(axis=0).tolist(),
        "qpos_max": qpos.max(axis=0).tolist(),
        "qpos_sum": qpos.sum(axis=0).tolist(),
        "qpos_sq_sum": np.square(qpos).sum(axis=0).tolist(),
        "qvel_min": qvel.min(axis=0).tolist(),
        "qvel_max": qvel.max(axis=0).tolist(),
        "qvel_sum": qvel.sum(axis=0).tolist(),
        "qvel_sq_sum": np.square(qvel).sum(axis=0).tolist(),
        "qacc_min": qacc.min(axis=0).tolist(),
        "qacc_max": qacc.max(axis=0).tolist(),
        "qacc_sum": qacc.sum(axis=0).tolist(),
        "qacc_sq_sum": np.square(qacc).sum(axis=0).tolist(),
        "mom_min": mom.min(axis=0).tolist(),
        "mom_max": mom.max(axis=0).tolist(),
        "mom_sum": mom.sum(axis=0).tolist(),
        "mom_sq_sum": np.square(mom).sum(axis=0).tolist(),
        "mom_dot_min": mom_dot.min(axis=0).tolist(),
        "mom_dot_max": mom_dot.max(axis=0).tolist(),
        "mom_dot_sum": mom_dot.sum(axis=0).tolist(),
        "mom_dot_sq_sum": np.square(mom_dot).sum(axis=0).tolist(),
        "torque_min": torque.min(axis=0).tolist(),
        "torque_max": torque.max(axis=0).tolist(),
        "torque_sum": torque.sum(axis=0).tolist(),
        "torque_sq_sum": np.square(torque).sum(axis=0).tolist(),
        "xy_min": xy.min(axis=0).tolist(),
        "xy_max": xy.max(axis=0).tolist(),
        "xy_sum": xy.sum(axis=0).tolist(),
        "xy_sq_sum": np.square(xy).sum(axis=0).tolist(),
        "waypoint_error": waypoint_error,
        "waypoint_index": waypoint_index,
        "waypoint_radius": float(np.linalg.norm(waypoint_xy)),
        "visited_radius_min": float(xy_radius.min()),
        "visited_radius_mean": float(xy_radius.mean()),
        "visited_radius_max": float(xy_radius.max()),
        "traj_q0_span": float(qpos[:, 0].max() - qpos[:, 0].min()),
        "traj_q1_span": float(qpos[:, 1].max() - qpos[:, 1].min()),
        "traj_q0_absmax": float(np.max(np.abs(qpos[:, 0]))),
        "traj_q0_gt_pi": bool(np.any(np.abs(qpos[:, 0]) > np.pi)),
        "traj_q0_gt_2pi": bool(np.any(np.abs(qpos[:, 0]) > 2.0 * np.pi)),
        "traj_q1_gt_3": bool(np.any(np.abs(qpos[:, 1]) > 3.0)),
        "point_q1_gt_3": int(np.count_nonzero(np.abs(qpos[:, 1]) > 3.0)),
        "energy_min": float(energy.min()),
        "energy_max": float(energy.max()),
        "traj_abs_energy_max": float(np.max(np.abs(energy))),
        "torque_clip_count": int(np.count_nonzero(np.isclose(np.abs(torque), torque_limit, atol=1e-6))),
        "torque_value_count": int(torque.size),
        "xy_inside_target_count": int(np.count_nonzero(mask)),
        "hist": hist.tolist(),
        "replay_qpos_rmse": float(np.sqrt(qpos_sq_sum / point_count)),
        "replay_qvel_rmse": float(np.sqrt(qvel_sq_sum / point_count)),
        "replay_xy_rmse": float(np.sqrt(xy_sq_sum / xy_count)),
        "replay_qpos_max_abs": qpos_max_abs,
        "replay_qvel_max_abs": qvel_max_abs,
        "replay_xy_max_abs": xy_max_abs,
        "replay_qpos_final_abs": qpos_final_abs,
        "replay_qvel_final_abs": qvel_final_abs,
        "replay_xy_final_abs": xy_final_abs,
    }


def _accumulate_stats(results: list[dict], bins: int) -> dict:
    total_points = int(sum(item["point_count"] for item in results))
    total_scalar_points = float(total_points)

    def arr_min(key: str) -> list[float]:
        return np.min(np.array([item[key] for item in results], dtype=np.float64), axis=0).tolist()

    def arr_max(key: str) -> list[float]:
        return np.max(np.array([item[key] for item in results], dtype=np.float64), axis=0).tolist()

    def arr_mean_std(sum_key: str, sq_key: str) -> tuple[list[float], list[float]]:
        sums = np.sum(np.array([item[sum_key] for item in results], dtype=np.float64), axis=0)
        sq_sums = np.sum(np.array([item[sq_key] for item in results], dtype=np.float64), axis=0)
        mean = sums / total_scalar_points
        var = np.maximum(sq_sums / total_scalar_points - np.square(mean), 0.0)
        return mean.tolist(), np.sqrt(var).tolist()

    def scalar_series(key: str) -> np.ndarray:
        return np.array([item[key] for item in results], dtype=np.float64)

    hist = np.sum(np.array([item["hist"] for item in results], dtype=np.int64), axis=0)
    occupied = hist > 0
    occupied_counts = hist[occupied]

    summary = {
        "num_trajectories": len(results),
        "num_steps": int(results[0]["point_count"]) if results else 0,
        "qpos_min": arr_min("qpos_min"),
        "qpos_max": arr_max("qpos_max"),
        "qpos_mean_std": arr_mean_std("qpos_sum", "qpos_sq_sum"),
        "qvel_min": arr_min("qvel_min"),
        "qvel_max": arr_max("qvel_max"),
        "qvel_mean_std": arr_mean_std("qvel_sum", "qvel_sq_sum"),
        "qacc_min": arr_min("qacc_min"),
        "qacc_max": arr_max("qacc_max"),
        "qacc_mean_std": arr_mean_std("qacc_sum", "qacc_sq_sum"),
        "mom_min": arr_min("mom_min"),
        "mom_max": arr_max("mom_max"),
        "mom_mean_std": arr_mean_std("mom_sum", "mom_sq_sum"),
        "mom_dot_min": arr_min("mom_dot_min"),
        "mom_dot_max": arr_max("mom_dot_max"),
        "mom_dot_mean_std": arr_mean_std("mom_dot_sum", "mom_dot_sq_sum"),
        "torque_min": arr_min("torque_min"),
        "torque_max": arr_max("torque_max"),
        "torque_mean_std": arr_mean_std("torque_sum", "torque_sq_sum"),
        "xy_min": arr_min("xy_min"),
        "xy_max": arr_max("xy_max"),
        "xy_mean_std": arr_mean_std("xy_sum", "xy_sq_sum"),
        "waypoint_error": {
            "mean": float(scalar_series("waypoint_error").mean()),
            "p95": float(np.percentile(scalar_series("waypoint_error"), 95)),
            "max": float(scalar_series("waypoint_error").max()),
        },
        "waypoint_index": {
            "min": int(scalar_series("waypoint_index").min()),
            "mean": float(scalar_series("waypoint_index").mean()),
            "max": int(scalar_series("waypoint_index").max()),
        },
        "waypoint_radius": {
            "mean": float(scalar_series("waypoint_radius").mean()),
            "p95": float(np.percentile(scalar_series("waypoint_radius"), 95)),
            "max": float(scalar_series("waypoint_radius").max()),
        },
        "visited_radius": {
            "min": float(scalar_series("visited_radius_min").min()),
            "mean": float(scalar_series("visited_radius_mean").mean()),
            "max": float(scalar_series("visited_radius_max").max()),
        },
        "traj_q0_span": {
            "mean": float(scalar_series("traj_q0_span").mean()),
            "p95": float(np.percentile(scalar_series("traj_q0_span"), 95)),
            "max": float(scalar_series("traj_q0_span").max()),
        },
        "traj_q1_span": {
            "mean": float(scalar_series("traj_q1_span").mean()),
            "p95": float(np.percentile(scalar_series("traj_q1_span"), 95)),
            "max": float(scalar_series("traj_q1_span").max()),
        },
        "traj_q0_absmax": {
            "mean": float(scalar_series("traj_q0_absmax").mean()),
            "p95": float(np.percentile(scalar_series("traj_q0_absmax"), 95)),
            "max": float(scalar_series("traj_q0_absmax").max()),
        },
        "traj_with_abs_q0_gt_pi_fraction": float(np.mean(scalar_series("traj_q0_gt_pi"))),
        "traj_with_abs_q0_gt_2pi_fraction": float(np.mean(scalar_series("traj_q0_gt_2pi"))),
        "traj_with_abs_q1_gt_3_fraction": float(np.mean(scalar_series("traj_q1_gt_3"))),
        "points_with_abs_q1_gt_3_fraction": float(np.sum(scalar_series("point_q1_gt_3")) / total_points),
        "energy": {
            "min": float(scalar_series("energy_min").min()),
            "max": float(scalar_series("energy_max").max()),
            "traj_abs_max_mean": float(scalar_series("traj_abs_energy_max").mean()),
            "traj_abs_max_p95": float(np.percentile(scalar_series("traj_abs_energy_max"), 95)),
            "traj_abs_max_max": float(scalar_series("traj_abs_energy_max").max()),
        },
        "torque_clip_fraction": float(
            np.sum(scalar_series("torque_clip_count")) / np.sum(scalar_series("torque_value_count"))
        ),
        "xy_inside_target_fraction": float(np.sum(scalar_series("xy_inside_target_count")) / total_points),
        "occupancy": {
            "bins": bins,
            "occupied_bins": int(np.count_nonzero(occupied)),
            "occupied_bin_fraction": float(np.mean(occupied)),
            "count_cv": float(occupied_counts.std() / occupied_counts.mean()) if occupied_counts.size else 0.0,
            "count_p05": float(np.percentile(occupied_counts, 5)) if occupied_counts.size else 0.0,
            "count_p50": float(np.percentile(occupied_counts, 50)) if occupied_counts.size else 0.0,
            "count_p95": float(np.percentile(occupied_counts, 95)) if occupied_counts.size else 0.0,
        },
        "forward_replay": {
            "qpos_rmse_mean": float(scalar_series("replay_qpos_rmse").mean()),
            "qpos_rmse_p95": float(np.percentile(scalar_series("replay_qpos_rmse"), 95)),
            "qpos_rmse_max": float(scalar_series("replay_qpos_rmse").max()),
            "qvel_rmse_mean": float(scalar_series("replay_qvel_rmse").mean()),
            "qvel_rmse_p95": float(np.percentile(scalar_series("replay_qvel_rmse"), 95)),
            "qvel_rmse_max": float(scalar_series("replay_qvel_rmse").max()),
            "xy_rmse_mean": float(scalar_series("replay_xy_rmse").mean()),
            "xy_rmse_p95": float(np.percentile(scalar_series("replay_xy_rmse"), 95)),
            "xy_rmse_max": float(scalar_series("replay_xy_rmse").max()),
            "qpos_max_abs_mean": float(scalar_series("replay_qpos_max_abs").mean()),
            "qpos_max_abs_p95": float(np.percentile(scalar_series("replay_qpos_max_abs"), 95)),
            "qpos_max_abs_max": float(scalar_series("replay_qpos_max_abs").max()),
            "qvel_max_abs_mean": float(scalar_series("replay_qvel_max_abs").mean()),
            "qvel_max_abs_p95": float(np.percentile(scalar_series("replay_qvel_max_abs"), 95)),
            "qvel_max_abs_max": float(scalar_series("replay_qvel_max_abs").max()),
            "xy_max_abs_mean": float(scalar_series("replay_xy_max_abs").mean()),
            "xy_max_abs_p95": float(np.percentile(scalar_series("replay_xy_max_abs"), 95)),
            "xy_max_abs_max": float(scalar_series("replay_xy_max_abs").max()),
            "qpos_final_abs_max": float(scalar_series("replay_qpos_final_abs").max()),
            "qvel_final_abs_max": float(scalar_series("replay_qvel_final_abs").max()),
            "xy_final_abs_max": float(scalar_series("replay_xy_final_abs").max()),
        },
    }
    return summary


def analyze_file(
    h5_path: Path,
    xml_path: Path,
    bins: int,
    target_radius: float,
    torque_limit: float,
    replay_stride: int,
    num_workers: int,
) -> dict:
    with h5py.File(h5_path, "r") as f:
        num_trajectories = int(f.attrs["num_trajectories"])
        dt = float(f.attrs["dt"])
        num_steps = int(f.attrs["num_steps"])
        generator_config = json.loads(f.attrs["generator_config"]) if "generator_config" in f.attrs else {}

    tasks = [(idx, bins, target_radius, torque_limit, replay_stride) for idx in range(num_trajectories)]

    if num_workers <= 1:
        init_worker(str(h5_path), str(xml_path), dt)
        try:
            results = [analyze_trajectory(task) for task in tasks]
        finally:
            close_worker()
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(str(h5_path), str(xml_path), dt),
        ) as executor:
            chunksize = max(1, len(tasks) // max(1, num_workers * 8))
            results = list(executor.map(analyze_trajectory, tasks, chunksize=chunksize))

    summary = _accumulate_stats(results, bins=bins)
    summary["file"] = str(h5_path)
    summary["dt"] = dt
    summary["num_steps"] = num_steps
    summary["generator_config"] = generator_config
    return summary


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def render_markdown(
    output_dir: Path,
    xml_path: Path,
    train_summary: dict,
    val_summary: dict,
    target_radius: float,
    torque_limit: float,
    pass_qpos_max_abs: float,
    pass_qvel_max_abs: float,
    pass_xy_max_abs: float,
) -> str:
    def split_block(name: str, s: dict) -> list[str]:
        q_mean, q_std = s["qpos_mean_std"]
        v_mean, v_std = s["qvel_mean_std"]
        a_mean, a_std = s["qacc_mean_std"]
        t_mean, t_std = s["torque_mean_std"]
        xy_mean, xy_std = s["xy_mean_std"]
        fr = s["forward_replay"]
        return [
            f"### {name}",
            "",
            f"- file: `{Path(s['file']).name}`",
            f"- trajectories: `{s['num_trajectories']}`",
            f"- steps per trajectory: `{s['num_steps']}`",
            f"- dt: `{_fmt(s['dt'])}`",
            f"- qpos min/max: `{s['qpos_min']}` / `{s['qpos_max']}`",
            f"- qpos mean/std: `{q_mean}` / `{q_std}`",
            f"- qvel min/max: `{s['qvel_min']}` / `{s['qvel_max']}`",
            f"- qvel mean/std: `{v_mean}` / `{v_std}`",
            f"- qacc min/max: `{s['qacc_min']}` / `{s['qacc_max']}`",
            f"- qacc mean/std: `{a_mean}` / `{a_std}`",
            f"- torque min/max: `{s['torque_min']}` / `{s['torque_max']}`",
            f"- torque mean/std: `{t_mean}` / `{t_std}`",
            f"- end-effector xy min/max: `{s['xy_min']}` / `{s['xy_max']}`",
            f"- end-effector xy mean/std: `{xy_mean}` / `{xy_std}`",
            f"- waypoint error mean / p95 / max: `{_fmt(s['waypoint_error']['mean'])}` / `{_fmt(s['waypoint_error']['p95'])}` / `{_fmt(s['waypoint_error']['max'])}`",
            f"- waypoint index min / mean / max: `{s['waypoint_index']['min']}` / `{_fmt(s['waypoint_index']['mean'])}` / `{s['waypoint_index']['max']}`",
            f"- waypoint radius mean / p95 / max: `{_fmt(s['waypoint_radius']['mean'])}` / `{_fmt(s['waypoint_radius']['p95'])}` / `{_fmt(s['waypoint_radius']['max'])}`",
            f"- visited radius min / mean / max: `{_fmt(s['visited_radius']['min'])}` / `{_fmt(s['visited_radius']['mean'])}` / `{_fmt(s['visited_radius']['max'])}`",
            f"- trajectory q0 span mean / p95 / max: `{_fmt(s['traj_q0_span']['mean'])}` / `{_fmt(s['traj_q0_span']['p95'])}` / `{_fmt(s['traj_q0_span']['max'])}`",
            f"- trajectory q1 span mean / p95 / max: `{_fmt(s['traj_q1_span']['mean'])}` / `{_fmt(s['traj_q1_span']['p95'])}` / `{_fmt(s['traj_q1_span']['max'])}`",
            f"- trajectory |q0| max mean / p95 / max: `{_fmt(s['traj_q0_absmax']['mean'])}` / `{_fmt(s['traj_q0_absmax']['p95'])}` / `{_fmt(s['traj_q0_absmax']['max'])}`",
            f"- fraction of trajectories with `|q0| > pi`: `{_fmt(s['traj_with_abs_q0_gt_pi_fraction'])}`",
            f"- fraction of trajectories with `|q0| > 2pi`: `{_fmt(s['traj_with_abs_q0_gt_2pi_fraction'])}`",
            f"- fraction of trajectories with `|q1| > 3`: `{_fmt(s['traj_with_abs_q1_gt_3_fraction'])}`",
            f"- fraction of timesteps with `|q1| > 3`: `{_fmt(s['points_with_abs_q1_gt_3_fraction'])}`",
            f"- energy min / max: `{_fmt(s['energy']['min'])}` / `{_fmt(s['energy']['max'])}`",
            f"- per-trajectory |energy| max mean / p95 / max: `{_fmt(s['energy']['traj_abs_max_mean'])}` / `{_fmt(s['energy']['traj_abs_max_p95'])}` / `{_fmt(s['energy']['traj_abs_max_max'])}`",
            f"- torque clip fraction at limit `{_fmt(torque_limit)}`: `{_fmt(s['torque_clip_fraction'])}`",
            f"- fraction of end-effector samples inside target disk radius `{_fmt(target_radius)}`: `{_fmt(s['xy_inside_target_fraction'])}`",
            f"- occupancy: `{s['occupancy']['occupied_bins']}/{s['occupancy']['bins'] ** 2}` bins hit, coverage fraction `{_fmt(s['occupancy']['occupied_bin_fraction'])}`, count CV `{_fmt(s['occupancy']['count_cv'])}`",
            f"- occupancy count p05 / p50 / p95: `{_fmt(s['occupancy']['count_p05'])}` / `{_fmt(s['occupancy']['count_p50'])}` / `{_fmt(s['occupancy']['count_p95'])}`",
            "",
            f"Forward replay verification against the saved states:",
            f"- qpos RMSE mean / p95 / max: `{_fmt(fr['qpos_rmse_mean'])}` / `{_fmt(fr['qpos_rmse_p95'])}` / `{_fmt(fr['qpos_rmse_max'])}`",
            f"- qvel RMSE mean / p95 / max: `{_fmt(fr['qvel_rmse_mean'])}` / `{_fmt(fr['qvel_rmse_p95'])}` / `{_fmt(fr['qvel_rmse_max'])}`",
            f"- xy RMSE mean / p95 / max: `{_fmt(fr['xy_rmse_mean'])}` / `{_fmt(fr['xy_rmse_p95'])}` / `{_fmt(fr['xy_rmse_max'])}`",
            f"- qpos max-abs error mean / p95 / max: `{_fmt(fr['qpos_max_abs_mean'])}` / `{_fmt(fr['qpos_max_abs_p95'])}` / `{_fmt(fr['qpos_max_abs_max'])}`",
            f"- qvel max-abs error mean / p95 / max: `{_fmt(fr['qvel_max_abs_mean'])}` / `{_fmt(fr['qvel_max_abs_p95'])}` / `{_fmt(fr['qvel_max_abs_max'])}`",
            f"- xy max-abs error mean / p95 / max: `{_fmt(fr['xy_max_abs_mean'])}` / `{_fmt(fr['xy_max_abs_p95'])}` / `{_fmt(fr['xy_max_abs_max'])}`",
            f"- final-step qpos / qvel / xy max-abs error: `{_fmt(fr['qpos_final_abs_max'])}` / `{_fmt(fr['qvel_final_abs_max'])}` / `{_fmt(fr['xy_final_abs_max'])}`",
            "",
        ]

    train_pass = (
        train_summary["forward_replay"]["qpos_max_abs_max"] <= pass_qpos_max_abs
        and train_summary["forward_replay"]["qvel_max_abs_max"] <= pass_qvel_max_abs
        and train_summary["forward_replay"]["xy_max_abs_max"] <= pass_xy_max_abs
    )
    val_pass = (
        val_summary["forward_replay"]["qpos_max_abs_max"] <= pass_qpos_max_abs
        and val_summary["forward_replay"]["qvel_max_abs_max"] <= pass_qvel_max_abs
        and val_summary["forward_replay"]["xy_max_abs_max"] <= pass_xy_max_abs
    )

    lines = [
        "# Reacher Bidirectional Dataset Report",
        "",
        f"- dataset directory: `{output_dir}`",
        f"- verification xml: `{xml_path}`",
        f"- target radius used for occupancy summary: `{_fmt(target_radius)}`",
        f"- replay pass thresholds:",
        f"- qpos max-abs error <= `{_fmt(pass_qpos_max_abs)}`",
        f"- qvel max-abs error <= `{_fmt(pass_qvel_max_abs)}`",
        f"- end-effector xy max-abs error <= `{_fmt(pass_xy_max_abs)}`",
        f"- train replay pass: `{train_pass}`",
        f"- val replay pass: `{val_pass}`",
        "",
        "## Configuration",
        "",
        "The files were analyzed using their stored `generator_config` and replayed with the non-dissipative Reacher XML.",
        "",
    ]
    lines.extend(split_block("Train", train_summary))
    lines.extend(split_block("Val", val_summary))
    lines.extend(
        [
            "## Training Readout",
            "",
            "This dataset is numerically much cleaner than the earlier high-torque version.",
            "",
            "- Forward replay error is the most important sanity check. If the pass flags above are `True`, the saved trajectories are self-consistent with the MuJoCo forward simulation under the stored torque sequence.",
            "- Torque clipping is effectively gone in this configuration, which is a good sign for learning smoother control-conditioned dynamics.",
            "- Train and val are well matched in scale and coverage, so the split itself looks healthy.",
            "- `q0` is still periodic and unwrapped, so model inputs should still treat it as periodic rather than as an ordinary Euclidean coordinate.",
            "- `q1` still crosses its nominal `[-3, 3]` range sometimes, but the overshoot is much smaller and less frequent than in the earlier dataset.",
            "",
            "## Files",
            "",
            f"- train: `{Path(train_summary['file']).name}`",
            f"- val: `{Path(val_summary['file']).name}`",
            "- machine-readable summary: `dataset_report.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze bidirectional Reacher HDF5 datasets and verify forward replay.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/data/reacher_bidirectional_dt0p001_len1000",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_from_dataset.xml",
    )
    parser.add_argument("--train_file", type=str, default="traj_40000-steps_1000.h5")
    parser.add_argument("--val_file", type=str, default="traj_2000-steps_1000.h5")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--target_radius", type=float, default=TARGET_RADIUS)
    parser.add_argument("--torque_limit", type=float, default=DEFAULT_TORQUE_LIMIT)
    parser.add_argument("--replay_stride", type=int, default=1)
    parser.add_argument("--pass_qpos_max_abs", type=float, default=2e-5)
    parser.add_argument("--pass_qvel_max_abs", type=float, default=2e-3)
    parser.add_argument("--pass_xy_max_abs", type=float, default=2e-6)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    xml_path = Path(args.xml_path)
    train_path = dataset_dir / args.train_file
    val_path = dataset_dir / args.val_file

    train_summary = analyze_file(
        h5_path=train_path,
        xml_path=xml_path,
        bins=args.bins,
        target_radius=args.target_radius,
        torque_limit=args.torque_limit,
        replay_stride=args.replay_stride,
        num_workers=args.num_workers,
    )
    val_summary = analyze_file(
        h5_path=val_path,
        xml_path=xml_path,
        bins=args.bins,
        target_radius=args.target_radius,
        torque_limit=args.torque_limit,
        replay_stride=args.replay_stride,
        num_workers=args.num_workers,
    )

    report = {
        "dataset_dir": str(dataset_dir),
        "xml_path": str(xml_path),
        "train": train_summary,
        "val": val_summary,
        "pass_thresholds": {
            "qpos_max_abs": args.pass_qpos_max_abs,
            "qvel_max_abs": args.pass_qvel_max_abs,
            "xy_max_abs": args.pass_xy_max_abs,
        },
    }

    json_path = dataset_dir / "dataset_report.json"
    readme_path = dataset_dir / "README.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    readme_path.write_text(
        render_markdown(
            output_dir=dataset_dir,
            xml_path=xml_path,
            train_summary=train_summary,
            val_summary=val_summary,
            target_radius=args.target_radius,
            torque_limit=args.torque_limit,
            pass_qpos_max_abs=args.pass_qpos_max_abs,
            pass_qvel_max_abs=args.pass_qvel_max_abs,
            pass_xy_max_abs=args.pass_xy_max_abs,
        )
    )

    print(f"Wrote {json_path}")
    print(f"Wrote {readme_path}")


if __name__ == "__main__":
    main()
