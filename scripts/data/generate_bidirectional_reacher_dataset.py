#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path

import h5py
import mujoco
import numpy as np
from tqdm import tqdm


L1 = 0.10
L2 = 0.11
TRAJ_KEYS = ("seq_qpos", "seq_qvel", "seq_qacc", "seq_mom", "seq_mom_dot", "seq_torque", "seq_energy")

_WORKER_MODEL = None
_WORKER_XML_PATH = None
_WORKER_DT = None
_WORKER_IDS = None


def sample_waypoint_in_disk(rng: np.random.Generator, max_radius: float) -> np.ndarray:
    while True:
        xy = rng.uniform(-max_radius, max_radius, size=2)
        if np.linalg.norm(xy) <= max_radius:
            return xy


def ik_2link(xy: np.ndarray, elbow_branch: int) -> np.ndarray:
    x, y = float(xy[0]), float(xy[1])
    r2 = x * x + y * y
    cos_q2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
    base = math.acos(cos_q2)
    q2 = base if elbow_branch >= 0 else -base
    q1 = math.atan2(y, x) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return np.array([q1, q2], dtype=np.float64)


def generate_smooth_torque(
    rng: np.random.Generator,
    num_steps: int,
    dt: float,
    nu: int,
    scale: float,
    num_sin: int = 4,
) -> np.ndarray:
    t = np.arange(num_steps, dtype=np.float64) * dt
    torque = np.zeros((num_steps, nu), dtype=np.float64)
    for dim in range(nu):
        for _ in range(num_sin):
            amp = rng.uniform(0.15 * scale, scale)
            freq = rng.uniform(0.4, 2.6)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            torque[:, dim] += amp * np.sin(2.0 * math.pi * freq * t + phase)
    return np.clip(torque, -0.9, 0.9)


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


def init_worker_model(xml_path: str, dt: float) -> None:
    global _WORKER_MODEL, _WORKER_XML_PATH, _WORKER_DT, _WORKER_IDS
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = float(dt)
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    _WORKER_MODEL = model
    _WORKER_XML_PATH = xml_path
    _WORKER_DT = float(dt)
    _WORKER_IDS = get_model_ids(model)


def get_or_create_worker_model(xml_path: str, dt: float) -> tuple[mujoco.MjModel, dict[str, int]]:
    global _WORKER_MODEL, _WORKER_XML_PATH, _WORKER_DT, _WORKER_IDS
    if _WORKER_MODEL is None or _WORKER_XML_PATH != xml_path or _WORKER_DT != float(dt):
        init_worker_model(xml_path, dt)
    return _WORKER_MODEL, _WORKER_IDS


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

    data.qpos[ids["joint0_qpos"]] = arm_qpos[0]
    data.qpos[ids["joint1_qpos"]] = arm_qpos[1]
    data.qpos[ids["target_x_qpos"]] = waypoint_xy[0]
    data.qpos[ids["target_y_qpos"]] = waypoint_xy[1]

    data.qvel[ids["joint0_dof"]] = arm_qvel[0]
    data.qvel[ids["joint1_dof"]] = arm_qvel[1]
    data.qvel[ids["target_x_dof"]] = 0.0
    data.qvel[ids["target_y_dof"]] = 0.0
    mujoco.mj_forward(model, data)


def simulate_helper_segment(
    model: mujoco.MjModel,
    arm_qpos: np.ndarray,
    arm_qvel: np.ndarray,
    waypoint_xy: np.ndarray,
    torque_seq: np.ndarray,
    ids: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    set_reacher_state(model, data, arm_qpos, arm_qvel, waypoint_xy, ids)

    qpos_hist = np.empty((torque_seq.shape[0] + 1, model.nq), dtype=np.float64)
    qvel_hist = np.empty((torque_seq.shape[0] + 1, model.nv), dtype=np.float64)
    qpos_hist[0] = data.qpos.copy()
    qvel_hist[0] = data.qvel.copy()

    for step, tau in enumerate(torque_seq, start=1):
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        qpos_hist[step] = data.qpos.copy()
        qvel_hist[step] = data.qvel.copy()

    return qpos_hist, qvel_hist


def simulate_dataset_rollout(
    model: mujoco.MjModel,
    start_arm_qpos: np.ndarray,
    start_arm_qvel: np.ndarray,
    waypoint_xy: np.ndarray,
    torque_seq: np.ndarray,
    ids: dict[str, int],
) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    set_reacher_state(model, data, start_arm_qpos, start_arm_qvel, waypoint_xy, ids)

    num_steps = int(torque_seq.shape[0])
    nq = model.nq
    nv = model.nv
    M = np.zeros((nv, nv), dtype=np.float64)

    seq_qpos = np.empty((num_steps, 2), dtype=np.float32)
    seq_qvel = np.empty((num_steps, 2), dtype=np.float32)
    seq_qacc = np.empty((num_steps, 2), dtype=np.float32)
    seq_mom = np.empty((num_steps, 2), dtype=np.float32)
    seq_mom_dot = np.empty((num_steps, 2), dtype=np.float32)
    seq_torque = torque_seq.astype(np.float32, copy=False)
    seq_energy = np.empty((num_steps,), dtype=np.float32)
    seq_fingertip_xy = np.empty((num_steps, 2), dtype=np.float32)

    for step in range(num_steps):
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise FloatingPointError("non-finite state encountered before stepping")

        qvel_curr = data.qvel.copy()
        mujoco.mj_fullM(model, M, data.qM)
        mom_curr = M @ qvel_curr

        seq_qpos[step] = data.qpos[[ids["joint0_qpos"], ids["joint1_qpos"]]]
        seq_qvel[step] = qvel_curr[[ids["joint0_dof"], ids["joint1_dof"]]]
        seq_mom[step] = mom_curr[[ids["joint0_dof"], ids["joint1_dof"]]]
        seq_energy[step] = np.float32(data.energy[0] + data.energy[1])
        seq_fingertip_xy[step] = data.xpos[ids["fingertip_body_id"], :2]

        data.ctrl[:] = torque_seq[step]
        mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise FloatingPointError("non-finite state encountered after stepping")

        mujoco.mj_fullM(model, M, data.qM)
        mom_next = M @ data.qvel
        qacc = (data.qvel - qvel_curr) / model.opt.timestep
        mom_dot = (mom_next - mom_curr) / model.opt.timestep
        seq_qacc[step] = qacc[[ids["joint0_dof"], ids["joint1_dof"]]]
        seq_mom_dot[step] = mom_dot[[ids["joint0_dof"], ids["joint1_dof"]]]

    return {
        "seq_qpos": seq_qpos,
        "seq_qvel": seq_qvel,
        "seq_qacc": seq_qacc,
        "seq_mom": seq_mom,
        "seq_mom_dot": seq_mom_dot,
        "seq_torque": seq_torque,
        "seq_energy": seq_energy,
        "seq_fingertip_xy": seq_fingertip_xy,
    }


def build_bidirectional_reacher_trajectory(task: tuple) -> tuple[bool, str, dict | None]:
    (
        xml_path,
        dt,
        trajectory_length,
        seed,
        waypoint_radius,
        waypoint_qvel_scale,
        torque_scale,
        waypoint_tolerance,
        max_abs_qvel,
        max_abs_qacc,
    ) = task

    rng = np.random.default_rng(seed)
    model, ids = get_or_create_worker_model(xml_path, dt)

    waypoint_xy = sample_waypoint_in_disk(rng, max_radius=waypoint_radius)
    elbow_branch = 1 if rng.random() < 0.5 else -1
    arm_qpos = ik_2link(waypoint_xy, elbow_branch=elbow_branch)
    arm_qvel = rng.uniform(-waypoint_qvel_scale, waypoint_qvel_scale, size=2)

    prefix_steps = int(rng.integers(1, trajectory_length - 1))
    suffix_steps = int(trajectory_length - 1 - prefix_steps)

    full_tau = generate_smooth_torque(rng, trajectory_length, dt, model.nu, torque_scale)
    prefix_rollout_tau = full_tau[:prefix_steps]
    prefix_helper_tau = prefix_rollout_tau[::-1].copy()
    suffix_tau_full = full_tau[prefix_steps:]

    prefix_qpos_helper, prefix_qvel_helper = simulate_helper_segment(
        model=model,
        arm_qpos=arm_qpos,
        arm_qvel=-arm_qvel,
        waypoint_xy=waypoint_xy,
        torque_seq=prefix_helper_tau,
        ids=ids,
    )

    start_arm_qpos = prefix_qpos_helper[-1, [ids["joint0_qpos"], ids["joint1_qpos"]]]
    start_arm_qvel = -prefix_qvel_helper[-1, [ids["joint0_dof"], ids["joint1_dof"]]]
    replay_tau = full_tau

    try:
        result = simulate_dataset_rollout(
            model=model,
            start_arm_qpos=start_arm_qpos,
            start_arm_qvel=start_arm_qvel,
            waypoint_xy=waypoint_xy,
            torque_seq=replay_tau,
            ids=ids,
        )
    except FloatingPointError:
        return False, "non_finite", None

    if np.any(np.abs(result["seq_qvel"]) > max_abs_qvel):
        return False, "qvel_limit", None
    if np.any(np.abs(result["seq_qacc"]) > max_abs_qacc):
        return False, "qacc_limit", None

    waypoint_error = float(np.linalg.norm(result["seq_fingertip_xy"][prefix_steps] - waypoint_xy))
    if waypoint_error > waypoint_tolerance:
        return False, "waypoint_miss", None

    if not all(np.all(np.isfinite(result[key])) for key in TRAJ_KEYS):
        return False, "non_finite", None

    result["waypoint_xy"] = waypoint_xy.astype(np.float32)
    result["waypoint_index"] = np.int32(prefix_steps)
    result["waypoint_error"] = np.float32(waypoint_error)
    return True, "accepted", result


def write_h5_header(
    file: h5py.File,
    xml_path: str,
    num_steps: int,
    num_trajectories: int,
    dt: float,
    config: dict,
) -> None:
    file.attrs["xml"] = Path(xml_path).read_text()
    file.attrs["num_steps"] = int(num_steps)
    file.attrs["num_trajectories"] = int(num_trajectories)
    file.attrs["dt"] = float(dt)
    file.attrs["data_dt"] = float(dt)
    file.attrs["skip_steps"] = 1
    file.attrs["state_alignment"] = "pre_step"
    file.attrs["torque_alignment"] = "interval_mean"
    file.attrs["derivative_alignment"] = "forward_difference"
    file.attrs["generator"] = "bidirectional_reacher"
    file.attrs["generator_config"] = json.dumps(config, sort_keys=True)


def write_trajectory_group(file: h5py.File, traj_index: int, result: dict) -> None:
    group = file.create_group(f"traj_{traj_index}")
    for key in TRAJ_KEYS:
        group.create_dataset(key, data=result[key], dtype="f4")
    group.create_dataset("waypoint_xy", data=result["waypoint_xy"], dtype="f4")
    group.create_dataset("seq_fingertip_xy", data=result["seq_fingertip_xy"], dtype="f4")
    group.attrs["waypoint_index"] = int(result["waypoint_index"])
    group.attrs["waypoint_error"] = float(result["waypoint_error"])


def generate_split(
    output_path: Path,
    xml_path: str,
    num_trajectories: int,
    trajectory_length: int,
    dt: float,
    waypoint_radius: float,
    waypoint_qvel_scale: float,
    torque_scale: float,
    waypoint_tolerance: float,
    max_abs_qvel: float,
    max_abs_qacc: float,
    num_workers: int,
    batch_size: int,
    seed_offset: int,
    split_name: str,
) -> None:
    config = {
        "trajectory_length": int(trajectory_length),
        "dt": float(dt),
        "waypoint_radius": float(waypoint_radius),
        "waypoint_qvel_scale": float(waypoint_qvel_scale),
        "torque_scale": float(torque_scale),
        "waypoint_tolerance": float(waypoint_tolerance),
        "max_abs_qvel": float(max_abs_qvel),
        "max_abs_qacc": float(max_abs_qacc),
        "split_name": split_name,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    total_generated = 0
    seed_counter = int(seed_offset)
    reject_counts = {"waypoint_miss": 0, "qvel_limit": 0, "qacc_limit": 0, "non_finite": 0}
    pbar = tqdm(total=num_trajectories, desc=f"Generating {split_name}")

    with h5py.File(output_path, "w") as h5_file:
        write_h5_header(
            file=h5_file,
            xml_path=xml_path,
            num_steps=trajectory_length,
            num_trajectories=num_trajectories,
            dt=dt,
            config=config,
        )

        use_parallel = num_workers is not None and num_workers > 1
        executor = None
        if use_parallel:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=init_worker_model,
                initargs=(xml_path, dt),
            )
        else:
            init_worker_model(xml_path, dt)

        try:
            while accepted < num_trajectories:
                current_batch = min(batch_size, num_trajectories - accepted + max(16, batch_size // 4))
                tasks = [
                    (
                        xml_path,
                        dt,
                        trajectory_length,
                        seed_counter + idx,
                        waypoint_radius,
                        waypoint_qvel_scale,
                        torque_scale,
                        waypoint_tolerance,
                        max_abs_qvel,
                        max_abs_qacc,
                    )
                    for idx in range(current_batch)
                ]
                seed_counter += current_batch

                if use_parallel:
                    chunksize = max(1, len(tasks) // max(1, num_workers * 8))
                    results = list(executor.map(build_bidirectional_reacher_trajectory, tasks, chunksize=chunksize))
                else:
                    results = [build_bidirectional_reacher_trajectory(task) for task in tasks]

                total_generated += len(results)

                for ok, reason, result in results:
                    if ok:
                        write_trajectory_group(h5_file, accepted, result)
                        accepted += 1
                        pbar.update(1)
                        if accepted >= num_trajectories:
                            break
                    else:
                        reject_counts[reason] = reject_counts.get(reason, 0) + 1

                pbar.set_postfix(
                    generated=total_generated,
                    accept_rate=f"{accepted / max(1, total_generated):.1%}",
                )
        finally:
            if executor is not None:
                executor.shutdown(wait=True)

    pbar.close()
    print(
        f"[{split_name}] saved {accepted} trajectories to {output_path} "
        f"(generated {total_generated}, accept rate {accepted / max(1, total_generated):.1%})",
        flush=True,
    )
    for reason, count in sorted(reject_counts.items()):
        print(f"  - {reason}: {count}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bidirectional non-dissipative Reacher HDF5 datasets")
    parser.add_argument(
        "--output_dir",
        "--save_dir",
        dest="output_dir",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/data/reacher_bidirectional_unbounded_j1_dt0p001_len1000",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default="/home/gsang/Projects/hnn_guided_dpf/configs/reacher_non_diss_unbounded_j1.xml",
    )
    parser.add_argument("--train_trajectories", "--num_trajectories", dest="train_trajectories", type=int, default=40000)
    parser.add_argument("--val_trajectories", "--num_val", dest="val_trajectories", type=int, default=2000)
    parser.add_argument("--trajectory_length", type=int, default=1000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--waypoint_radius", type=float, default=0.18)
    parser.add_argument("--waypoint_qvel_scale", type=float, default=0.8)
    parser.add_argument("--torque_scale", type=float, default=0.2)
    parser.add_argument("--waypoint_tolerance", type=float, default=0.01)
    parser.add_argument("--max_abs_qvel", type=float, default=200.0)
    parser.add_argument("--max_abs_qacc", type=float, default=10000.0)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_path = output_dir / f"traj_{args.train_trajectories}-steps_{args.trajectory_length}.h5"
    val_path = output_dir / f"traj_{args.val_trajectories}-steps_{args.trajectory_length}.h5"

    print("Configuration:", flush=True)
    print(f"  output_dir: {output_dir}", flush=True)
    print(f"  xml_path: {args.xml_path}", flush=True)
    print(f"  dt: {args.dt}", flush=True)
    print(f"  trajectory_length: {args.trajectory_length}", flush=True)
    print(f"  train trajectories: {args.train_trajectories}", flush=True)
    print(f"  val trajectories: {args.val_trajectories}", flush=True)
    print(f"  waypoint_radius: {args.waypoint_radius}", flush=True)
    print(f"  waypoint_qvel_scale: {args.waypoint_qvel_scale}", flush=True)
    print(f"  torque_scale: {args.torque_scale}", flush=True)
    print(f"  waypoint_tolerance: {args.waypoint_tolerance}", flush=True)
    print(f"  max_abs_qvel: {args.max_abs_qvel}", flush=True)
    print(f"  max_abs_qacc: {args.max_abs_qacc}", flush=True)
    print(f"  num_workers: {args.num_workers}", flush=True)
    print(f"  batch_size: {args.batch_size}", flush=True)

    generate_split(
        output_path=train_path,
        xml_path=args.xml_path,
        num_trajectories=args.train_trajectories,
        trajectory_length=args.trajectory_length,
        dt=args.dt,
        waypoint_radius=args.waypoint_radius,
        waypoint_qvel_scale=args.waypoint_qvel_scale,
        torque_scale=args.torque_scale,
        waypoint_tolerance=args.waypoint_tolerance,
        max_abs_qvel=args.max_abs_qvel,
        max_abs_qacc=args.max_abs_qacc,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        seed_offset=0,
        split_name="train",
    )

    if args.val_trajectories > 0:
        generate_split(
            output_path=val_path,
            xml_path=args.xml_path,
            num_trajectories=args.val_trajectories,
            trajectory_length=args.trajectory_length,
            dt=args.dt,
            waypoint_radius=args.waypoint_radius,
            waypoint_qvel_scale=args.waypoint_qvel_scale,
            torque_scale=args.torque_scale,
            waypoint_tolerance=args.waypoint_tolerance,
            max_abs_qvel=args.max_abs_qvel,
            max_abs_qacc=args.max_abs_qacc,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            seed_offset=10_000_000,
            split_name="val",
        )


if __name__ == "__main__":
    main()
