from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Callable

import gymnasium as gym
from gymnasium import spaces
import h5py
import mujoco
import numpy as np

from scripts.data.run_reacher_goal_prefix_expansion import (
    TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_FUTURE_TARGET,
    TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS,
    ReacherRolloutStepper,
    build_time_series_plot,
    build_validation_random_source_random_future_target_task,
    build_validation_random_source_random_target_across_trajs_task,
    build_workspace_gif,
    build_workspace_plot,
    fingertip_xy_from_qpos_raw,
    normalize_task_mode,
)


DEFAULT_REWARD_CONFIG = {
    "progress_scale": 10.0,
    "distance_scale": 1.0,
    "action_l2_weight": 0.01,
    "success_bonus": 5.0,
}
DEFAULT_BUDGETS = (256, 512, 1000)


def aggregate_metric(rows: list[dict], key: str) -> dict[str, float]:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return {}
    values_np = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values_np.mean()),
        "median": float(np.median(values_np)),
        "min": float(values_np.min()),
        "max": float(values_np.max()),
    }


def compact_summary_row(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key
        not in {
            "rollout_qpos_raw",
            "rollout_mom",
            "rollout_tau",
            "goal_distance_history",
        }
    }


def make_observation(
    qpos_raw: np.ndarray,
    mom: np.ndarray,
    goal_xy: np.ndarray,
) -> np.ndarray:
    ee_xy = fingertip_xy_from_qpos_raw(np.asarray(qpos_raw, dtype=np.float64)[None, :])[0]
    delta_xy = goal_xy - ee_xy
    return np.concatenate(
        [
            np.asarray(qpos_raw, dtype=np.float32),
            np.asarray(mom, dtype=np.float32),
            ee_xy.astype(np.float32, copy=False),
            np.asarray(goal_xy, dtype=np.float32),
            delta_xy.astype(np.float32, copy=False),
        ],
        axis=0,
    )


def list_traj_indices(h5_file: h5py.File) -> list[int]:
    return sorted(
        int(name.split("_")[1])
        for name in h5_file.keys()
        if name.startswith("traj_")
    )


def sample_reacher_task(
    *,
    h5_file: h5py.File,
    task_mode: str,
    rng: np.random.Generator,
    rollout_limit: int,
    min_initial_distance: float,
    cross_traj_sampling_max_tries: int,
) -> dict:
    task_mode = normalize_task_mode(task_mode)
    if task_mode == TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_FUTURE_TARGET:
        traj_indices = list_traj_indices(h5_file)
        if not traj_indices:
            raise ValueError("No traj_* groups were found in the HDF5 file.")
        traj_index = int(traj_indices[int(rng.integers(0, len(traj_indices)))])
        task_seed = int(rng.integers(0, 2**31 - 1))
        return build_validation_random_source_random_future_target_task(
            h5_file=h5_file,
            traj_index=traj_index,
            rollout_limit=int(rollout_limit),
            task_seed=task_seed,
            min_initial_distance=float(min_initial_distance),
        )
    task_seed = int(rng.integers(0, 2**31 - 1))
    return build_validation_random_source_random_target_across_trajs_task(
        h5_file=h5_file,
        task_seed=task_seed,
        rollout_limit=int(rollout_limit),
        min_initial_distance=float(min_initial_distance),
        max_tries=int(cross_traj_sampling_max_tries),
    )


def collect_fixed_tasks(
    *,
    h5_path: str,
    task_mode: str,
    task_ids: list[int],
    seed: int,
    rollout_limit: int,
    min_initial_distance: float,
    cross_traj_sampling_max_tries: int,
) -> list[dict]:
    task_mode = normalize_task_mode(task_mode)
    tasks: list[dict] = []
    with h5py.File(h5_path, "r") as h5_file:
        for task_id in task_ids:
            if task_mode == TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_FUTURE_TARGET:
                task = build_validation_random_source_random_future_target_task(
                    h5_file=h5_file,
                    traj_index=int(task_id),
                    rollout_limit=int(rollout_limit),
                    task_seed=int(seed) + int(task_id),
                    min_initial_distance=float(min_initial_distance),
                )
            else:
                task = build_validation_random_source_random_target_across_trajs_task(
                    h5_file=h5_file,
                    task_seed=int(seed) + int(task_id),
                    rollout_limit=int(rollout_limit),
                    min_initial_distance=float(min_initial_distance),
                    max_tries=int(cross_traj_sampling_max_tries),
                )
            tasks.append(task)
    return tasks


def aggregate_budget_metrics(
    rows: list[dict],
    budgets: tuple[int, ...],
    goal_tolerance: float,
) -> dict[str, dict]:
    budget_payload: dict[str, dict] = {}
    for budget in budgets:
        if budget <= 0:
            continue
        budget_rows: list[dict] = []
        for row in rows:
            goal_history = np.asarray(row["goal_distance_history"], dtype=np.float64)
            trunc_len = min(len(goal_history), int(budget) + 1)
            truncated_history = goal_history[:trunc_len]
            budget_rows.append(
                {
                    "reached_goal": bool(np.any(truncated_history <= float(goal_tolerance))),
                    "best_goal_distance": float(np.min(truncated_history)),
                    "final_goal_distance": float(truncated_history[-1]),
                }
            )
        budget_payload[str(int(budget))] = {
            "success_rate": (
                float(np.mean([float(budget_row["reached_goal"]) for budget_row in budget_rows]))
                if budget_rows
                else 0.0
            ),
            "best_goal_distance": aggregate_metric(budget_rows, "best_goal_distance"),
            "final_goal_distance": aggregate_metric(budget_rows, "final_goal_distance"),
        }
    return budget_payload


def build_rl_summary_payload(
    *,
    algorithm: str,
    checkpoint_path: str | None,
    train_h5_path: str | None,
    eval_h5_path: str,
    task_mode: str,
    task_ids: list[int],
    rows: list[dict],
    seed: int,
    goal_tolerance: float,
    stall_patience_steps: int,
    max_episode_steps: int,
    deterministic_policy: bool,
    action_scale: float,
    reward_config: dict[str, float],
    budgets: tuple[int, ...],
    is_complete: bool,
    compact_rows_only: bool,
    extra_metadata: dict | None = None,
) -> dict:
    payload_rows = [compact_summary_row(row) if compact_rows_only else row for row in rows]
    payload = {
        "algorithm": str(algorithm),
        "checkpoint_path": checkpoint_path,
        "train_h5_path": train_h5_path,
        "h5_path": eval_h5_path,
        "seed": int(seed),
        "task_mode": str(task_mode),
        "task_ids": [int(task_id) for task_id in task_ids],
        "num_total_tasks": int(len(task_ids)),
        "num_completed_tasks": int(len(rows)),
        "completed_task_ids": [int(row["task_id"]) for row in rows],
        "is_complete": bool(is_complete),
        "goal_tolerance": float(goal_tolerance),
        "stall_patience_steps": int(stall_patience_steps),
        "max_episode_steps": int(max_episode_steps),
        "deterministic_policy": bool(deterministic_policy),
        "action_scale": float(action_scale),
        "reward_config": {
            key: float(value) for key, value in reward_config.items()
        },
        "aggregate": {
            "success_rate": float(np.mean([float(row["reached_goal"]) for row in rows])) if rows else 0.0,
            "best_goal_distance": aggregate_metric(rows, "best_goal_distance"),
            "final_goal_distance": aggregate_metric(rows, "final_goal_distance"),
            "qpos_mse_to_replay": aggregate_metric(rows, "qpos_mse_to_replay"),
            "mom_mse_to_replay": aggregate_metric(rows, "mom_mse_to_replay"),
            "ee_xy_mse_to_replay": aggregate_metric(rows, "ee_xy_mse_to_replay"),
            "task_wall_clock_seconds": aggregate_metric(rows, "task_wall_clock_seconds"),
            "budget_metrics": aggregate_budget_metrics(rows, budgets, goal_tolerance),
        },
        "rows": payload_rows,
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class ReacherGoalEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        h5_path: str,
        task_mode: str = TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS,
        torque_scale: float = 0.2,
        goal_tolerance: float = 0.01,
        max_episode_steps: int = 1000,
        stall_patience_steps: int = 500,
        random_future_target_min_initial_distance: float = 0.1,
        cross_traj_sampling_max_tries: int = 128,
        reward_config: dict[str, float] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.h5_path = str(h5_path)
        self.task_mode = normalize_task_mode(task_mode)
        self.torque_scale = float(torque_scale)
        self.goal_tolerance = float(goal_tolerance)
        self.max_episode_steps = int(max_episode_steps)
        self.stall_patience_steps = int(stall_patience_steps)
        self.random_future_target_min_initial_distance = float(random_future_target_min_initial_distance)
        self.cross_traj_sampling_max_tries = int(cross_traj_sampling_max_tries)
        self.reward_config = dict(DEFAULT_REWARD_CONFIG)
        if reward_config:
            self.reward_config.update({key: float(value) for key, value in reward_config.items()})
        self._rng = np.random.default_rng(int(seed))
        self._seed = int(seed)

        self.h5_file = h5py.File(self.h5_path, "r")
        xml_content = self.h5_file.attrs["xml"]
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode()
        self.xml_content = str(xml_content)
        self.trajectory_length_total = int(self.h5_file.attrs["num_steps"])
        self.data_dt = float(self.h5_file.attrs.get("data_dt", self.h5_file.attrs.get("dt", 0.001)))
        self.sim_dt = float(self.h5_file.attrs.get("dt", self.data_dt))
        self.model = mujoco.MjModel.from_xml_string(self.xml_content)
        self.stepper = ReacherRolloutStepper(self.model, dt=self.sim_dt, data_dt=self.data_dt)
        self.coordinate_dim = int(self.model.nu)

        obs_high = np.full((10,), np.inf, dtype=np.float32)
        act_high = np.full((self.coordinate_dim,), self.torque_scale, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)

        self.current_task: dict | None = None
        self.qpos_raw = np.zeros((self.coordinate_dim,), dtype=np.float64)
        self.mom = np.zeros((self.coordinate_dim,), dtype=np.float64)
        self.goal_xy = np.zeros((2,), dtype=np.float64)
        self.step_count = 0
        self.prev_goal_distance = 0.0
        self.best_goal_distance_so_far = 0.0
        self.stall_steps_since_decrease = 0

    def close(self) -> None:
        if getattr(self, "h5_file", None) is not None:
            self.h5_file.close()
            self.h5_file = None

    def _sample_task(self) -> dict:
        return sample_reacher_task(
            h5_file=self.h5_file,
            task_mode=self.task_mode,
            rng=self._rng,
            rollout_limit=max(2, int(self.max_episode_steps) + 1),
            min_initial_distance=float(self.random_future_target_min_initial_distance),
            cross_traj_sampling_max_tries=int(self.cross_traj_sampling_max_tries),
        )

    def _task_info(self) -> dict:
        if self.current_task is None:
            return {}
        return {
            "task_id": int(self.current_task["task_id"]),
            "task_label": str(self.current_task["task_label"]),
            "task_mode": str(self.current_task["task_mode"]),
            "task_metadata": self.current_task.get("metadata", {}),
            "goal_xy": np.asarray(self.goal_xy, dtype=np.float64).copy(),
            "initial_goal_distance": float(self.current_task["metadata"]["initial_goal_distance"]),
        }

    def _build_obs(self) -> np.ndarray:
        return make_observation(self.qpos_raw, self.mom, self.goal_xy)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = int(seed)
            self._rng = np.random.default_rng(int(seed))

        fixed_task = None if options is None else options.get("fixed_task")
        self.current_task = fixed_task if fixed_task is not None else self._sample_task()

        self.qpos_raw = np.asarray(self.current_task["initial_qpos_raw"], dtype=np.float64).copy()
        self.mom = np.asarray(self.current_task["initial_mom"], dtype=np.float64).copy()
        self.goal_xy = np.asarray(self.current_task["goal_xy"], dtype=np.float64).copy()
        self.step_count = 0
        self.prev_goal_distance = float(
            np.linalg.norm(fingertip_xy_from_qpos_raw(self.qpos_raw[None, :])[0] - self.goal_xy)
        )
        self.best_goal_distance_so_far = float(self.prev_goal_distance)
        self.stall_steps_since_decrease = 0
        return self._build_obs(), self._task_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_np = np.asarray(action, dtype=np.float64)
        action_np = np.clip(action_np, -self.torque_scale, self.torque_scale)
        next_qpos_raw, next_mom = self.stepper.step(
            qpos_raw=self.qpos_raw,
            mom=self.mom,
            torque=action_np,
        )
        next_goal_distance = float(
            np.linalg.norm(fingertip_xy_from_qpos_raw(next_qpos_raw[None, :])[0] - self.goal_xy)
        )
        improvement = float(self.prev_goal_distance - next_goal_distance)
        reward = (
            float(self.reward_config["progress_scale"]) * improvement
            - float(self.reward_config["distance_scale"]) * next_goal_distance
            - float(self.reward_config["action_l2_weight"]) * float(np.mean(action_np**2))
        )

        self.qpos_raw = next_qpos_raw
        self.mom = next_mom
        self.step_count += 1

        reached_goal = bool(next_goal_distance <= self.goal_tolerance)
        if reached_goal:
            reward += float(self.reward_config["success_bonus"])

        if next_goal_distance + 1e-12 < self.best_goal_distance_so_far:
            self.best_goal_distance_so_far = float(next_goal_distance)
        if next_goal_distance + 1e-12 < self.prev_goal_distance:
            self.stall_steps_since_decrease = 0
        else:
            self.stall_steps_since_decrease += 1
        self.prev_goal_distance = float(next_goal_distance)

        terminated = reached_goal
        truncated = bool(
            self.step_count >= int(self.max_episode_steps)
            or self.stall_steps_since_decrease >= int(self.stall_patience_steps)
        )
        info = self._task_info()
        info.update(
            {
                "goal_distance": float(next_goal_distance),
                "best_goal_distance_so_far": float(self.best_goal_distance_so_far),
                "reached_goal": bool(reached_goal),
                "stopped_due_to_stall": bool(
                    (not reached_goal)
                    and self.stall_steps_since_decrease >= int(self.stall_patience_steps)
                ),
                "action": action_np.astype(np.float32, copy=False),
                "step_count": int(self.step_count),
            }
        )
        return self._build_obs(), float(reward), terminated, truncated, info


def evaluate_policy_on_tasks(
    *,
    algorithm: str,
    policy_fn: Callable[[np.ndarray, bool], np.ndarray],
    eval_h5_path: str,
    output_dir: str | Path,
    task_mode: str,
    task_ids: list[int],
    seed: int,
    goal_tolerance: float,
    stall_patience_steps: int,
    max_episode_steps: int,
    deterministic_policy: bool,
    action_scale: float,
    reward_config: dict[str, float],
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    gif_fps: int = 18,
    gif_max_frames: int = 200,
    render_artifacts: bool = True,
    random_future_target_min_initial_distance: float = 0.1,
    cross_traj_sampling_max_tries: int = 128,
    checkpoint_path: str | None = None,
    train_h5_path: str | None = None,
    summary_name: str = "reacher_rl_policy_eval_summary.json",
    progress_summary_name: str = "reacher_rl_policy_eval_progress.json",
    extra_metadata: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / summary_name
    progress_path = output_dir / progress_summary_name

    tasks = collect_fixed_tasks(
        h5_path=eval_h5_path,
        task_mode=task_mode,
        task_ids=task_ids,
        seed=seed,
        rollout_limit=max(2, int(max_episode_steps) + 1),
        min_initial_distance=float(random_future_target_min_initial_distance),
        cross_traj_sampling_max_tries=int(cross_traj_sampling_max_tries),
    )

    with h5py.File(eval_h5_path, "r") as h5_file:
        xml_content = h5_file.attrs["xml"]
        if isinstance(xml_content, bytes):
            xml_content = xml_content.decode()
        data_dt = float(h5_file.attrs.get("data_dt", h5_file.attrs.get("dt", 0.001)))
        sim_dt = float(h5_file.attrs.get("dt", data_dt))
        model = mujoco.MjModel.from_xml_string(xml_content)
        stepper = ReacherRolloutStepper(model, dt=sim_dt, data_dt=data_dt)

        rows: list[dict] = []
        for task in tasks:
            task_start_time = time.time()
            goal_xy = np.asarray(task["goal_xy"], dtype=np.float64)
            qpos_prefix_raw = [np.asarray(task["initial_qpos_raw"], dtype=np.float64).copy()]
            mom_prefix = [np.asarray(task["initial_mom"], dtype=np.float64).copy()]
            torque_prefix: list[np.ndarray] = []
            goal_history = [
                float(np.linalg.norm(fingertip_xy_from_qpos_raw(qpos_prefix_raw[0][None, :])[0] - goal_xy))
            ]
            best_goal_distance_so_far = float(goal_history[-1])
            stall_steps_since_decrease = 0
            reached_goal = bool(goal_history[-1] <= float(goal_tolerance))
            stopped_due_to_stall = False

            while not reached_goal and len(torque_prefix) < int(max_episode_steps):
                if stall_steps_since_decrease >= int(stall_patience_steps):
                    stopped_due_to_stall = True
                    break
                obs = make_observation(qpos_prefix_raw[-1], mom_prefix[-1], goal_xy)
                action = np.asarray(policy_fn(obs, deterministic_policy), dtype=np.float64).reshape(-1)
                action = np.clip(action, -float(action_scale), float(action_scale))
                next_qpos_raw, next_mom = stepper.step(
                    qpos_raw=qpos_prefix_raw[-1],
                    mom=mom_prefix[-1],
                    torque=action,
                )
                next_goal_distance = float(
                    np.linalg.norm(fingertip_xy_from_qpos_raw(next_qpos_raw[None, :])[0] - goal_xy)
                )
                qpos_prefix_raw.append(next_qpos_raw)
                mom_prefix.append(next_mom)
                torque_prefix.append(action.astype(np.float64, copy=False))
                goal_history.append(next_goal_distance)
                reached_goal = bool(next_goal_distance <= float(goal_tolerance))
                if next_goal_distance + 1e-12 < best_goal_distance_so_far:
                    best_goal_distance_so_far = float(next_goal_distance)
                if next_goal_distance + 1e-12 < goal_history[-2]:
                    stall_steps_since_decrease = 0
                else:
                    stall_steps_since_decrease += 1

            rollout_qpos_raw = np.asarray(qpos_prefix_raw, dtype=np.float64)
            rollout_mom = np.asarray(mom_prefix, dtype=np.float64)
            rollout_tau = (
                np.asarray(torque_prefix, dtype=np.float64)
                if torque_prefix
                else np.zeros((0, rollout_qpos_raw.shape[-1]), dtype=np.float64)
            )
            goal_history_np = np.asarray(goal_history, dtype=np.float64)
            reference_qpos_raw = np.asarray(task["reference_qpos_raw"], dtype=np.float64)
            reference_mom = np.asarray(task["reference_mom"], dtype=np.float64)
            common_horizon = min(len(rollout_qpos_raw), len(reference_qpos_raw))
            qpos_mse = float(
                np.mean((rollout_qpos_raw[:common_horizon] - reference_qpos_raw[:common_horizon]) ** 2)
            )
            mom_mse = float(
                np.mean((rollout_mom[:common_horizon] - reference_mom[:common_horizon]) ** 2)
            )
            ee_mse = float(
                np.mean(
                    (
                        fingertip_xy_from_qpos_raw(rollout_qpos_raw[:common_horizon])
                        - fingertip_xy_from_qpos_raw(reference_qpos_raw[:common_horizon])
                    )
                    ** 2
                )
            )

            workspace_plot = output_dir / f"{task['task_label']}_workspace.png"
            workspace_gif = output_dir / f"{task['task_label']}_workspace.gif"
            timeseries_plot = output_dir / f"{task['task_label']}_timeseries.png"
            if bool(render_artifacts):
                build_workspace_plot(
                    rollout_qpos_raw=rollout_qpos_raw,
                    goal_xy=goal_xy,
                    goal_tolerance=float(goal_tolerance),
                    best_goal_distance=float(np.min(goal_history_np)),
                    final_goal_distance=float(goal_history_np[-1]),
                    title=f"{algorithm.upper()} rollout: {task['task_label']}",
                    save_path=workspace_plot,
                )
                build_workspace_gif(
                    rollout_qpos_raw=rollout_qpos_raw,
                    goal_xy=goal_xy,
                    goal_tolerance=float(goal_tolerance),
                    goal_distance=goal_history_np,
                    title=f"{algorithm.upper()} rollout: {task['task_label']}",
                    save_path=workspace_gif,
                    fps=int(gif_fps),
                    max_frames=int(gif_max_frames),
                )
                build_time_series_plot(
                    rollout_qpos_raw=rollout_qpos_raw,
                    rollout_mom=rollout_mom,
                    goal_xy=goal_xy,
                    goal_tolerance=float(goal_tolerance),
                    goal_distance=goal_history_np,
                    best_goal_distance=float(np.min(goal_history_np)),
                    final_goal_distance=float(goal_history_np[-1]),
                    save_path=timeseries_plot,
                )

            row = {
                "task_id": int(task["task_id"]),
                "task_label": str(task["task_label"]),
                "task_mode": str(task["task_mode"]),
                "goal_xy": goal_xy.tolist(),
                "initial_goal_distance": float(goal_history_np[0]),
                "best_goal_distance": float(np.min(goal_history_np)),
                "final_goal_distance": float(goal_history_np[-1]),
                "reached_goal": bool(reached_goal),
                "stopped_due_to_stall": bool(stopped_due_to_stall),
                "steps_taken": int(len(rollout_tau)),
                "qpos_mse_to_replay": float(qpos_mse),
                "mom_mse_to_replay": float(mom_mse),
                "ee_xy_mse_to_replay": float(ee_mse),
                "workspace_plot": str(workspace_plot) if bool(render_artifacts) else None,
                "workspace_gif": str(workspace_gif) if bool(render_artifacts) else None,
                "timeseries_plot": str(timeseries_plot) if bool(render_artifacts) else None,
                "task_wall_clock_seconds": float(time.time() - task_start_time),
                "goal_distance_history": goal_history_np.tolist(),
                "rollout_qpos_raw": rollout_qpos_raw.tolist(),
                "rollout_mom": rollout_mom.tolist(),
                "rollout_tau": rollout_tau.tolist(),
                "task_metadata": task.get("metadata", {}),
            }
            rows.append(row)
            progress_payload = build_rl_summary_payload(
                algorithm=algorithm,
                checkpoint_path=checkpoint_path,
                train_h5_path=train_h5_path,
                eval_h5_path=eval_h5_path,
                task_mode=task_mode,
                task_ids=task_ids,
                rows=rows,
                seed=seed,
                goal_tolerance=goal_tolerance,
                stall_patience_steps=stall_patience_steps,
                max_episode_steps=max_episode_steps,
                deterministic_policy=deterministic_policy,
                action_scale=action_scale,
                reward_config=reward_config,
                budgets=budgets,
                is_complete=False,
                compact_rows_only=True,
                extra_metadata=extra_metadata,
            )
            write_json(progress_path, progress_payload)

    summary_payload = build_rl_summary_payload(
        algorithm=algorithm,
        checkpoint_path=checkpoint_path,
        train_h5_path=train_h5_path,
        eval_h5_path=eval_h5_path,
        task_mode=task_mode,
        task_ids=task_ids,
        rows=rows,
        seed=seed,
        goal_tolerance=goal_tolerance,
        stall_patience_steps=stall_patience_steps,
        max_episode_steps=max_episode_steps,
        deterministic_policy=deterministic_policy,
        action_scale=action_scale,
        reward_config=reward_config,
        budgets=budgets,
        is_complete=True,
        compact_rows_only=False,
        extra_metadata=extra_metadata,
    )
    write_json(summary_path, summary_payload)
    write_json(
        progress_path,
        build_rl_summary_payload(
            algorithm=algorithm,
            checkpoint_path=checkpoint_path,
            train_h5_path=train_h5_path,
            eval_h5_path=eval_h5_path,
            task_mode=task_mode,
            task_ids=task_ids,
            rows=rows,
            seed=seed,
            goal_tolerance=goal_tolerance,
            stall_patience_steps=stall_patience_steps,
            max_episode_steps=max_episode_steps,
            deterministic_policy=deterministic_policy,
            action_scale=action_scale,
            reward_config=reward_config,
            budgets=budgets,
            is_complete=True,
            compact_rows_only=True,
            extra_metadata=extra_metadata,
        ),
    )
    return summary_payload
