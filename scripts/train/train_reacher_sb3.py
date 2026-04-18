#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import numpy as np
import torch

from scripts.data.run_reacher_goal_prefix_expansion import (
    TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS,
    parse_indices,
)
from src.rl.reacher_goal_rl import (
    DEFAULT_BUDGETS,
    DEFAULT_REWARD_CONFIG,
    ReacherGoalEnv,
    evaluate_policy_on_tasks,
)


DEFAULT_TRAIN_H5 = (
    "/home/gsang/Projects/hnn_guided_dpf/data/"
    "reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/train_traj_40000-steps_1000.h5"
)
DEFAULT_EVAL_H5 = (
    "/home/gsang/Projects/hnn_guided_dpf/data/"
    "reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5"
)
DEFAULT_OUTPUT_ROOT = "/home/gsang/Projects/hnn_guided_dpf/checkpoints/reacher"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Stable-Baselines3 SAC or TD3 goal-conditioned Reacher baseline "
            "using the IID-uniform exploration dataset."
        )
    )
    parser.add_argument("--algorithm", type=str, default="sac", choices=["sac", "td3"])
    parser.add_argument("--train_h5_path", type=str, default=DEFAULT_TRAIN_H5)
    parser.add_argument("--eval_h5_path", type=str, default=DEFAULT_EVAL_H5)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--train_task_mode", type=str, default=TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS)
    parser.add_argument("--eval_task_mode", type=str, default=TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS)
    parser.add_argument("--eval_task_ids", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--total_timesteps", type=int, default=300000)
    parser.add_argument("--learning_starts", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=500000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--policy_hidden_dim", type=int, default=256)
    parser.add_argument("--policy_hidden_depth", type=int, default=2)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--torque_scale", type=float, default=0.2)
    parser.add_argument("--goal_tolerance", type=float, default=0.01)
    parser.add_argument("--train_max_episode_steps", type=int, default=1000)
    parser.add_argument("--eval_max_episode_steps", type=int, default=5000)
    parser.add_argument("--stall_patience_steps", type=int, default=500)
    parser.add_argument("--random_future_target_min_initial_distance", type=float, default=0.1)
    parser.add_argument("--cross_traj_sampling_max_tries", type=int, default=128)
    parser.add_argument("--progress_scale", type=float, default=DEFAULT_REWARD_CONFIG["progress_scale"])
    parser.add_argument("--distance_scale", type=float, default=DEFAULT_REWARD_CONFIG["distance_scale"])
    parser.add_argument("--action_l2_weight", type=float, default=DEFAULT_REWARD_CONFIG["action_l2_weight"])
    parser.add_argument("--success_bonus", type=float, default=DEFAULT_REWARD_CONFIG["success_bonus"])
    parser.add_argument("--eval_every_steps", type=int, default=25000)
    parser.add_argument("--save_every_steps", type=int, default=50000)
    parser.add_argument("--gif_fps", type=int, default=18)
    parser.add_argument("--gif_max_frames", type=int, default=200)
    parser.add_argument("--budget_steps", type=str, default="256,512,1000")
    parser.add_argument("--td3_action_noise_sigma", type=float, default=0.1)
    parser.add_argument("--use_her", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--her_n_sampled_goal", type=int, default=4)
    parser.add_argument("--her_goal_selection_strategy", type=str, default="future")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_budget_steps(text: str) -> tuple[int, ...]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    return tuple(values) if values else DEFAULT_BUDGETS


def default_output_dir(algorithm: str) -> str:
    return str(Path(DEFAULT_OUTPUT_ROOT) / f"{algorithm}_exploration_iid_uniform_len1000_v1")


def reward_config_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "progress_scale": float(args.progress_scale),
        "distance_scale": float(args.distance_scale),
        "action_l2_weight": float(args.action_l2_weight),
        "success_bonus": float(args.success_bonus),
    }


def make_env_fn(
    *,
    rank: int,
    args: argparse.Namespace,
    reward_config: dict[str, float],
):
    def _factory():
        return ReacherGoalEnv(
            h5_path=args.train_h5_path,
            task_mode=args.train_task_mode,
            torque_scale=float(args.torque_scale),
            goal_tolerance=float(args.goal_tolerance),
            max_episode_steps=int(args.train_max_episode_steps),
            stall_patience_steps=int(args.stall_patience_steps),
            random_future_target_min_initial_distance=float(args.random_future_target_min_initial_distance),
            cross_traj_sampling_max_tries=int(args.cross_traj_sampling_max_tries),
            reward_config=reward_config,
            goal_conditioned=bool(args.use_her),
            seed=int(args.seed) + int(rank),
        )

    return _factory


def model_predict_fn(model):
    def _predict(obs: np.ndarray, deterministic: bool) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    return _predict


def main() -> None:
    args = parse_args()

    try:
        from stable_baselines3 import HerReplayBuffer, SAC, TD3
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.noise import NormalActionNoise
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed in the perceiver environment. "
            "Install it first, then rerun this script."
        ) from exc

    output_dir = Path(args.output_dir or default_output_dir(args.algorithm))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    evaluations_dir = output_dir / "evaluations"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    evaluations_dir.mkdir(parents=True, exist_ok=True)

    reward_config = reward_config_from_args(args)
    budgets = parse_budget_steps(args.budget_steps)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    effective_learning_starts = int(args.learning_starts)
    if bool(args.use_her):
        effective_learning_starts = max(
            effective_learning_starts,
            int(args.train_max_episode_steps) * max(1, int(args.num_envs)) + 1,
        )
        if effective_learning_starts != int(args.learning_starts):
            print(
                f"[SB3] bumped learning_starts from {int(args.learning_starts)} "
                f"to {effective_learning_starts} for HER."
            )

    config_payload = dict(vars(args))
    config_payload["reward_config"] = reward_config
    config_payload["effective_learning_starts"] = int(effective_learning_starts)
    (output_dir / "training_config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    num_envs = max(1, int(args.num_envs))
    env_fns = [make_env_fn(rank=rank, args=args, reward_config=reward_config) for rank in range(num_envs)]
    if num_envs == 1:
        vec_env = DummyVecEnv([lambda fn=env_fns[0]: fn()])
    else:
        vec_env = SubprocVecEnv([lambda fn=fn: fn() for fn in env_fns])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = {
        "net_arch": {
            "pi": [int(args.policy_hidden_dim)] * int(args.policy_hidden_depth),
            "qf": [int(args.policy_hidden_dim)] * int(args.policy_hidden_depth),
        }
    }

    algorithm_name = str(args.algorithm).lower()
    algo_cls = SAC if algorithm_name == "sac" else TD3
    model_kwargs = {
        "policy": "MultiInputPolicy" if bool(args.use_her) else "MlpPolicy",
        "env": vec_env,
        "learning_rate": float(args.learning_rate),
        "buffer_size": int(args.buffer_size),
        "learning_starts": int(effective_learning_starts),
        "batch_size": int(args.batch_size),
        "tau": float(args.tau),
        "gamma": float(args.gamma),
        "train_freq": int(args.train_freq),
        "gradient_steps": int(args.gradient_steps),
        "policy_kwargs": policy_kwargs,
        "verbose": 1,
        "seed": int(args.seed),
        "device": args.device,
    }
    if bool(args.use_her):
        model_kwargs["replay_buffer_class"] = HerReplayBuffer
        model_kwargs["replay_buffer_kwargs"] = {
            "n_sampled_goal": int(args.her_n_sampled_goal),
            "goal_selection_strategy": str(args.her_goal_selection_strategy),
            "copy_info_dict": True,
        }
    if algorithm_name == "td3":
        noise_sigma = float(args.td3_action_noise_sigma) * float(args.torque_scale)
        action_noise = NormalActionNoise(
            mean=np.zeros((2,), dtype=np.float32),
            sigma=np.full((2,), noise_sigma, dtype=np.float32),
        )
        model_kwargs["action_noise"] = action_noise

    model = algo_cls(**model_kwargs)
    train_start = time.time()
    best_score: tuple[float, float] | None = None

    class FixedTaskEvalCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.last_save_step = 0

        def _save_checkpoint(self, name: str) -> Path:
            base_path = checkpoints_dir / name
            self.model.save(str(base_path))
            return base_path.with_suffix(".zip")

        def _run_eval(self, step: int, checkpoint_path: str) -> dict:
            eval_dir = evaluations_dir / f"step_{step:08d}"
            return evaluate_policy_on_tasks(
                algorithm=algorithm_name,
                policy_fn=model_predict_fn(self.model),
                eval_h5_path=args.eval_h5_path,
                output_dir=eval_dir,
                task_mode=args.eval_task_mode,
                task_ids=parse_indices(args.eval_task_ids),
                seed=int(args.seed),
                goal_tolerance=float(args.goal_tolerance),
                stall_patience_steps=int(args.stall_patience_steps),
                max_episode_steps=int(args.eval_max_episode_steps),
                deterministic_policy=True,
                action_scale=float(args.torque_scale),
                reward_config=reward_config,
                budgets=budgets,
                gif_fps=int(args.gif_fps),
                gif_max_frames=int(args.gif_max_frames),
                render_artifacts=False,
                random_future_target_min_initial_distance=float(args.random_future_target_min_initial_distance),
                cross_traj_sampling_max_tries=int(args.cross_traj_sampling_max_tries),
                checkpoint_path=checkpoint_path,
                train_h5_path=args.train_h5_path,
                goal_conditioned_policy=bool(args.use_her),
                extra_metadata={
                    "training_step": int(step),
                    "wall_clock_seconds": float(time.time() - train_start),
                    "algorithm": algorithm_name,
                    "use_her": bool(args.use_her),
                },
            )

        def _on_step(self) -> bool:
            nonlocal best_score
            step = int(self.num_timesteps)

            if int(args.save_every_steps) > 0 and step - self.last_save_step >= int(args.save_every_steps):
                self._save_checkpoint(f"{algorithm_name}_step_{step:08d}")
                self._save_checkpoint("latest")
                self.last_save_step = step

            if int(args.eval_every_steps) > 0 and step % int(args.eval_every_steps) == 0:
                latest_path = str(self._save_checkpoint("latest"))
                summary = self._run_eval(step, latest_path)
                success_rate = float(summary["aggregate"]["success_rate"])
                best_goal_mean = float(summary["aggregate"]["best_goal_distance"].get("mean", float("inf")))
                score = (success_rate, -best_goal_mean)
                if best_score is None or score > best_score:
                    best_score = score
                    self._save_checkpoint("best_eval")
                print(
                    "[SB3 eval] algo={} step={} success_rate={:.3f} best_goal_mean={:.5f}".format(
                        algorithm_name,
                        step,
                        success_rate,
                        best_goal_mean,
                    )
                )
            return True

    callback = FixedTaskEvalCallback()
    print(f"[SB3] algorithm={algorithm_name}")
    print(f"[SB3] output_dir={output_dir}")
    print(f"[SB3] train_h5={args.train_h5_path}")
    print(f"[SB3] eval_h5={args.eval_h5_path}")
    print(f"[SB3] num_envs={num_envs}")
    print(f"[SB3] use_her={bool(args.use_her)}")
    print(f"[SB3] learning_starts={effective_learning_starts}")

    model.learn(total_timesteps=int(args.total_timesteps), callback=callback, progress_bar=False)
    latest_path = checkpoints_dir / "latest.zip"
    model.save(str(latest_path.with_suffix("")))
    summary = callback._run_eval(int(model.num_timesteps), str(latest_path))
    success_rate = float(summary["aggregate"]["success_rate"])
    best_goal_mean = float(summary["aggregate"]["best_goal_distance"].get("mean", float("inf")))
    print(
        "[SB3] finished algo={} success_rate={:.3f} best_goal_mean={:.5f} wall_clock={:.1f}s".format(
            algorithm_name,
            success_rate,
            best_goal_mean,
            time.time() - train_start,
        )
    )
    vec_env.close()


if __name__ == "__main__":
    main()
