#!/home/gsang/miniconda3/envs/perceiver/bin/python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

from scripts.data.run_reacher_goal_prefix_expansion import (
    TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS,
    parse_indices,
)
from src.rl.reacher_goal_rl import (
    DEFAULT_BUDGETS,
    DEFAULT_REWARD_CONFIG,
    evaluate_policy_on_tasks,
)


DEFAULT_EVAL_H5 = (
    "/home/gsang/Projects/hnn_guided_dpf/data/"
    "reacher_exploration_iid_uniform_traj40000_val4000_len1000_v1/val_traj_4000-steps_1000.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stable-Baselines3 SAC or TD3 policy on the fixed Reacher benchmark."
    )
    parser.add_argument("--algorithm", type=str, required=True, choices=["sac", "td3"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--eval_h5_path", type=str, default=DEFAULT_EVAL_H5)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_mode", type=str, default=TASK_MODE_VALIDATION_RANDOM_SOURCE_RANDOM_TARGET_ACROSS_TRAJS)
    parser.add_argument("--task_ids", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--goal_tolerance", type=float, default=0.01)
    parser.add_argument("--stall_patience_steps", type=int, default=500)
    parser.add_argument("--max_episode_steps", type=int, default=5000)
    parser.add_argument("--torque_scale", type=float, default=0.2)
    parser.add_argument("--budget_steps", type=str, default="256,512,1000")
    parser.add_argument("--gif_fps", type=int, default=18)
    parser.add_argument("--gif_max_frames", type=int, default=200)
    parser.add_argument("--render_artifacts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random_future_target_min_initial_distance", type=float, default=0.1)
    parser.add_argument("--cross_traj_sampling_max_tries", type=int, default=128)
    parser.add_argument("--progress_scale", type=float, default=DEFAULT_REWARD_CONFIG["progress_scale"])
    parser.add_argument("--distance_scale", type=float, default=DEFAULT_REWARD_CONFIG["distance_scale"])
    parser.add_argument("--action_l2_weight", type=float, default=DEFAULT_REWARD_CONFIG["action_l2_weight"])
    parser.add_argument("--success_bonus", type=float, default=DEFAULT_REWARD_CONFIG["success_bonus"])
    parser.add_argument("--train_h5_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def parse_budget_steps(text: str) -> tuple[int, ...]:
    values = [int(token.strip()) for token in text.split(",") if token.strip()]
    return tuple(values) if values else DEFAULT_BUDGETS


def reward_config_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        "progress_scale": float(args.progress_scale),
        "distance_scale": float(args.distance_scale),
        "action_l2_weight": float(args.action_l2_weight),
        "success_bonus": float(args.success_bonus),
    }


def main() -> None:
    args = parse_args()
    try:
        from stable_baselines3 import SAC, TD3
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed in the perceiver environment. "
            "Install it first, then rerun this script."
        ) from exc

    algo_cls = SAC if str(args.algorithm).lower() == "sac" else TD3
    model = algo_cls.load(args.checkpoint_path, device=args.device)

    def policy_fn(obs, deterministic: bool):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    metadata = {
        "algorithm": str(args.algorithm).lower(),
        "checkpoint_path": args.checkpoint_path,
    }
    config_path = Path(args.checkpoint_path).with_suffix(".json")
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            metadata["checkpoint_json"] = json.load(handle)

    summary = evaluate_policy_on_tasks(
        algorithm=str(args.algorithm).lower(),
        policy_fn=policy_fn,
        eval_h5_path=args.eval_h5_path,
        output_dir=args.output_dir,
        task_mode=args.task_mode,
        task_ids=parse_indices(args.task_ids),
        seed=int(args.seed),
        goal_tolerance=float(args.goal_tolerance),
        stall_patience_steps=int(args.stall_patience_steps),
        max_episode_steps=int(args.max_episode_steps),
        deterministic_policy=True,
        action_scale=float(args.torque_scale),
        reward_config=reward_config_from_args(args),
        budgets=parse_budget_steps(args.budget_steps),
        gif_fps=int(args.gif_fps),
        gif_max_frames=int(args.gif_max_frames),
        render_artifacts=bool(args.render_artifacts),
        random_future_target_min_initial_distance=float(args.random_future_target_min_initial_distance),
        cross_traj_sampling_max_tries=int(args.cross_traj_sampling_max_tries),
        checkpoint_path=args.checkpoint_path,
        train_h5_path=(args.train_h5_path or None),
        extra_metadata=metadata,
    )
    print(
        "[SB3 eval] algo={} success_rate={:.3f} best_goal_mean={:.5f}".format(
            str(args.algorithm).lower(),
            float(summary["aggregate"]["success_rate"]),
            float(summary["aggregate"]["best_goal_distance"].get("mean", float("inf"))),
        )
    )


if __name__ == "__main__":
    main()
