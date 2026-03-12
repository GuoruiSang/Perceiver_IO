"""Shared guidance presets and eval-time configuration helpers."""

from __future__ import annotations

import os

from scripts.system_eval_utils import (
    HAMRES_MIN_SCALE_P,
    HAMRES_MIN_SCALE_Q,
    HAMRES_PSEUDO_HUBER_DELTA,
    HAMRES_SMOOTH_SIGMA,
)

SMOOTH_SIGMA = 0.0
SMOOTH_GUIDANCE_ONLY = False
SMOOTH_LAST_STEP_ONLY = False

GUIDANCE_PRESET = "strategy2_best"
GUIDANCE_METHOD = "strategy2"
GUIDANCE_NUM_CANDIDATES = 16
OPTIMIZE_TARGET = "both"
GUIDANCE_ENERGY_MODE = "one_step"
GUIDANCE_TRUST_LAMBDA = 0.0
GUIDANCE_HAMRES_SMOOTH_SIGMA = 1.0
GUIDANCE_HAMRES_DELTA = 1.0
GUIDANCE_HAMRES_MIN_SCALE_Q = 1e-3
GUIDANCE_HAMRES_MIN_SCALE_P = 1e-3
ALPHA_Q = 1e-2
ALPHA_P = 1e-2
GUIDANCE_NORMALIZE_GRAD = True
GUIDANCE_JOINT_UPDATE = True
LANGEVIN_STEP_SIZE = 0.0
LANGEVIN_NOISE_SCALE = 0.0
CHUNK_LENGTH = 0
NUM_DIFFUSION_STEPS = 20

HAMRES_SMOOTH_SIGMA_CFG = None
HAMRES_DELTA_CFG = None
HAMRES_MIN_SCALE_Q_CFG = None
HAMRES_MIN_SCALE_P_CFG = None


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw if raw else default


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


SMOOTH_SIGMA = _env_float("SMOOTH_SIGMA", SMOOTH_SIGMA)
SMOOTH_GUIDANCE_ONLY = _env_bool("SMOOTH_GUIDANCE_ONLY", SMOOTH_GUIDANCE_ONLY)
SMOOTH_LAST_STEP_ONLY = _env_bool("SMOOTH_LAST_STEP_ONLY", SMOOTH_LAST_STEP_ONLY)
GUIDANCE_PRESET = _env_str("GUIDANCE_PRESET", GUIDANCE_PRESET)
GUIDANCE_METHOD = _env_str("GUIDANCE_METHOD", GUIDANCE_METHOD)
GUIDANCE_NUM_CANDIDATES = _env_int("GUIDANCE_NUM_CANDIDATES", GUIDANCE_NUM_CANDIDATES)
OPTIMIZE_TARGET = _env_str("OPTIMIZE_TARGET", OPTIMIZE_TARGET)
GUIDANCE_ENERGY_MODE = _env_str("GUIDANCE_ENERGY_MODE", GUIDANCE_ENERGY_MODE)
GUIDANCE_TRUST_LAMBDA = _env_float("GUIDANCE_TRUST_LAMBDA", GUIDANCE_TRUST_LAMBDA)
GUIDANCE_HAMRES_SMOOTH_SIGMA = _env_float("GUIDANCE_HAMRES_SMOOTH_SIGMA", GUIDANCE_HAMRES_SMOOTH_SIGMA)
GUIDANCE_HAMRES_DELTA = _env_float("GUIDANCE_HAMRES_DELTA", GUIDANCE_HAMRES_DELTA)
GUIDANCE_HAMRES_MIN_SCALE_Q = _env_float("GUIDANCE_HAMRES_MIN_SCALE_Q", GUIDANCE_HAMRES_MIN_SCALE_Q)
GUIDANCE_HAMRES_MIN_SCALE_P = _env_float("GUIDANCE_HAMRES_MIN_SCALE_P", GUIDANCE_HAMRES_MIN_SCALE_P)
ALPHA_Q = _env_float("ALPHA_Q", ALPHA_Q)
ALPHA_P = _env_float("ALPHA_P", ALPHA_P)
GUIDANCE_NORMALIZE_GRAD = _env_bool("GUIDANCE_NORMALIZE_GRAD", GUIDANCE_NORMALIZE_GRAD)
GUIDANCE_JOINT_UPDATE = _env_bool("GUIDANCE_JOINT_UPDATE", GUIDANCE_JOINT_UPDATE)
LANGEVIN_STEP_SIZE = _env_float("LANGEVIN_STEP_SIZE", LANGEVIN_STEP_SIZE)
LANGEVIN_NOISE_SCALE = _env_float("LANGEVIN_NOISE_SCALE", LANGEVIN_NOISE_SCALE)
CHUNK_LENGTH = _env_int("CHUNK_LENGTH", CHUNK_LENGTH)
NUM_DIFFUSION_STEPS = _env_int("NUM_DIFFUSION_STEPS", NUM_DIFFUSION_STEPS)

GUIDANCE_PRESETS = {
    "strategy2_best": dict(
        guidance_method="strategy2",
        optimize_target="both",
        guidance_energy_mode="one_step",
        guidance_trust_lambda=1e-3,
        guidance_num_candidates=16,
        guidance_hamres_smooth_sigma=1.0,
        guidance_hamres_delta=1.0,
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
    ),
    "strategy1_best": dict(
        guidance_method="strategy1",
        optimize_target="both",
        guidance_energy_mode="robust_hamres",
        guidance_trust_lambda=1e-3,
        guidance_num_candidates=16,
        guidance_hamres_smooth_sigma=0.5,
        guidance_hamres_delta=2.0,
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
    ),
    "one_step_best": dict(
        guidance_method="strategy2",
        optimize_target="both",
        guidance_energy_mode="one_step",
        guidance_trust_lambda=1e-3,
        guidance_num_candidates=16,
        guidance_hamres_smooth_sigma=1.0,
        guidance_hamres_delta=1.0,
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
    ),
    "robust_r3_sigma05": dict(
        guidance_method="strategy1",
        optimize_target="both",
        guidance_energy_mode="robust_hamres",
        guidance_trust_lambda=1e-3,
        guidance_num_candidates=16,
        guidance_hamres_smooth_sigma=0.5,
        guidance_hamres_delta=1.0,
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
    ),
    "robust_r9_comboa": dict(
        guidance_method="strategy1",
        optimize_target="both",
        guidance_energy_mode="robust_hamres",
        guidance_trust_lambda=1e-3,
        guidance_num_candidates=16,
        guidance_hamres_smooth_sigma=0.5,
        guidance_hamres_delta=2.0,
        guidance_hamres_min_scale_q=1e-3,
        guidance_hamres_min_scale_p=1e-3,
    ),
}


def resolve_guidance_preset_key(system: str | None = None, auto_map: dict[str | None, str] | None = None) -> str:
    preset_key = GUIDANCE_PRESET.lower()
    if preset_key != "auto":
        return preset_key

    if auto_map is None:
        auto_map = {
            "2dof": "strategy2_best",
            "3dof": "strategy2_best",
        }
    target_system = system
    return auto_map.get(target_system, auto_map.get(None, "strategy2_best"))


def resolve_guidance_config(system: str | None = None, auto_map: dict[str | None, str] | None = None):
    preset_key = resolve_guidance_preset_key(system=system, auto_map=auto_map)
    if preset_key == "custom":
        return {
            "guidance_method": GUIDANCE_METHOD,
            "optimize_target": OPTIMIZE_TARGET,
            "guidance_energy_mode": GUIDANCE_ENERGY_MODE,
            "guidance_num_candidates": GUIDANCE_NUM_CANDIDATES,
            "guidance_trust_lambda": GUIDANCE_TRUST_LAMBDA,
            "guidance_hamres_smooth_sigma": GUIDANCE_HAMRES_SMOOTH_SIGMA,
            "guidance_hamres_delta": GUIDANCE_HAMRES_DELTA,
            "guidance_hamres_min_scale_q": GUIDANCE_HAMRES_MIN_SCALE_Q,
            "guidance_hamres_min_scale_p": GUIDANCE_HAMRES_MIN_SCALE_P,
            "guidance_normalize_grad": GUIDANCE_NORMALIZE_GRAD,
            "guidance_joint_update": GUIDANCE_JOINT_UPDATE,
        }, preset_key

    if preset_key not in GUIDANCE_PRESETS:
        raise ValueError(
            f"Unknown GUIDANCE_PRESET={GUIDANCE_PRESET!r}. "
            f"Choose from {list(GUIDANCE_PRESETS.keys()) + ['custom']}"
        )

    preset = GUIDANCE_PRESETS[preset_key]
    return {
        "guidance_method": preset["guidance_method"],
        "optimize_target": preset["optimize_target"],
        "guidance_energy_mode": preset["guidance_energy_mode"],
        "guidance_num_candidates": preset.get("guidance_num_candidates", 16),
        "guidance_trust_lambda": preset["guidance_trust_lambda"],
        "guidance_hamres_smooth_sigma": preset["guidance_hamres_smooth_sigma"],
        "guidance_hamres_delta": preset["guidance_hamres_delta"],
        "guidance_hamres_min_scale_q": preset["guidance_hamres_min_scale_q"],
        "guidance_hamres_min_scale_p": preset["guidance_hamres_min_scale_p"],
        "guidance_normalize_grad": preset.get("guidance_normalize_grad", GUIDANCE_NORMALIZE_GRAD),
        "guidance_joint_update": preset.get("guidance_joint_update", GUIDANCE_JOINT_UPDATE),
    }, preset_key


def build_sampling_shared_kwargs() -> dict[str, object]:
    shared_kwargs: dict[str, object] = {}
    if NUM_DIFFUSION_STEPS is not None:
        shared_kwargs["num_diffusion_steps"] = NUM_DIFFUSION_STEPS
    return shared_kwargs


def build_unguided_kwargs() -> dict[str, object]:
    return dict(hnn=None, smooth_sigma=SMOOTH_SIGMA, **build_sampling_shared_kwargs())


def build_guided_kwargs(hnn, guidance: dict[str, object]) -> dict[str, object]:
    return dict(
        hnn=hnn,
        guidance_method=guidance["guidance_method"],
        guidance_num_candidates=guidance["guidance_num_candidates"],
        optimize_target=guidance["optimize_target"],
        guidance_energy_mode=guidance["guidance_energy_mode"],
        guidance_hamres_smooth_sigma=guidance["guidance_hamres_smooth_sigma"],
        guidance_hamres_delta=guidance["guidance_hamres_delta"],
        guidance_hamres_min_scale_q=guidance["guidance_hamres_min_scale_q"],
        guidance_hamres_min_scale_p=guidance["guidance_hamres_min_scale_p"],
        guidance_trust_lambda=guidance["guidance_trust_lambda"],
        guidance_normalize_grad=guidance["guidance_normalize_grad"],
        guidance_joint_update=guidance["guidance_joint_update"],
        smooth_sigma=SMOOTH_SIGMA,
        smooth_guidance_only=SMOOTH_GUIDANCE_ONLY,
        smooth_last_step_only=SMOOTH_LAST_STEP_ONLY,
        alpha_q=ALPHA_Q,
        alpha_p=ALPHA_P,
        langevin_step_size=LANGEVIN_STEP_SIZE,
        langevin_noise_scale=LANGEVIN_NOISE_SCALE,
        chunk_length=CHUNK_LENGTH,
        **build_sampling_shared_kwargs(),
    )


def build_hamres_eval_kwargs() -> dict[str, float]:
    return {
        "smooth_sigma": HAMRES_SMOOTH_SIGMA if HAMRES_SMOOTH_SIGMA_CFG is None else HAMRES_SMOOTH_SIGMA_CFG,
        "delta": HAMRES_PSEUDO_HUBER_DELTA if HAMRES_DELTA_CFG is None else HAMRES_DELTA_CFG,
        "min_scale_q": HAMRES_MIN_SCALE_Q if HAMRES_MIN_SCALE_Q_CFG is None else HAMRES_MIN_SCALE_Q_CFG,
        "min_scale_p": HAMRES_MIN_SCALE_P if HAMRES_MIN_SCALE_P_CFG is None else HAMRES_MIN_SCALE_P_CFG,
    }
