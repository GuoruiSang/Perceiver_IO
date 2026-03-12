#!/usr/bin/env python3
"""Run diffusion-vs-DPF/HNN evaluation across one or more systems."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.diffusion_eval_shared import build_system_eval_config, run_system_eval


def _env_csv(name: str, default_values: list[str]) -> list[str]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return list(default_values)
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values if values else list(default_values)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    systems = _env_csv("EVAL_SYSTEMS", ["2dof", "3dof"])
    for system in systems:
        cfg = build_system_eval_config(system, project_root_override=project_root)
        run_system_eval(cfg)


if __name__ == "__main__":
    main()
