"""Helpers for dataset-specific qpos representations."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


RAW_QPOS = "raw"
REACHER_Q0_SINCOS_Q1 = "reacher_q0_sincos_q1"


def infer_qpos_representation_from_xml(xml_content: Optional[str]) -> str:
    if not xml_content:
        return RAW_QPOS
    if (
        'model="reacher"' in xml_content
        and 'name="joint0"' in xml_content
        and 'name="joint1"' in xml_content
        and 'name="target_x"' in xml_content
        and 'name="target_y"' in xml_content
    ):
        return REACHER_Q0_SINCOS_Q1
    return RAW_QPOS


def encoded_qpos_dim(raw_qpos_dim: int, qpos_representation: str) -> int:
    if qpos_representation == REACHER_Q0_SINCOS_Q1:
        if raw_qpos_dim != 2:
            raise ValueError(
                f"{REACHER_Q0_SINCOS_Q1} expects raw qpos dim 2, got {raw_qpos_dim}"
            )
        return 3
    return raw_qpos_dim


def raw_qpos_dim(encoded_qpos_dim_value: int, qpos_representation: str) -> int:
    if qpos_representation == REACHER_Q0_SINCOS_Q1:
        if encoded_qpos_dim_value != 3:
            raise ValueError(
                f"{REACHER_Q0_SINCOS_Q1} expects encoded qpos dim 3, got {encoded_qpos_dim_value}"
            )
        return 2
    return encoded_qpos_dim_value


def encode_qpos_array(qpos: np.ndarray, qpos_representation: str) -> np.ndarray:
    if qpos_representation == REACHER_Q0_SINCOS_Q1:
        q0 = qpos[..., 0]
        q1 = qpos[..., 1]
        return np.stack([np.sin(q0), np.cos(q0), q1], axis=-1).astype(qpos.dtype, copy=False)
    return qpos


def decode_qpos_array(qpos: np.ndarray, qpos_representation: str) -> np.ndarray:
    if qpos_representation == REACHER_Q0_SINCOS_Q1:
        sin_q0 = qpos[..., 0]
        cos_q0 = qpos[..., 1]
        q1 = qpos[..., 2]
        q0 = np.arctan2(sin_q0, cos_q0)
        return np.stack([q0, q1], axis=-1).astype(qpos.dtype, copy=False)
    return qpos


def encode_qpos_tensor(qpos: torch.Tensor, qpos_representation: str) -> torch.Tensor:
    if qpos_representation == REACHER_Q0_SINCOS_Q1:
        q0 = qpos[..., 0]
        q1 = qpos[..., 1]
        return torch.stack([torch.sin(q0), torch.cos(q0), q1], dim=-1)
    return qpos


def decode_qpos_tensor(qpos: torch.Tensor, qpos_representation: str) -> torch.Tensor:
    if qpos_representation == REACHER_Q0_SINCOS_Q1:
        sin_q0 = qpos[..., 0]
        cos_q0 = qpos[..., 1]
        q1 = qpos[..., 2]
        q0 = torch.atan2(sin_q0, cos_q0)
        return torch.stack([q0, q1], dim=-1)
    return qpos


def override_qpos_normalization_stats(
    qpos_min: torch.Tensor,
    qpos_max: torch.Tensor,
    qpos_representation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if qpos_representation == REACHER_Q0_SINCOS_Q1:
        qpos_min = qpos_min.clone()
        qpos_max = qpos_max.clone()
        qpos_min[0] = -1.0
        qpos_max[0] = 1.0
        qpos_min[1] = -1.0
        qpos_max[1] = 1.0
    return qpos_min, qpos_max
