#!/usr/bin/env python
"""Offline baseline-vs-residual comparison for the Proactive DynamicVLA MVP.

This is the fallback comparison loop when the closed-loop Isaac/DOM simulator is
not available. It runs both policies on the same held-out DOM samples and reports
offline action and object-relative tracking metrics.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml

PROJECT_HOME = Path(__file__).resolve().parents[1]
if str(PROJECT_HOME) not in sys.path:
    sys.path.insert(0, str(PROJECT_HOME))

CI95_Z = 1.959963984540054


class _AttrDict(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _nested_attr_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttrDict({key: _nested_attr_dict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_nested_attr_dict(item) for item in value]
    return value


def _load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    try:
        import easydict

        return easydict.EasyDict(cfg)
    except ImportError:
        return _nested_attr_dict(cfg)


def _to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    if isinstance(out.get("task"), list) and out["task"] and isinstance(out["task"][0], (tuple, list)):
        out["task"] = out["task"][0]
    return out


def _build_dataset(cfg, split: str):
    import utils.datasets
    import utils.helpers

    episodes = utils.datasets.resolve_episode_selection(cfg.DATASET, split)
    return utils.datasets.get_dataset(
        cfg.DATASET.NAME,
        split=split,
        pin_memory=cfg.DATASET.PIN_MEMORY,
        delta_action=cfg.DATASET.USE_DELTA_ACTION,
        required_features=cfg.DATASET.REQUIRED_FEATURES,
        episodes=episodes,
        image_transforms=utils.datasets.ImageTransforms(cfg.DATASET.IMG_SIZE),
        delta_timestamps=utils.helpers.get_delta_timestamps(cfg.POLICY, cfg.DATASET.DELTA_TIMESTAMPS),
    )


def _checkpoint_residual_config(checkpoint: str) -> Dict[str, Any]:
    config_path = Path(checkpoint) / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception:
        return {}
    return {
        key.upper(): value
        for key, value in data.items()
        if key == "enable_residual_flow" or key.startswith("residual_")
    }


def _build_policy(
    cfg,
    dataset_meta,
    checkpoint: str,
    residual_enabled: bool,
    device: torch.device,
    label: str = "policy",
):
    import utils.helpers

    tick = time.perf_counter()
    logging.info("Constructing %s policy ...", label)
    policy_cfg = copy.deepcopy(cfg.POLICY)
    if residual_enabled:
        for key, value in _checkpoint_residual_config(checkpoint).items():
            if key == "RESIDUAL_DRIFT_MEAN" or key not in policy_cfg or policy_cfg.get(key) is None:
                policy_cfg[key] = value
    policy_cfg.ENABLE_RESIDUAL_FLOW = bool(residual_enabled)
    policy = utils.helpers.get_policy(
        policy_cfg,
        dataset_meta,
        cfg.DATASET.IMG_SIZE,
        cfg.DATASET.REQUIRED_FEATURES,
    )
    policy.config.device = str(device)
    logging.info("Loading %s checkpoint from %s ...", label, checkpoint)
    policy = policy.from_pretrained(checkpoint, config=policy.config)
    _apply_runtime_residual_config(policy, policy_cfg)
    policy = policy.to(device).eval()
    logging.info("Loaded %s policy in %.1f seconds.", label, time.perf_counter() - tick)
    return policy


def _apply_runtime_residual_config(policy, policy_cfg) -> None:
    """Keep residual ablations from being shadowed by checkpoint config values."""
    for key, value in policy_cfg.items():
        if value is None:
            continue
        if key != "ENABLE_RESIDUAL_FLOW" and not str(key).startswith("RESIDUAL_"):
            continue
        attr_key = str(key).lower()
        for config_owner in (policy, getattr(policy, "model", None)):
            config = getattr(config_owner, "config", None)
            if config is not None and hasattr(config, attr_key):
                setattr(config, attr_key, value)


def _current_state(batch: Dict[str, Any]) -> torch.Tensor:
    state = batch["observation.state"]
    return state[:, -1, :] if state.ndim == 3 else state


def _latest_dynamic(batch: Dict[str, Any], key: str) -> torch.Tensor:
    value = batch[key]
    return value[:, -1, :] if value.ndim == 3 else value


def _absolute_actions(actions: torch.Tensor, state: torch.Tensor, use_delta_action: bool) -> torch.Tensor:
    if not use_delta_action:
        return actions
    out = actions.clone()
    rel_dim = min(out.shape[-1] - 1, state.shape[-1])
    out[..., :rel_dim] = out[..., :rel_dim] + state[:, None, :rel_dim]
    return out


def _slice_batch(batch: Dict[str, Any], mask: torch.Tensor) -> Dict[str, Any]:
    out = {}
    batch_size = int(mask.shape[0])
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def _valid_anchor_mask(batch: Dict[str, Any], eval_steps: int) -> torch.Tensor | None:
    pad = batch.get("action_is_pad", batch.get("actions_id_pad"))
    if pad is None or not isinstance(pad, torch.Tensor):
        return None
    if pad.ndim == 1:
        return ~pad.bool()
    return ~pad[:, :eval_steps].bool().any(dim=1)


def _resolve_eval_action_steps(cfg, explicit_eval_action_steps: int | None, chunk_size: int) -> int:
    if explicit_eval_action_steps is not None and explicit_eval_action_steps > 0:
        requested_steps = explicit_eval_action_steps
    else:
        requested_steps = int(getattr(cfg.POLICY, "N_ACTION_STEPS", chunk_size))
    return max(1, min(int(requested_steps), int(chunk_size)))


def _add_value_sums(out: Dict[str, float], key: str, values: torch.Tensor) -> None:
    values = values.detach()
    out[f"{key}_sum"] = float(values.sum().cpu())
    out[f"{key}_sq_sum"] = float(torch.square(values).sum().cpu())


def _add_axis_sums(out: Dict[str, float], key: str, values_xyz: torch.Tensor) -> None:
    values_xyz = values_xyz.detach()
    names = ("x", "y", "z")
    for axis_idx, axis_name in enumerate(names[: values_xyz.shape[-1]]):
        out[f"{key}_{axis_name}_sum"] = float(values_xyz[:, axis_idx].sum().cpu())


def _add_axis_abs_sums(out: Dict[str, float], key: str, values_xyz: torch.Tensor) -> None:
    _add_axis_sums(out, key, torch.abs(values_xyz))


def _mean_ci95(total: Dict[str, float], key: str, n: float) -> list[float] | None:
    if n <= 0:
        return None
    mean = total.get(f"{key}_sum", 0.0) / n
    if n <= 1:
        return [mean, mean]

    sq_sum = total.get(f"{key}_sq_sum")
    if sq_sum is None:
        return None
    variance = max((sq_sum - n * mean * mean) / (n - 1), 0.0)
    half_width = CI95_Z * float(np.sqrt(variance / n))
    return [mean - half_width, mean + half_width]


def _wilson_ci95(successes: float, n: float) -> list[float] | None:
    if n <= 0:
        return None
    p_hat = successes / n
    denom = 1.0 + CI95_Z * CI95_Z / n
    center = (p_hat + CI95_Z * CI95_Z / (2.0 * n)) / denom
    half_width = (
        CI95_Z
        * float(np.sqrt((p_hat * (1.0 - p_hat) / n) + (CI95_Z * CI95_Z / (4.0 * n * n))))
        / denom
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _batch_metrics(
    pred_actions: torch.Tensor,
    batch: Dict[str, Any],
    *,
    fps: float,
    use_delta_action: bool,
    success_distance_m: float,
    success_distance_sweep_m: list[float],
    eval_action_steps: int,
) -> Dict[str, float]:
    eval_steps = max(1, min(int(eval_action_steps), int(pred_actions.shape[1])))
    valid_mask = _valid_anchor_mask(batch, eval_steps)
    if valid_mask is not None:
        pred_actions = pred_actions[valid_mask]
        batch = _slice_batch(batch, valid_mask)
        if pred_actions.shape[0] == 0:
            return {"n": 0.0, "eval_action_steps_sum": 0.0, "full_chunk_steps_sum": 0.0}

    gt_actions = batch["action"][:, : pred_actions.shape[1], : pred_actions.shape[2]]
    state = _current_state(batch)
    pred_abs = _absolute_actions(pred_actions, state, use_delta_action)
    gt_abs = _absolute_actions(gt_actions, state, use_delta_action)

    eval_pred_actions = pred_actions[:, :eval_steps, :]
    eval_gt_actions = gt_actions[:, :eval_steps, :]
    eval_pred_abs = pred_abs[:, :eval_steps, :]
    eval_gt_abs = gt_abs[:, :eval_steps, :]

    action_mse = torch.mean((eval_pred_actions - eval_gt_actions) ** 2, dim=(1, 2))
    ee_mse = torch.mean((eval_pred_abs[..., :3] - eval_gt_abs[..., :3]) ** 2, dim=(1, 2))
    full_chunk_action_mse = torch.mean((pred_actions - gt_actions) ** 2, dim=(1, 2))
    full_chunk_ee_mse = torch.mean((pred_abs[..., :3] - gt_abs[..., :3]) ** 2, dim=(1, 2))

    obj_pos = _latest_dynamic(batch, "dynamic_object.position_xyz")
    obj_vel = _latest_dynamic(batch, "dynamic_object.velocity_xyz")
    step_times = torch.arange(
        pred_actions.shape[1],
        device=pred_actions.device,
        dtype=pred_actions.dtype,
    ) / max(float(fps), 1.0)
    object_path = obj_pos[:, None, :] + obj_vel[:, None, :] * step_times[None, :, None]
    eval_distances = torch.linalg.norm(
        eval_pred_abs[..., :3] - object_path[:, :eval_steps, :],
        dim=-1,
    )
    min_dist, min_step_idx = torch.min(eval_distances, dim=1)
    min_contact_time = min_step_idx.to(dtype=pred_actions.dtype) / max(float(fps), 1.0)

    executed_final_obj = object_path[:, eval_steps - 1, :]
    executed_final_ee = eval_pred_abs[:, -1, :3]
    executed_final_axis_error = executed_final_ee - executed_final_obj
    executed_final_axis_abs_error = torch.abs(executed_final_axis_error)
    executed_final_dist = torch.linalg.norm(executed_final_axis_error, dim=-1)
    final_gt_axis_error = executed_final_ee - eval_gt_abs[:, -1, :3]
    final_gt_axis_abs_error = torch.abs(final_gt_axis_error)

    legacy_horizon_s = float(max(pred_actions.shape[1] - 1, 0)) / max(float(fps), 1.0)
    legacy_future_obj = obj_pos + obj_vel * legacy_horizon_s
    legacy_final_ee = pred_abs[:, -1, :3]
    legacy_final_axis_error = legacy_final_ee - legacy_future_obj
    legacy_final_axis_abs_error = torch.abs(legacy_final_axis_error)
    legacy_final_dist = torch.linalg.norm(legacy_final_axis_error, dim=-1)
    ee_axis_mse = torch.mean((eval_pred_abs[..., :3] - eval_gt_abs[..., :3]) ** 2, dim=1)

    contact_steps = []
    for sample_idx in range(pred_abs.shape[0]):
        first_contact = float("nan")
        for step_idx in range(eval_steps):
            dist = eval_distances[sample_idx, step_idx]
            if float(dist) <= success_distance_m:
                first_contact = float(step_idx) / max(float(fps), 1.0)
                break
        contact_steps.append(first_contact)

    out = {
        "success_count": float((min_dist <= success_distance_m).sum().detach().cpu()),
        "final_success_count": float((executed_final_dist <= success_distance_m).sum().detach().cpu()),
        "legacy_final_success_count": float((legacy_final_dist <= success_distance_m).sum().detach().cpu()),
        "time_to_contact_sum": float(np.nansum(contact_steps)),
        "time_to_contact_count": float(np.isfinite(contact_steps).sum()),
        "min_distance_time_sum": float(min_contact_time.sum().detach().cpu()),
        "drift_magnitude_sum": float(torch.linalg.norm(obj_vel, dim=-1).sum().detach().cpu()),
        "eval_action_steps_sum": float(eval_steps * pred_actions.shape[0]),
        "full_chunk_steps_sum": float(pred_actions.shape[1] * pred_actions.shape[0]),
        "n": float(pred_actions.shape[0]),
    }
    future_pos = batch.get("dynamic_object.future_position_xyz")
    if isinstance(future_pos, torch.Tensor):
        future_pos = future_pos[:, : pred_actions.shape[1], :3].to(
            device=pred_actions.device,
            dtype=pred_actions.dtype,
        )
        forecast_error = object_path[:, : future_pos.shape[1], :] - future_pos
        forecast_l2 = torch.linalg.norm(forecast_error[:, :eval_steps, :], dim=-1).mean(dim=1)
        executed_forecast_error = object_path[:, eval_steps - 1, :] - future_pos[:, eval_steps - 1, :]
        _add_value_sums(out, "object_forecast_l2", forecast_l2)
        _add_axis_sums(out, "object_forecast_axis_error", executed_forecast_error)
        _add_axis_abs_sums(out, "object_forecast_axis_abs_error", executed_forecast_error)
    _add_value_sums(out, "action_mse", action_mse)
    _add_value_sums(out, "ee_tracking_mse", ee_mse)
    _add_value_sums(out, "full_chunk_action_mse", full_chunk_action_mse)
    _add_value_sums(out, "full_chunk_ee_tracking_mse", full_chunk_ee_mse)
    _add_value_sums(out, "min_object_distance", min_dist)
    _add_value_sums(out, "final_object_distance", executed_final_dist)
    _add_value_sums(out, "legacy_final_object_distance", legacy_final_dist)
    _add_axis_sums(out, "ee_tracking_mse", ee_axis_mse)
    _add_axis_sums(out, "final_gt_axis_abs_error", final_gt_axis_abs_error)
    _add_axis_sums(out, "final_gt_axis_error", final_gt_axis_error)
    _add_axis_sums(out, "final_object_axis_abs_error", executed_final_axis_abs_error)
    _add_axis_sums(out, "final_object_axis_error", executed_final_axis_error)
    _add_axis_sums(out, "legacy_final_object_axis_abs_error", legacy_final_axis_abs_error)
    _add_axis_sums(out, "legacy_final_object_axis_error", legacy_final_axis_error)
    for threshold in success_distance_sweep_m:
        out[f"success_count_at_{threshold:.3f}m"] = float(
            (min_dist <= threshold).sum().detach().cpu()
        )
        out[f"final_success_count_at_{threshold:.3f}m"] = float(
            (executed_final_dist <= threshold).sum().detach().cpu()
        )
        out[f"legacy_final_success_count_at_{threshold:.3f}m"] = float(
            (legacy_final_dist <= threshold).sum().detach().cpu()
        )
    return out


def _merge_metrics(items: Iterable[Dict[str, float]], success_distance_sweep_m: list[float]) -> Dict[str, float]:
    total: Dict[str, float] = {}
    for item in items:
        for key, value in item.items():
            total[key] = total.get(key, 0.0) + float(value)
    actual_n = total.get("n", 0.0)
    if actual_n <= 0:
        return {
            "samples": 0,
            "eval_action_steps": 0,
            "full_chunk_steps": 0,
            "success_rate": 0.0,
            "success_rate_ci95": None,
            "success_rate_by_threshold_m": {f"{threshold:.3f}m": 0.0 for threshold in success_distance_sweep_m},
            "success_rate_by_threshold_ci95_m": {f"{threshold:.3f}m": None for threshold in success_distance_sweep_m},
            "final_success_rate": 0.0,
            "final_success_rate_ci95": None,
            "final_success_rate_by_threshold_m": {f"{threshold:.3f}m": 0.0 for threshold in success_distance_sweep_m},
            "final_success_rate_by_threshold_ci95_m": {f"{threshold:.3f}m": None for threshold in success_distance_sweep_m},
        }
    n = actual_n
    contact_n = total.get("time_to_contact_count", 0.0)
    success_rate_by_threshold = {
        f"{threshold:.3f}m": total.get(f"success_count_at_{threshold:.3f}m", 0.0) / n
        for threshold in success_distance_sweep_m
    }
    success_rate_by_threshold_ci95 = {
        f"{threshold:.3f}m": _wilson_ci95(total.get(f"success_count_at_{threshold:.3f}m", 0.0), n)
        for threshold in success_distance_sweep_m
    }
    final_success_rate_by_threshold = {
        f"{threshold:.3f}m": total.get(f"final_success_count_at_{threshold:.3f}m", 0.0) / n
        for threshold in success_distance_sweep_m
    }
    final_success_rate_by_threshold_ci95 = {
        f"{threshold:.3f}m": _wilson_ci95(
            total.get(f"final_success_count_at_{threshold:.3f}m", 0.0),
            n,
        )
        for threshold in success_distance_sweep_m
    }
    legacy_final_success_rate_by_threshold = {
        f"{threshold:.3f}m": total.get(f"legacy_final_success_count_at_{threshold:.3f}m", 0.0) / n
        for threshold in success_distance_sweep_m
    }
    legacy_final_success_rate_by_threshold_ci95 = {
        f"{threshold:.3f}m": _wilson_ci95(
            total.get(f"legacy_final_success_count_at_{threshold:.3f}m", 0.0),
            n,
        )
        for threshold in success_distance_sweep_m
    }
    success_rate = total.get("success_count", 0.0) / n
    final_success_rate = total.get("final_success_count", 0.0) / n
    legacy_final_success_rate = total.get("legacy_final_success_count", 0.0) / n
    return {
        "samples": int(total.get("n", 0.0)),
        "eval_action_steps": int(round(total.get("eval_action_steps_sum", 0.0) / n)),
        "full_chunk_steps": int(round(total.get("full_chunk_steps_sum", 0.0) / n)),
        "success_rate": success_rate,
        "success_rate_ci95": _wilson_ci95(total.get("success_count", 0.0), n),
        "success_rate_by_threshold_m": success_rate_by_threshold,
        "success_rate_by_threshold_ci95_m": success_rate_by_threshold_ci95,
        "final_success_rate": final_success_rate,
        "final_success_rate_ci95": _wilson_ci95(total.get("final_success_count", 0.0), n),
        "final_success_rate_by_threshold_m": final_success_rate_by_threshold,
        "final_success_rate_by_threshold_ci95_m": final_success_rate_by_threshold_ci95,
        "action_mse": total.get("action_mse_sum", 0.0) / n,
        "action_mse_ci95": _mean_ci95(total, "action_mse", n),
        "ee_tracking_mse": total.get("ee_tracking_mse_sum", 0.0) / n,
        "ee_tracking_mse_ci95": _mean_ci95(total, "ee_tracking_mse", n),
        "ee_tracking_mse_xyz": [
            total.get("ee_tracking_mse_x_sum", 0.0) / n,
            total.get("ee_tracking_mse_y_sum", 0.0) / n,
            total.get("ee_tracking_mse_z_sum", 0.0) / n,
        ],
        "final_gt_axis_abs_error_xyz": [
            total.get("final_gt_axis_abs_error_x_sum", 0.0) / n,
            total.get("final_gt_axis_abs_error_y_sum", 0.0) / n,
            total.get("final_gt_axis_abs_error_z_sum", 0.0) / n,
        ],
        "final_gt_axis_error_xyz": [
            total.get("final_gt_axis_error_x_sum", 0.0) / n,
            total.get("final_gt_axis_error_y_sum", 0.0) / n,
            total.get("final_gt_axis_error_z_sum", 0.0) / n,
        ],
        "full_chunk_action_mse": total.get("full_chunk_action_mse_sum", 0.0) / n,
        "full_chunk_action_mse_ci95": _mean_ci95(total, "full_chunk_action_mse", n),
        "full_chunk_ee_tracking_mse": total.get("full_chunk_ee_tracking_mse_sum", 0.0) / n,
        "full_chunk_ee_tracking_mse_ci95": _mean_ci95(total, "full_chunk_ee_tracking_mse", n),
        "min_object_distance": total.get("min_object_distance_sum", 0.0) / n,
        "min_object_distance_ci95": _mean_ci95(total, "min_object_distance", n),
        "min_distance_time": total.get("min_distance_time_sum", 0.0) / n,
        "final_object_distance": total.get("final_object_distance_sum", 0.0) / n,
        "final_object_distance_ci95": _mean_ci95(total, "final_object_distance", n),
        "final_object_axis_abs_error_xyz": [
            total.get("final_object_axis_abs_error_x_sum", 0.0) / n,
            total.get("final_object_axis_abs_error_y_sum", 0.0) / n,
            total.get("final_object_axis_abs_error_z_sum", 0.0) / n,
        ],
        "final_object_axis_error_xyz": [
            total.get("final_object_axis_error_x_sum", 0.0) / n,
            total.get("final_object_axis_error_y_sum", 0.0) / n,
            total.get("final_object_axis_error_z_sum", 0.0) / n,
        ],
        "legacy_final_success_rate": legacy_final_success_rate,
        "legacy_final_success_rate_ci95": _wilson_ci95(total.get("legacy_final_success_count", 0.0), n),
        "legacy_final_success_rate_by_threshold_m": legacy_final_success_rate_by_threshold,
        "legacy_final_success_rate_by_threshold_ci95_m": legacy_final_success_rate_by_threshold_ci95,
        "legacy_final_object_distance": total.get("legacy_final_object_distance_sum", 0.0) / n,
        "legacy_final_object_distance_ci95": _mean_ci95(total, "legacy_final_object_distance", n),
        "legacy_final_object_axis_abs_error_xyz": [
            total.get("legacy_final_object_axis_abs_error_x_sum", 0.0) / n,
            total.get("legacy_final_object_axis_abs_error_y_sum", 0.0) / n,
            total.get("legacy_final_object_axis_abs_error_z_sum", 0.0) / n,
        ],
        "legacy_final_object_axis_error_xyz": [
            total.get("legacy_final_object_axis_error_x_sum", 0.0) / n,
            total.get("legacy_final_object_axis_error_y_sum", 0.0) / n,
            total.get("legacy_final_object_axis_error_z_sum", 0.0) / n,
        ],
        "time_to_contact": (
            total.get("time_to_contact_sum", 0.0) / contact_n if contact_n > 0 else None
        ),
        "mean_object_velocity_norm": total.get("drift_magnitude_sum", 0.0) / n,
    }
    if "object_forecast_l2_sum" in total:
        out.update(
            {
                "object_forecast_l2": total.get("object_forecast_l2_sum", 0.0) / n,
                "object_forecast_l2_ci95": _mean_ci95(total, "object_forecast_l2", n),
                "object_forecast_axis_abs_error_xyz": [
                    total.get("object_forecast_axis_abs_error_x_sum", 0.0) / n,
                    total.get("object_forecast_axis_abs_error_y_sum", 0.0) / n,
                    total.get("object_forecast_axis_abs_error_z_sum", 0.0) / n,
                ],
                "object_forecast_axis_error_xyz": [
                    total.get("object_forecast_axis_error_x_sum", 0.0) / n,
                    total.get("object_forecast_axis_error_y_sum", 0.0) / n,
                    total.get("object_forecast_axis_error_z_sum", 0.0) / n,
                ],
            }
        )
    return out


def _example_arrays(pred_actions: torch.Tensor, batch: Dict[str, Any], cfg, fps: float) -> Dict[str, np.ndarray]:
    gt_actions = batch["action"][:, : pred_actions.shape[1], : pred_actions.shape[2]]
    state = _current_state(batch)
    pred_abs = _absolute_actions(pred_actions, state, bool(cfg.DATASET.USE_DELTA_ACTION))
    gt_abs = _absolute_actions(gt_actions, state, bool(cfg.DATASET.USE_DELTA_ACTION))
    obj_vel = _latest_dynamic(batch, "dynamic_object.velocity_xyz")
    steps = pred_actions.shape[1]
    t = torch.arange(steps, device=pred_actions.device, dtype=pred_actions.dtype) / max(float(fps), 1.0)
    drift_xyz = obj_vel[:, None, :] * t[None, :, None]
    return {
        "pred_xyz": pred_abs[0, :, :3].detach().cpu().numpy(),
        "gt_xyz": gt_abs[0, :, :3].detach().cpu().numpy(),
        "object_velocity_xyz": obj_vel[0].detach().cpu().numpy(),
        "integrated_drift_xyz": drift_xyz[0].detach().cpu().numpy(),
    }


def _write_example_plots(
    baseline_example: Dict[str, np.ndarray],
    residual_example: Dict[str, np.ndarray],
    output_dir: Path,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "offline_comparison_example.png"
    dims = ["x", "y", "z"]
    steps = np.arange(baseline_example["gt_xyz"].shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), constrained_layout=True)
    for dim_idx, name in enumerate(dims):
        axes[0].plot(steps, baseline_example["gt_xyz"][:, dim_idx], "--", label=f"gt {name}")
        axes[0].plot(steps, baseline_example["pred_xyz"][:, dim_idx], label=f"baseline {name}")
        axes[0].plot(steps, residual_example["pred_xyz"][:, dim_idx], label=f"residual {name}")
    axes[0].set_title("Predicted action chunk XYZ")
    axes[0].set_xlabel("action step")
    axes[0].set_ylabel("absolute EE position")
    axes[0].legend(ncol=3, fontsize=8)

    width = 0.25
    x = np.arange(3)
    axes[1].bar(x - width / 2, baseline_example["object_velocity_xyz"], width, label="baseline batch")
    axes[1].bar(x + width / 2, residual_example["object_velocity_xyz"], width, label="residual batch")
    axes[1].set_xticks(x, dims)
    axes[1].set_title("Object velocity XYZ")
    axes[1].set_ylabel("velocity")
    axes[1].legend(fontsize=8)

    for dim_idx, name in enumerate(dims):
        axes[2].plot(steps, residual_example["integrated_drift_xyz"][:, dim_idx], label=f"drift {name}")
    axes[2].set_title("Integrated residual drift")
    axes[2].set_xlabel("action step")
    axes[2].set_ylabel("object velocity * horizon")
    axes[2].legend(ncol=3, fontsize=8)

    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _sample_noise_for_batch(
    batch: Dict[str, Any],
    *,
    chunk_size: int,
    action_dim: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    sample_index = batch.get("sample_index")
    if not isinstance(sample_index, torch.Tensor):
        sample_index = torch.arange(batch["action"].shape[0])
    noise = []
    for idx in sample_index.detach().cpu().flatten().tolist():
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(idx))
        noise.append(torch.randn(chunk_size, action_dim, generator=generator))
    return torch.stack(noise, dim=0).to(device=device, dtype=torch.float32)


def _group_masks(
    batch: Dict[str, Any],
    *,
    include_episode_groups: bool = True,
    include_velocity_groups: bool = True,
) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    episode_index = batch.get("episode_index")
    if include_episode_groups and isinstance(episode_index, torch.Tensor):
        for episode in torch.unique(episode_index.detach().cpu()).tolist():
            masks[f"episode_{int(episode)}"] = episode_index == int(episode)

    if include_velocity_groups and "dynamic_object.velocity_xyz" in batch:
        vel = _latest_dynamic(batch, "dynamic_object.velocity_xyz")
        speed = torch.linalg.norm(vel, dim=-1)
        masks["velocity_static_lt_0.02"] = speed < 0.02
        masks["velocity_slow_0.02_0.05"] = (speed >= 0.02) & (speed < 0.05)
        masks["velocity_medium_0.05_0.10"] = (speed >= 0.05) & (speed < 0.10)
        masks["velocity_fast_ge_0.10"] = speed >= 0.10
        masks["velocity_train_lt_0.08"] = speed < 0.08
        masks["velocity_gap_0.08_0.18"] = (speed >= 0.08) & (speed < 0.18)
        masks["velocity_strict_ood_ge_0.18"] = speed >= 0.18
        masks["velocity_very_fast_ge_0.25"] = speed >= 0.25
    return {key: mask for key, mask in masks.items() if bool(mask.any())}


def _append_group_metrics(
    grouped: Dict[str, List[Dict[str, float]]],
    pred_actions: torch.Tensor,
    batch: Dict[str, Any],
    *,
    fps: float,
    use_delta_action: bool,
    success_distance_m: float,
    success_distance_sweep_m: list[float],
    eval_action_steps: int,
    include_episode_groups: bool = True,
    include_velocity_groups: bool = True,
) -> None:
    for group_name, mask in _group_masks(
        batch,
        include_episode_groups=include_episode_groups,
        include_velocity_groups=include_velocity_groups,
    ).items():
        group_batch = _slice_batch(batch, mask)
        grouped.setdefault(group_name, []).append(
            _batch_metrics(
                pred_actions[mask],
                group_batch,
                fps=fps,
                use_delta_action=use_delta_action,
                success_distance_m=success_distance_m,
                success_distance_sweep_m=success_distance_sweep_m,
                eval_action_steps=eval_action_steps,
            )
        )


def _merge_grouped_metrics(
    grouped: Dict[str, List[Dict[str, float]]],
    success_distance_sweep_m: list[float],
) -> Dict[str, Dict[str, float]]:
    return {
        group_name: _merge_metrics(items, success_distance_sweep_m)
        for group_name, items in sorted(grouped.items())
    }


def _transform_velocity(velocity: torch.Tensor, transform: str) -> torch.Tensor:
    out = velocity.clone()
    if transform == "identity":
        return out
    if transform == "flip_x":
        out[..., 0] *= -1
        return out
    if transform == "flip_y":
        out[..., 1] *= -1
        return out
    if transform == "flip_xy":
        out[..., :2] *= -1
        return out
    if transform == "swap_xy":
        return out[..., [1, 0, 2]]
    if transform == "swap_xy_flip_x":
        swapped = out[..., [1, 0, 2]]
        swapped[..., 0] *= -1
        return swapped
    if transform == "swap_xy_flip_y":
        swapped = out[..., [1, 0, 2]]
        swapped[..., 1] *= -1
        return swapped
    if transform == "swap_xy_flip_xy":
        swapped = out[..., [1, 0, 2]]
        swapped[..., :2] *= -1
        return swapped
    raise ValueError(f"Unknown velocity transform: {transform}")


def _axis_audit_batch(
    batch: Dict[str, Any],
    *,
    fps: float,
    eval_action_steps: int,
) -> Dict[str, Dict[str, float]]:
    future_pos = batch.get("dynamic_object.future_position_xyz")
    if not isinstance(future_pos, torch.Tensor):
        return {}
    obj_pos = _latest_dynamic(batch, "dynamic_object.position_xyz")
    obj_vel = _latest_dynamic(batch, "dynamic_object.velocity_xyz")
    steps = min(int(eval_action_steps), int(future_pos.shape[1]))
    valid_mask = _valid_anchor_mask(batch, steps)
    if valid_mask is not None:
        obj_pos = obj_pos[valid_mask]
        obj_vel = obj_vel[valid_mask]
        future_pos = future_pos[valid_mask]
    if obj_pos.shape[0] == 0:
        return {}

    step_times = torch.arange(
        steps,
        device=obj_pos.device,
        dtype=obj_pos.dtype,
    ) / max(float(fps), 1.0)
    out: Dict[str, Dict[str, float]] = {}
    transforms = [
        "identity",
        "flip_x",
        "flip_y",
        "flip_xy",
        "swap_xy",
        "swap_xy_flip_x",
        "swap_xy_flip_y",
        "swap_xy_flip_xy",
    ]
    for transform in transforms:
        vel = _transform_velocity(obj_vel, transform)
        forecast = obj_pos[:, None, :] + vel[:, None, :] * step_times[None, :, None]
        error = forecast - future_pos[:, :steps, :3]
        final_error = error[:, -1, :]
        out[transform] = {
            "n": float(obj_pos.shape[0]),
            "l2_sum": float(torch.linalg.norm(error, dim=-1).mean(dim=1).sum().detach().cpu()),
            "final_l2_sum": float(torch.linalg.norm(final_error, dim=-1).sum().detach().cpu()),
            "final_abs_x_sum": float(torch.abs(final_error[:, 0]).sum().detach().cpu()),
            "final_abs_y_sum": float(torch.abs(final_error[:, 1]).sum().detach().cpu()),
            "final_abs_z_sum": float(torch.abs(final_error[:, 2]).sum().detach().cpu()),
        }
    return out


def _merge_axis_audit(items: Iterable[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Any]]:
    total: Dict[str, Dict[str, float]] = {}
    for item in items:
        for transform, metrics in item.items():
            bucket = total.setdefault(transform, {})
            for key, value in metrics.items():
                bucket[key] = bucket.get(key, 0.0) + float(value)
    merged = {}
    for transform, metrics in sorted(total.items()):
        n = metrics.get("n", 0.0)
        if n <= 0:
            continue
        merged[transform] = {
            "samples": int(n),
            "mean_l2": metrics.get("l2_sum", 0.0) / n,
            "final_l2": metrics.get("final_l2_sum", 0.0) / n,
            "final_abs_error_xyz": [
                metrics.get("final_abs_x_sum", 0.0) / n,
                metrics.get("final_abs_y_sum", 0.0) / n,
                metrics.get("final_abs_z_sum", 0.0) / n,
            ],
        }
    if merged:
        best = min(merged.items(), key=lambda item: item[1]["mean_l2"])
        merged["best_transform"] = {"name": best[0], **best[1]}
    return merged


@torch.no_grad()
def evaluate_policy(
    policy,
    loader,
    device,
    cfg,
    max_batches: int,
    success_distance_m: float,
    success_distance_sweep_m: list[float],
    eval_action_steps: int,
    noise_seed: int,
    label: str = "policy",
    progress_every: int = 0,
    include_episode_groups: bool = True,
    include_velocity_groups: bool = True,
) -> Dict[str, float]:
    metrics: List[Dict[str, float]] = []
    grouped: Dict[str, List[Dict[str, float]]] = {}
    total_batches = len(loader)
    if max_batches > 0:
        total_batches = min(total_batches, max_batches)
    logging.info("Evaluating %s on %d batches ...", label, total_batches)
    tick = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        policy.reset()
        batch = _to_device_batch(batch, device)
        noise = _sample_noise_for_batch(
            batch,
            chunk_size=int(cfg.POLICY.CHUNK_SIZE),
            action_dim=int(getattr(policy.config, "max_action_dim", 32)),
            device=device,
            seed=noise_seed,
        )
        pred_actions = policy.predict_action_chunk(batch, noise=noise)
        fps = float(getattr(loader.dataset.meta, "fps", 25))
        item = _batch_metrics(
            pred_actions,
            batch,
            fps=fps,
            use_delta_action=bool(cfg.DATASET.USE_DELTA_ACTION),
            success_distance_m=success_distance_m,
            success_distance_sweep_m=success_distance_sweep_m,
            eval_action_steps=eval_action_steps,
        )
        metrics.append(item)
        _append_group_metrics(
            grouped,
            pred_actions,
            batch,
            fps=fps,
            use_delta_action=bool(cfg.DATASET.USE_DELTA_ACTION),
            success_distance_m=success_distance_m,
            success_distance_sweep_m=success_distance_sweep_m,
            eval_action_steps=eval_action_steps,
            include_episode_groups=include_episode_groups,
            include_velocity_groups=include_velocity_groups,
        )
        seen = batch_idx + 1
        if progress_every > 0 and (seen == 1 or seen % progress_every == 0 or seen == total_batches):
            elapsed = time.perf_counter() - tick
            rate = seen / max(elapsed, 1e-9)
            remaining = max(total_batches - seen, 0) / max(rate, 1e-9)
            logging.info(
                "%s progress: %d/%d batches (%.2f batches/s, ETA %.1f min).",
                label,
                seen,
                total_batches,
                rate,
                remaining / 60.0,
            )
    merged = _merge_metrics(metrics, success_distance_sweep_m)
    merged["groups"] = _merge_grouped_metrics(grouped, success_distance_sweep_m)
    return merged


@torch.no_grad()
def evaluate_gt_oracle_and_axis_audit(
    loader,
    device,
    cfg,
    max_batches: int,
    success_distance_m: float,
    success_distance_sweep_m: list[float],
    eval_action_steps: int,
    progress_every: int = 0,
    include_episode_groups: bool = True,
    include_velocity_groups: bool = True,
) -> tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    metrics: List[Dict[str, float]] = []
    grouped: Dict[str, List[Dict[str, float]]] = {}
    axis_items: List[Dict[str, Dict[str, float]]] = []
    total_batches = len(loader)
    if max_batches > 0:
        total_batches = min(total_batches, max_batches)
    logging.info("Evaluating GT oracle/axis audit on %d batches ...", total_batches)
    tick = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        batch = _to_device_batch(batch, device)
        gt_actions = batch["action"]
        fps = float(getattr(loader.dataset.meta, "fps", 25))
        item = _batch_metrics(
            gt_actions,
            batch,
            fps=fps,
            use_delta_action=bool(cfg.DATASET.USE_DELTA_ACTION),
            success_distance_m=success_distance_m,
            success_distance_sweep_m=success_distance_sweep_m,
            eval_action_steps=eval_action_steps,
        )
        metrics.append(item)
        _append_group_metrics(
            grouped,
            gt_actions,
            batch,
            fps=fps,
            use_delta_action=bool(cfg.DATASET.USE_DELTA_ACTION),
            success_distance_m=success_distance_m,
            success_distance_sweep_m=success_distance_sweep_m,
            eval_action_steps=eval_action_steps,
            include_episode_groups=include_episode_groups,
            include_velocity_groups=include_velocity_groups,
        )
        axis_items.append(
            _axis_audit_batch(
                batch,
                fps=fps,
                eval_action_steps=eval_action_steps,
            )
        )
        seen = batch_idx + 1
        if progress_every > 0 and (seen == 1 or seen % progress_every == 0 or seen == total_batches):
            elapsed = time.perf_counter() - tick
            rate = seen / max(elapsed, 1e-9)
            remaining = max(total_batches - seen, 0) / max(rate, 1e-9)
            logging.info(
                "GT oracle/axis audit progress: %d/%d batches (%.2f batches/s, ETA %.1f min).",
                seen,
                total_batches,
                rate,
                remaining / 60.0,
            )
    merged = _merge_metrics(metrics, success_distance_sweep_m)
    merged["groups"] = _merge_grouped_metrics(grouped, success_distance_sweep_m)
    return merged, _merge_axis_audit(axis_items)


def _relative_reduction(baseline_value: float | None, residual_value: float | None) -> float | None:
    if baseline_value is None or residual_value is None or abs(baseline_value) < 1e-12:
        return None
    return (baseline_value - residual_value) / baseline_value


def _comparison_summary(
    baseline_metrics: Dict[str, float],
    residual_metrics: Dict[str, float],
) -> Dict[str, float | None]:
    return {
        "success_rate_delta": residual_metrics["success_rate"] - baseline_metrics["success_rate"],
        "final_success_rate_delta": (
            residual_metrics["final_success_rate"] - baseline_metrics["final_success_rate"]
        ),
        "legacy_final_success_rate_delta": (
            residual_metrics["legacy_final_success_rate"] - baseline_metrics["legacy_final_success_rate"]
        ),
        "action_mse_delta": residual_metrics["action_mse"] - baseline_metrics["action_mse"],
        "action_mse_relative_reduction": _relative_reduction(
            baseline_metrics["action_mse"], residual_metrics["action_mse"]
        ),
        "ee_tracking_mse_delta": residual_metrics["ee_tracking_mse"] - baseline_metrics["ee_tracking_mse"],
        "ee_tracking_mse_relative_reduction": _relative_reduction(
            baseline_metrics["ee_tracking_mse"], residual_metrics["ee_tracking_mse"]
        ),
        "final_object_distance_delta": (
            residual_metrics["final_object_distance"] - baseline_metrics["final_object_distance"]
        ),
        "final_object_distance_relative_reduction": _relative_reduction(
            baseline_metrics["final_object_distance"],
            residual_metrics["final_object_distance"],
        ),
        "min_object_distance_delta": (
            residual_metrics["min_object_distance"] - baseline_metrics["min_object_distance"]
        ),
        "min_object_distance_relative_reduction": _relative_reduction(
            baseline_metrics["min_object_distance"],
            residual_metrics["min_object_distance"],
        ),
        "legacy_final_object_distance_delta": (
            residual_metrics["legacy_final_object_distance"] - baseline_metrics["legacy_final_object_distance"]
        ),
        "legacy_final_object_distance_relative_reduction": _relative_reduction(
            baseline_metrics["legacy_final_object_distance"],
            residual_metrics["legacy_final_object_distance"],
        ),
    }


@torch.no_grad()
def collect_examples(
    baseline,
    residual,
    loader,
    device,
    cfg,
    noise_seed: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    batch = next(iter(loader))
    batch = _to_device_batch(batch, device)
    fps = float(getattr(loader.dataset.meta, "fps", 25))
    noise = _sample_noise_for_batch(
        batch,
        chunk_size=int(cfg.POLICY.CHUNK_SIZE),
        action_dim=int(getattr(baseline.config, "max_action_dim", 32)),
        device=device,
        seed=noise_seed,
    )
    baseline.reset()
    residual.reset()
    baseline_pred = baseline.predict_action_chunk(batch, noise=noise)
    residual_pred = residual.predict_action_chunk(batch, noise=noise)
    return (
        _example_arrays(baseline_pred, batch, cfg, fps),
        _example_arrays(residual_pred, batch, cfg, fps),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=str(PROJECT_HOME / "configs" / "residual_dynamicvla_dom_tiny.yaml"))
    parser.add_argument("--baseline-checkpoint", default="hzxie/dynamic-vla-DOM")
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--output", default="runs/proactive_vla/offline_compare/report.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--eval-action-steps",
        type=int,
        default=0,
        help="Number of predicted actions to score. <=0 uses POLICY.N_ACTION_STEPS.",
    )
    parser.add_argument("--success-distance-m", type=float, default=0.05)
    parser.add_argument(
        "--success-distance-sweep-m",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.25],
        help="Distance thresholds used for auxiliary offline success rates.",
    )
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N batches for each evaluation phase. <=0 disables progress logs.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = _load_cfg(args.cfg)
    device = torch.device(args.device)

    dataset = _build_dataset(cfg, "test")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.CONST.N_WORKERS,
        pin_memory=cfg.DATASET.PIN_MEMORY,
    )

    baseline = _build_policy(cfg, dataset.meta, args.baseline_checkpoint, False, device, "baseline")
    residual = _build_policy(cfg, dataset.meta, args.residual_checkpoint, True, device, "residual")

    success_distance_sweep_m = sorted(set(args.success_distance_sweep_m + [args.success_distance_m]))
    eval_action_steps = _resolve_eval_action_steps(cfg, args.eval_action_steps, int(cfg.POLICY.CHUNK_SIZE))
    gt_oracle_metrics, axis_audit = evaluate_gt_oracle_and_axis_audit(
        loader,
        device,
        cfg,
        args.max_batches,
        args.success_distance_m,
        success_distance_sweep_m,
        eval_action_steps,
        args.progress_every,
    )
    baseline_metrics = evaluate_policy(
        baseline,
        loader,
        device,
        cfg,
        args.max_batches,
        args.success_distance_m,
        success_distance_sweep_m,
        eval_action_steps,
        args.noise_seed,
        "baseline",
        args.progress_every,
    )
    residual_metrics = evaluate_policy(
        residual,
        loader,
        device,
        cfg,
        args.max_batches,
        args.success_distance_m,
        success_distance_sweep_m,
        eval_action_steps,
        args.noise_seed,
        "residual",
        args.progress_every,
    )
    output = Path(args.output)
    plot_path = None
    try:
        baseline_example, residual_example = collect_examples(
            baseline, residual, loader, device, cfg, args.noise_seed
        )
        plot_path = _write_example_plots(baseline_example, residual_example, output.parent)
    except Exception as exc:
        logging.warning("Could not write example comparison plot: %s", exc)

    report = {
        "evaluation_mode": "offline_proxy",
        "simulator_available": False,
        "baseline_checkpoint": args.baseline_checkpoint,
        "residual_checkpoint": args.residual_checkpoint,
        "baseline_success_rate": baseline_metrics["success_rate"],
        "residual_success_rate": residual_metrics["success_rate"],
        "gt_oracle_success_rate": gt_oracle_metrics["success_rate"],
        "absolute_success_improvement": residual_metrics["success_rate"] - baseline_metrics["success_rate"],
        "baseline_legacy_final_success_rate": baseline_metrics["legacy_final_success_rate"],
        "residual_legacy_final_success_rate": residual_metrics["legacy_final_success_rate"],
        "legacy_final_success_improvement": (
            residual_metrics["legacy_final_success_rate"] - baseline_metrics["legacy_final_success_rate"]
        ),
        "success_distance_m": args.success_distance_m,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "eval_action_steps": eval_action_steps,
        "noise_seed": args.noise_seed,
        "comparison": _comparison_summary(baseline_metrics, residual_metrics),
        "gt_oracle": gt_oracle_metrics,
        "axis_velocity_audit": axis_audit,
        "baseline": baseline_metrics,
        "residual": residual_metrics,
        "plots": {"offline_comparison_example": plot_path},
        "notes": (
            "Offline proxy: primary success is minimum predicted end-effector distance to "
            "a constant-velocity object forecast over the evaluated action horizon. "
            "legacy_* metrics preserve the previous full-chunk final-distance endpoint. "
            "gt_oracle scores the demonstrated action chunk with the same metric; "
            "axis_velocity_audit compares velocity-frame variants against actual future DOM object positions."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
