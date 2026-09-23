# -*- coding: utf-8 -*-
#
# @File:   train.py
# @Author: Haozhe Xie
# @Date:   2025-05-15 20:06:33
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-10-06 16:58:01
# @Email:  root@haozhexie.com

import hashlib
import json
import logging
import math
import os
import shutil
import time

import diffusers.optimization
import torch

import core.test
import utils.average_meter
import utils.datasets
import utils.distributed
import utils.helpers
import utils.summary_writer
from proactive_vla.residual_flow import project_velocity_to_action_drift


_TRAINING_STATE_FILENAME = "training_state.pt"


def _training_state_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, _TRAINING_STATE_FILENAME)


def _load_training_state(
    checkpoint_dir: str,
    optimizer,
    lr_scheduler,
    device: torch.device,
    init_epoch: int,
) -> int:
    state_path = _training_state_path(checkpoint_dir)
    if not os.path.exists(state_path):
        return init_epoch

    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    resumed_epoch = int(state.get("epoch", init_epoch))
    logging.info(
        "Loaded training state from %s; resuming at epoch %d.",
        state_path,
        resumed_epoch + 1,
    )
    return max(init_epoch, resumed_epoch)


def _save_training_state(save_dir: str, optimizer, lr_scheduler, epoch: int) -> None:
    state_path = _training_state_path(save_dir)
    tmp_path = f"{state_path}.tmp"
    torch.save(
        {
            "epoch": int(epoch),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
        },
        tmp_path,
    )
    os.replace(tmp_path, state_path)


def _configured_episodes(dataset_cfg, split: str):
    return utils.datasets.resolve_episode_selection(dataset_cfg, split)


def _eval_test_during_train(cfg) -> bool:
    train_cfg = getattr(cfg, "TRAIN", {})
    if isinstance(train_cfg, dict):
        return bool(train_cfg.get("EVAL_TEST_DURING_TRAIN", True))
    return bool(getattr(train_cfg, "EVAL_TEST_DURING_TRAIN", True))


def _cfg_get(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _train_eval_splits(cfg) -> list[str]:
    train_cfg = getattr(cfg, "TRAIN", {})
    configured = _cfg_get(train_cfg, "EVAL_SPLITS")
    if configured is not None:
        if isinstance(configured, str):
            return [item.strip() for item in configured.replace(",", " ").split() if item.strip()]
        return [str(item) for item in configured if str(item)]
    if _eval_test_during_train(cfg):
        return ["test"]
    return []


def _train_eval_freq_epoch(cfg) -> int:
    return max(0, int(_cfg_get(getattr(cfg, "TRAIN", {}), "EVAL_FREQ_EPOCH", 1) or 0))


def _train_eval_max_batches(cfg) -> int:
    return max(0, int(_cfg_get(getattr(cfg, "TRAIN", {}), "EVAL_MAX_BATCHES", 0) or 0))


def _train_eval_primary_metric(cfg) -> str:
    return str(_cfg_get(getattr(cfg, "TRAIN", {}), "EVAL_PRIMARY_METRIC", "action_mse"))


def _train_eval_primary_split(cfg, eval_splits: list[str]) -> str | None:
    configured = _cfg_get(getattr(cfg, "TRAIN", {}), "EVAL_PRIMARY_SPLIT")
    if configured:
        return str(configured)
    if "ood_val" in eval_splits:
        return "ood_val"
    return eval_splits[0] if eval_splits else None


def _resolve_train_eval_action_steps(cfg, chunk_size: int | None = None) -> int:
    policy_cfg = getattr(cfg, "POLICY", {})
    if chunk_size is None:
        chunk_size = int(_cfg_get(policy_cfg, "CHUNK_SIZE", 1))
    requested_steps = int(_cfg_get(policy_cfg, "N_ACTION_STEPS", chunk_size) or chunk_size)
    return max(1, min(requested_steps, int(chunk_size)))


def _shard_eval_dataset(dataset, rank: int, world_size: int):
    if world_size <= 1:
        return dataset
    indices = list(range(int(rank), len(dataset), int(world_size)))
    subset = torch.utils.data.Subset(dataset, indices)
    for attr_name in ("meta", "episodes"):
        if hasattr(dataset, attr_name):
            setattr(subset, attr_name, getattr(dataset, attr_name))
    return subset


def _to_device_batch(batch: dict, device: torch.device) -> dict:
    out = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    if isinstance(out.get("task"), list) and out["task"] and isinstance(out["task"][0], (tuple, list)):
        out["task"] = out["task"][0]
    return out


def _current_state(batch: dict) -> torch.Tensor:
    state = batch["observation.state"]
    return state[:, -1, :] if state.ndim == 3 else state


def _latest_dynamic(batch: dict, key: str) -> torch.Tensor:
    value = batch[key]
    return value[:, -1, :] if value.ndim == 3 else value


def _absolute_actions(actions: torch.Tensor, state: torch.Tensor, use_delta_action: bool) -> torch.Tensor:
    if not use_delta_action:
        return actions
    out = actions.clone()
    rel_dim = min(out.shape[-1] - 1, state.shape[-1])
    if rel_dim > 0:
        out[..., :rel_dim] = out[..., :rel_dim] + state[:, None, :rel_dim]
    return out


def _slice_batch(batch: dict, mask: torch.Tensor) -> dict:
    out = {}
    batch_size = int(mask.shape[0])
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_size:
            out[key] = value[mask]
        else:
            out[key] = value
    return out


def _valid_anchor_mask(batch: dict, eval_steps: int) -> torch.Tensor | None:
    pad = batch.get("action_is_pad", batch.get("actions_id_pad"))
    if pad is None or not isinstance(pad, torch.Tensor):
        return None
    if pad.ndim == 1:
        return ~pad.bool()
    return ~pad[:, :eval_steps].bool().any(dim=1)


def _loader_fps(loader) -> float:
    dataset = getattr(loader, "dataset", None)
    meta = getattr(dataset, "meta", None)
    if meta is None and hasattr(dataset, "dataset"):
        meta = getattr(dataset.dataset, "meta", None)
    return float(getattr(meta, "fps", 25))


def _sample_eval_noise_for_batch(
    batch: dict,
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


_EVAL_TOTAL_KEYS = (
    "batches",
    "samples",
    "eval_action_steps_sum",
    "action_mse_sum",
    "action_mse_sq_sum",
    "ee_tracking_mse_sum",
    "ee_tracking_mse_sq_sum",
    "final_object_distance_count",
    "final_object_distance_sum",
    "final_object_distance_sq_sum",
    "final_success_count",
)


def _empty_eval_totals() -> dict[str, float]:
    return {key: 0.0 for key in _EVAL_TOTAL_KEYS}


def _add_value_sums(totals: dict[str, float], key: str, values: torch.Tensor) -> None:
    values = values.detach().to(dtype=torch.float64)
    totals[f"{key}_sum"] += float(values.sum().cpu())
    totals[f"{key}_sq_sum"] += float(torch.square(values).sum().cpu())


def _batch_observed_eval_totals(
    pred_actions: torch.Tensor,
    batch: dict,
    *,
    fps: float,
    use_delta_action: bool,
    success_distance_m: float,
    eval_action_steps: int,
) -> dict[str, float]:
    totals = _empty_eval_totals()
    gt_actions = batch["action"].to(device=pred_actions.device, dtype=pred_actions.dtype)
    common_steps = min(int(eval_action_steps), int(pred_actions.shape[1]), int(gt_actions.shape[1]))
    common_dim = min(int(pred_actions.shape[2]), int(gt_actions.shape[2]))
    if common_steps <= 0 or common_dim <= 0:
        return totals

    valid_mask = _valid_anchor_mask(batch, common_steps)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=pred_actions.device)
        pred_actions = pred_actions[valid_mask]
        gt_actions = gt_actions[valid_mask]
        batch = _slice_batch(batch, valid_mask)
        if pred_actions.shape[0] == 0:
            return totals

    pred = pred_actions[:, :common_steps, :common_dim]
    gt = gt_actions[:, :common_steps, :common_dim]
    samples = int(pred.shape[0])
    state = _current_state(batch).to(device=pred.device, dtype=pred.dtype)
    pred_abs = _absolute_actions(pred, state, use_delta_action)
    gt_abs = _absolute_actions(gt, state, use_delta_action)

    action_mse = torch.mean((pred - gt) ** 2, dim=(1, 2))
    ee_dim = min(3, int(pred_abs.shape[-1]), int(gt_abs.shape[-1]))
    if ee_dim > 0:
        ee_mse = torch.mean((pred_abs[..., :ee_dim] - gt_abs[..., :ee_dim]) ** 2, dim=(1, 2))
    else:
        ee_mse = torch.zeros_like(action_mse)

    totals["samples"] += float(samples)
    totals["eval_action_steps_sum"] += float(common_steps * samples)
    _add_value_sums(totals, "action_mse", action_mse)
    _add_value_sums(totals, "ee_tracking_mse", ee_mse)

    if (
        ee_dim >= 3
        and "dynamic_object.position_xyz" in batch
        and "dynamic_object.velocity_xyz" in batch
    ):
        obj_pos = _latest_dynamic(batch, "dynamic_object.position_xyz").to(device=pred.device, dtype=pred.dtype)
        obj_vel = _latest_dynamic(batch, "dynamic_object.velocity_xyz").to(device=pred.device, dtype=pred.dtype)
        final_t = float(common_steps - 1) / max(float(fps), 1.0)
        final_obj = obj_pos[..., :3] + obj_vel[..., :3] * final_t
        final_ee = pred_abs[:, common_steps - 1, :3]
        final_dist = torch.linalg.norm(final_ee - final_obj, dim=-1)
        totals["final_object_distance_count"] += float(samples)
        totals["final_success_count"] += float((final_dist <= success_distance_m).sum().detach().cpu())
        _add_value_sums(totals, "final_object_distance", final_dist)

    return totals


def _merge_eval_totals(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    for key in _EVAL_TOTAL_KEYS:
        left[key] = float(left.get(key, 0.0)) + float(right.get(key, 0.0))
    return left


def _reduce_eval_totals(totals: dict[str, float], device: torch.device, distributed: bool) -> dict[str, float]:
    if distributed and torch.distributed.is_initialized():
        values = torch.tensor([totals.get(key, 0.0) for key in _EVAL_TOTAL_KEYS], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
        return {key: float(value) for key, value in zip(_EVAL_TOTAL_KEYS, values.detach().cpu().tolist())}
    return {key: float(totals.get(key, 0.0)) for key in _EVAL_TOTAL_KEYS}


def _metrics_from_eval_totals(totals: dict[str, float]) -> dict[str, float | int]:
    samples = float(totals.get("samples", 0.0))
    final_count = float(totals.get("final_object_distance_count", 0.0))
    metrics = {
        "samples": int(samples),
        "batches": int(totals.get("batches", 0.0)),
        "eval_action_steps": (totals.get("eval_action_steps_sum", 0.0) / samples if samples > 0 else 0.0),
        "action_mse": (totals.get("action_mse_sum", 0.0) / samples if samples > 0 else math.nan),
        "ee_tracking_mse": (totals.get("ee_tracking_mse_sum", 0.0) / samples if samples > 0 else math.nan),
        "final_object_distance": (
            totals.get("final_object_distance_sum", 0.0) / final_count if final_count > 0 else math.nan
        ),
        "final_success_rate": (totals.get("final_success_count", 0.0) / final_count if final_count > 0 else math.nan),
    }
    return metrics


def _finite_scalar_metrics(prefix: str, metrics: dict) -> dict[str, float]:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            out[f"{prefix}/{key}"] = float(value)
    return out


@torch.no_grad()
def _evaluate_observed_loader(
    policy,
    loader,
    cfg,
    device: torch.device,
    *,
    max_batches: int,
    eval_action_steps: int,
    success_distance_m: float = 0.05,
    distributed: bool = False,
) -> dict[str, float | int]:
    totals = _empty_eval_totals()
    was_training = bool(policy.training)
    policy.eval()
    eval_policy = policy.module if hasattr(policy, "module") else policy
    noise_seed = int(_cfg_get(getattr(cfg, "TRAIN", {}), "EVAL_NOISE_SEED", 1234) or 1234)
    try:
        for batch_idx, batch in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            batch = _to_device_batch(batch, device)
            if hasattr(eval_policy, "reset"):
                eval_policy.reset()
            chunk_size = int(getattr(getattr(eval_policy, "config", None), "chunk_size", _cfg_get(cfg.POLICY, "CHUNK_SIZE", batch["action"].shape[1])))
            action_dim = int(getattr(getattr(eval_policy, "config", None), "max_action_dim", batch["action"].shape[-1]))
            noise = _sample_eval_noise_for_batch(
                batch,
                chunk_size=chunk_size,
                action_dim=action_dim,
                device=device,
                seed=noise_seed,
            )
            try:
                pred_actions = eval_policy.predict_action_chunk(batch, noise=noise)
            except TypeError:
                pred_actions = eval_policy.predict_action_chunk(batch)
            batch_totals = _batch_observed_eval_totals(
                pred_actions,
                batch,
                fps=_loader_fps(loader),
                use_delta_action=bool(cfg.DATASET.USE_DELTA_ACTION),
                success_distance_m=success_distance_m,
                eval_action_steps=eval_action_steps,
            )
            batch_totals["batches"] += 1.0
            _merge_eval_totals(totals, batch_totals)
    finally:
        policy.train(was_training)
    totals = _reduce_eval_totals(totals, device, distributed)
    return _metrics_from_eval_totals(totals)


def _set_residual_drift_mean(policy, drift_mean) -> torch.Tensor:
    drift_mean_tensor = torch.as_tensor(drift_mean, dtype=torch.float32).cpu()
    policy.config.residual_drift_mean = drift_mean_tensor.tolist()
    if hasattr(policy, "model") and hasattr(policy.model, "config"):
        policy.model.config.residual_drift_mean = policy.config.residual_drift_mean
    return drift_mean_tensor


def _residual_drift_mean_cache_dir(policy, cfg) -> str:
    cache_dir = getattr(policy.config, "residual_drift_mean_cache_dir", None)
    if cache_dir:
        return cache_dir
    return os.path.join(cfg.DIR.OUTPUT, "residual_drift_mean_cache")


def _residual_drift_mean_cache_metadata(policy, train_data_loader, cfg) -> dict:
    dataset = getattr(train_data_loader, "dataset", None)
    train_episodes = getattr(dataset, "episodes", None)
    if train_episodes is not None:
        train_episodes = [int(ep_idx) for ep_idx in train_episodes]

    return {
        "version": 1,
        "dataset": {
            "name": cfg.DATASET.NAME,
            "train_episodes": train_episodes,
            "required_features": list(cfg.DATASET.get("REQUIRED_FEATURES", [])),
            "delta_timestamps": cfg.DATASET.get("DELTA_TIMESTAMPS"),
            "use_delta_action": bool(cfg.DATASET.USE_DELTA_ACTION),
        },
        "checkpoint": {
            "loaded": cfg.CONST.get("CKPT"),
            "policy_default": cfg.POLICY.get("CHECKPOINT"),
            "use_checkpoint_normalization": bool(
                getattr(policy.config, "use_checkpoint_normalization", False)
            ),
        },
        "action": {
            "chunk_size": int(policy.config.chunk_size),
            "max_action_dim": int(policy.config.max_action_dim),
        },
        "residual": {
            "drift_scale": float(policy.config.residual_drift_scale),
            "drift_mode": policy.config.residual_drift_mode,
            "action_dt": float(policy.config.residual_action_dt),
            "axis_scale_xyz": list(policy.config.residual_axis_scale_xyz),
            "lookahead_horizon_s": float(policy.config.residual_lookahead_horizon_s),
            "normalize_drift": bool(policy.config.residual_normalize_drift),
            "center_drift": bool(policy.config.residual_center_drift),
            "mean_max_batches": int(
                getattr(policy.config, "residual_drift_mean_max_batches", 0) or 0
            ),
            # d_bar is a mean of Phi(v_hat), before lambda scheduling.
            "lambda_type_excluded": True,
        },
    }


def _residual_drift_mean_cache_path(policy, train_data_loader, cfg) -> tuple[str, dict]:
    metadata = _residual_drift_mean_cache_metadata(policy, train_data_loader, cfg)
    signature = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return os.path.join(_residual_drift_mean_cache_dir(policy, cfg), f"{digest}.json"), metadata


def _load_cached_residual_drift_mean(policy, train_data_loader, cfg) -> bool:
    if not getattr(policy.config, "residual_drift_mean_cache", True):
        return False

    cache_path, _ = _residual_drift_mean_cache_path(policy, train_data_loader, cfg)
    if not os.path.exists(cache_path):
        return False

    try:
        with open(cache_path) as fp:
            payload = json.load(fp)
        drift_mean = torch.as_tensor(payload["residual_drift_mean"], dtype=torch.float32)
    except Exception as exc:
        logging.warning("Could not read residual drift mean cache %s: %s", cache_path, exc)
        return False

    expected_shape = (int(policy.config.chunk_size), int(policy.config.max_action_dim))
    if tuple(drift_mean.shape) != expected_shape:
        logging.warning(
            "Ignoring residual drift mean cache %s with shape %s; expected %s.",
            cache_path,
            tuple(drift_mean.shape),
            expected_shape,
        )
        return False

    drift_mean = _set_residual_drift_mean(policy, drift_mean)
    logging.info(
        "Loaded cached residual drift mean from %s; samples=%s; mean L2=%.6f.",
        cache_path,
        payload.get("sample_count", "unknown"),
        float(drift_mean.norm(dim=-1).mean().item()),
    )
    return True


def _save_cached_residual_drift_mean(
    policy,
    train_data_loader,
    cfg,
    drift_mean: torch.Tensor,
    sample_count: float,
) -> None:
    if not getattr(policy.config, "residual_drift_mean_cache", True):
        return
    if not utils.distributed.is_master():
        return

    cache_path, metadata = _residual_drift_mean_cache_path(policy, train_data_loader, cfg)
    payload = {
        "metadata": metadata,
        "sample_count": float(sample_count),
        "mean_l2": float(drift_mean.norm(dim=-1).mean().item()),
        "residual_drift_mean": drift_mean.cpu().tolist(),
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f"{cache_path}.tmp"
    with open(tmp_path, "w") as fp:
        json.dump(payload, fp, indent=2)
    os.replace(tmp_path, cache_path)
    logging.info("Saved residual drift mean cache to %s.", cache_path)


@torch.no_grad()
def _estimate_residual_drift_mean(policy, train_data_loader, cfg, distributed: bool) -> None:
    """Estimate global train-split ``d_bar`` in normalized action coordinates."""
    if not getattr(policy.config, "enable_residual_flow", False):
        return
    if not getattr(policy.config, "residual_center_drift", True):
        return
    if not getattr(policy.config, "residual_estimate_drift_mean", True):
        return
    if getattr(policy.config, "residual_drift_mean", None) is not None:
        logging.info("Using configured residual drift mean; skipping estimation.")
        return
    if _load_cached_residual_drift_mean(policy, train_data_loader, cfg):
        return

    device = torch.device("cpu")
    dtype = torch.float32
    max_batches = int(getattr(policy.config, "residual_drift_mean_max_batches", 0) or 0)
    total = None
    count = torch.zeros((), device=device, dtype=dtype)

    for batch_idx, batch in enumerate(train_data_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        residual_fields = policy.extract_residual_batch_fields(batch)
        batch.update(residual_fields)
        batch_size = int(batch["action"].shape[0])
        action_shape = (batch_size, policy.config.chunk_size, policy.config.max_action_dim)
        residual_position = policy.prepare_residual_xyz(
            batch, policy.config.residual_position_key, device, dtype
        )
        residual_velocity = policy.prepare_residual_velocity(batch, device, dtype)
        if residual_velocity is None:
            continue
        residual_lookahead_position = policy.prepare_residual_xyz(
            batch,
            policy.config.residual_lookahead_position_key,
            device,
            dtype,
        )
        residual_lookahead_horizon = policy.prepare_residual_horizon(
            batch,
            policy.config.residual_lookahead_horizon_key,
            device,
            dtype,
        )
        residual_action_std = policy.get_action_normalization_std(device, dtype)
        drift = project_velocity_to_action_drift(
            residual_velocity,
            action_shape,
            position_xyz=residual_position,
            lookahead_position_xyz=residual_lookahead_position,
            lookahead_horizon_s=residual_lookahead_horizon,
            action_std=residual_action_std,
            drift_scale=policy.config.residual_drift_scale,
            drift_mode=policy.config.residual_drift_mode,
            action_dt=policy.config.residual_action_dt,
            lookahead_horizon_default_s=policy.config.residual_lookahead_horizon_s,
            axis_scale_xyz=policy.config.residual_axis_scale_xyz,
            device=device,
            dtype=dtype,
        )
        if drift is None:
            continue
        batch_sum = drift.sum(dim=0)
        total = batch_sum if total is None else total + batch_sum
        count += batch_size

    if total is None or float(count.item()) <= 0:
        logging.warning("Could not estimate residual drift mean; using zero-centered drift.")
        return

    if distributed and torch.distributed.is_initialized():
        reduce_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        total = total.to(reduce_device)
        count = count.to(total.device)
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        total = total.cpu()
        count = count.cpu()

    drift_mean = _set_residual_drift_mean(policy, total / torch.clamp(count, min=1.0))
    logging.info(
        "Estimated residual drift mean from %.0f samples; mean L2=%.6f.",
        float(count.item()),
        float(drift_mean.norm(dim=-1).mean().item()),
    )
    _save_cached_residual_drift_mean(
        policy,
        train_data_loader,
        cfg,
        drift_mean,
        float(count.item()),
    )


def train(cfg):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    rank, local_rank = utils.distributed.get_rank(), utils.distributed.get_local_rank()
    distributed = utils.distributed.is_initialized()
    device = torch.device("cuda:%d" % local_rank if torch.cuda.is_available() else "cpu")

    # Set up datasets
    eval_splits = _train_eval_splits(cfg)
    eval_freq_epoch = _train_eval_freq_epoch(cfg)
    eval_max_batches = _train_eval_max_batches(cfg)
    eval_primary_metric = _train_eval_primary_metric(cfg)
    eval_primary_split = _train_eval_primary_split(cfg, eval_splits)
    train_episodes = _configured_episodes(cfg.DATASET, "train")
    eval_episode_map = {
        split: _configured_episodes(cfg.DATASET, split) for split in eval_splits
    }
    episodes_to_prefetch = set(train_episodes or [])
    for episodes in eval_episode_map.values():
        episodes_to_prefetch.update(episodes or [])
    if episodes_to_prefetch:
        episodes_to_prefetch = sorted(episodes_to_prefetch)
        if utils.distributed.is_local_master():
            logging.info(
                "Prefetching %d configured DOM/LeRobot episodes before dataset construction ...",
                len(episodes_to_prefetch),
            )
        utils.datasets.ensure_lerobot_episodes_cached(
            cfg.DATASET.NAME,
            episodes_to_prefetch,
            cfg.DATASET.REQUIRED_FEATURES,
        )

    if utils.distributed.is_local_master():
        logging.info("Building train dataset ...")
    train_dataset = utils.datasets.get_dataset(
        cfg.DATASET.NAME,
        split="train",
        pin_memory=cfg.DATASET.PIN_MEMORY,
        delta_action=cfg.DATASET.USE_DELTA_ACTION,
        required_features=cfg.DATASET.REQUIRED_FEATURES,
        episodes=train_episodes,
        image_transforms=utils.datasets.ImageTransforms(
            cfg.DATASET.IMG_SIZE, cfg.TRAIN.IMAGE_TRANSFORMS
        ),
        delta_timestamps=utils.helpers.get_delta_timestamps(
            cfg.POLICY, cfg.DATASET.DELTA_TIMESTAMPS
        ),
    )

    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, rank=rank, shuffle=True, drop_last=True
        )

    dataloader_kwargs = {
        "batch_size": cfg.TRAIN.BATCH_SIZE,
        "num_workers": cfg.CONST.N_WORKERS,
        "pin_memory": cfg.DATASET.PIN_MEMORY,
    }
    if cfg.CONST.N_WORKERS > 0:
        dataloader_kwargs["prefetch_factor"] = getattr(
            cfg.CONST, "DATALOADER_PREFETCH_FACTOR", 2
        )
        dataloader_kwargs["persistent_workers"] = getattr(
            cfg.CONST, "PERSISTENT_WORKERS", True
        )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        **dataloader_kwargs,
    )

    world_size = torch.distributed.get_world_size() if distributed else 1
    eval_data_loaders = {}
    if eval_splits:
        for eval_split in eval_splits:
            if utils.distributed.is_local_master():
                logging.info("Building %s eval dataset ...", eval_split)
            eval_dataset = utils.datasets.get_dataset(
                cfg.DATASET.NAME,
                split=eval_split,
                pin_memory=cfg.DATASET.PIN_MEMORY,
                delta_action=cfg.DATASET.USE_DELTA_ACTION,
                required_features=cfg.DATASET.REQUIRED_FEATURES,
                episodes=eval_episode_map[eval_split],
                image_transforms=utils.datasets.ImageTransforms(cfg.DATASET.IMG_SIZE),
                delta_timestamps=utils.helpers.get_delta_timestamps(
                    cfg.POLICY, cfg.DATASET.DELTA_TIMESTAMPS
                ),
            )
            eval_dataset_for_rank = _shard_eval_dataset(eval_dataset, rank, world_size)
            eval_data_loaders[eval_split] = torch.utils.data.DataLoader(
                dataset=eval_dataset_for_rank,
                shuffle=False,
                **dataloader_kwargs,
            )
    elif utils.distributed.is_local_master():
        logging.info("No training-time eval splits configured.")

    # Set up the policy
    if utils.distributed.is_local_master():
        logging.info("Constructing policy ...")
    policy = utils.helpers.get_policy(
        cfg.POLICY,
        train_dataset.meta,
        cfg.DATASET.IMG_SIZE,
        cfg.DATASET.REQUIRED_FEATURES,
    )
    if utils.distributed.is_local_master():
        logging.info(
            "Using policy: %s with config %s" % (cfg.POLICY.TYPE, policy.config)
        )
        logging.info(
            "#Parameters: %s/%s"
            % (
                utils.helpers.get_formatted_big_number(
                    utils.helpers.get_n_parameters(policy, trainable_only=True)
                ),
                utils.helpers.get_formatted_big_number(
                    utils.helpers.get_n_parameters(policy, trainable_only=False)
                ),
            )
        )

    init_epoch = 0
    checkpoint = cfg.CONST.get("CKPT") if "CKPT" in cfg.CONST else None
    init_only = bool(getattr(cfg.CONST, "INIT_ONLY", False))
    if checkpoint is not None:
        checkpoint = str(checkpoint)
        local_checkpoint = os.path.exists(checkpoint)
        likely_remote_checkpoint = (
            not local_checkpoint
            and not checkpoint.startswith(("/", ".", "runs/"))
            and len(checkpoint.split("/")) == 2
        )
        if not local_checkpoint and not likely_remote_checkpoint:
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

        if init_only:
            logging.info("Initializing model weights from %s and starting at epoch 0 ...", checkpoint)
        else:
            logging.info("Loading model checkpoint weights from %s ...", checkpoint)

        if (
            not init_only
            and local_checkpoint
            and os.path.exists(os.path.join(checkpoint, "config.json"))
        ):
            with open(os.path.join(checkpoint, "config.json")) as fp:
                model_cfg = json.load(fp)
                init_epoch = int(model_cfg.get("epoch", 0))
        use_checkpoint_normalization = bool(
            getattr(policy.config, "use_checkpoint_normalization", False)
        )
        normalizers = None
        if not use_checkpoint_normalization:
            # Save the normalizers to enable migration to new datasets.
            normalizers = {
                n: getattr(policy, n)
                for n in [
                    "normalize_inputs",
                    "normalize_targets",
                    "unnormalize_outputs",
                ]
            }
        policy.config.device = str(device)
        policy = policy.from_pretrained(checkpoint, config=policy.config)
        if normalizers is not None:
            for k, v in normalizers.items():
                setattr(policy, k, v)
        logging.info("Model weights loaded from %s.", checkpoint)

    _estimate_residual_drift_mean(policy, train_data_loader, cfg, distributed)

    if torch.cuda.is_available():
        if distributed:
            policy = torch.nn.parallel.DistributedDataParallel(
                policy.to(device),
                device_ids=[local_rank],
                find_unused_parameters=True,
            )
        else:
            policy = policy.to(device)
            policy.device = str(device)

    # Set up the optimizer
    n_batches = len(train_data_loader)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, policy.parameters()),
        lr=cfg.TRAIN.OPTIMIZER.LR,
        eps=cfg.TRAIN.OPTIMIZER.EPS,
        weight_decay=cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY,
        betas=cfg.TRAIN.OPTIMIZER.BETAS,
    )
    lr_scheduler = diffusers.optimization.get_scheduler(
        name=cfg.TRAIN.LR_SCHEDULER.NAME,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.LR_SCHEDULER.N_WARMUP_STEPS,
        num_training_steps=cfg.TRAIN.N_EPOCHS * n_batches,
    )
    if checkpoint is not None and not init_only and os.path.isdir(checkpoint):
        init_epoch = _load_training_state(
            checkpoint,
            optimizer,
            lr_scheduler,
            device,
            init_epoch,
        )

    # Set up folders for logs, snapshot and checkpoints
    if utils.distributed.is_master():
        output_dir = os.path.join(cfg.DIR.OUTPUT, "%s", cfg.CONST.EXP_NAME)
        cfg.DIR.CHECKPOINTS = output_dir % "checkpoints"
        cfg.DIR.LOGS = output_dir % "logs"
        os.makedirs(cfg.DIR.CHECKPOINTS, exist_ok=True)
        # Summary writer
        tb_writer = utils.summary_writer.SummaryWriter(cfg)
        # Log current config
        tb_writer.add_config(cfg.DATASET)
        tb_writer.add_config(cfg.POLICY)
        tb_writer.add_config(cfg.TRAIN)

    if utils.distributed.is_local_master():
        eval_desc = (
            ", ".join(f"{split}={len(loader)} batches/rank" for split, loader in eval_data_loaders.items())
            if eval_data_loaders
            else "none"
        )
        logging.info(
            "Starting training for %d epochs with %d train batches, batch size %d, grad accumulation %d, eval splits: %s.",
            cfg.TRAIN.N_EPOCHS,
            len(train_data_loader),
            cfg.TRAIN.BATCH_SIZE,
            cfg.TRAIN.GRAD_ACCUM_STEPS,
            eval_desc,
        )

    grad_accum_steps = int(cfg.TRAIN.GRAD_ACCUM_STEPS)
    best_eval_metric = math.inf
    best_eval_checkpoint_name = (
        "model.best_ood_action_mse.safetensors"
        if eval_primary_split == "ood_val" and eval_primary_metric == "action_mse"
        else f"model.best_{eval_primary_split}_{eval_primary_metric}.safetensors"
        if eval_primary_split is not None
        else "model.best_eval.safetensors"
    )
    optimizer.zero_grad(set_to_none=True)

    for epoch_idx in range(init_epoch, cfg.TRAIN.N_EPOCHS):
        epoch_start_time = time.perf_counter()
        batch_time = utils.average_meter.AverageMeter()
        data_time = utils.average_meter.AverageMeter()
        train_losses = utils.average_meter.AverageMeter()
        # Randomize the DistributedSampler
        if train_sampler:
            train_sampler.set_epoch(epoch_idx)

        # Training loop
        policy.train()
        batch_end_time = time.perf_counter()
        for batch_idx, batch in enumerate(train_data_loader):
            n_itr = epoch_idx * n_batches + batch_idx
            data_time.update(time.perf_counter() - batch_end_time)
            batch = {
                k: (
                    v.to(device, non_blocking=True)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in batch.items()
            }
            # Fix: Remove the additional dimension for task
            if isinstance(batch["task"], list) and isinstance(
                batch["task"][0], (tuple, list)
            ):
                batch["task"] = batch["task"][0]

            loss, _ = policy.forward(batch)
            loss = loss / grad_accum_steps
            loss.backward()
            should_step = (
                (batch_idx + 1) % grad_accum_steps == 0
                or (batch_idx + 1) == n_batches
            )
            if should_step:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            train_losses.update(loss.item())
            batch_time.update(time.perf_counter() - batch_end_time)
            batch_end_time = time.perf_counter()
            if utils.distributed.is_master():
                tb_writer.add_scalars({"Loss/Batch": train_losses.val()}, n_itr)
                # Save the model checkpoint every few batches
                if (
                    cfg.TRAIN.CKPT_SAVE_FREQ.BATCH != 0
                    and batch_idx % cfg.TRAIN.CKPT_SAVE_FREQ.BATCH == 0
                ):
                    logging.info("Saving checkpoint to %s ..." % cfg.DIR.CHECKPOINTS)
                    utils.helpers.save_checkpoint(
                        cfg, policy, cfg.DIR.CHECKPOINTS, epoch_idx + 1
                    )

            if utils.distributed.is_local_master():
                logging.info(
                    "[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss = %.4f"
                    % (
                        epoch_idx + 1,
                        cfg.TRAIN.N_EPOCHS,
                        batch_idx + 1,
                        n_batches,
                        batch_time.val(),
                        data_time.val(),
                        train_losses.val(),
                    )
                )

        epoch_end_time = time.perf_counter()
        if utils.distributed.is_master():
            tb_writer.add_scalars({"Loss/Epoch/Train": train_losses.avg()}, epoch_idx)

        if utils.distributed.is_local_master():
            logging.info(
                "[Epoch %d/%d] EpochTime = %.3f (s) Losses = %.4f"
                % (
                    epoch_idx + 1,
                    cfg.TRAIN.N_EPOCHS,
                    epoch_end_time - epoch_start_time,
                    train_losses.avg(),
                )
            )

        eval_results = {}
        best_eval_improved = False
        if (
            eval_data_loaders
            and eval_freq_epoch > 0
            and (epoch_idx + 1) % eval_freq_epoch == 0
        ):
            eval_action_steps = _resolve_train_eval_action_steps(cfg)
            for eval_split, eval_loader in eval_data_loaders.items():
                if utils.distributed.is_local_master():
                    logging.info(
                        "Evaluating %s with no gradients (%d max batches, %d action steps) ...",
                        eval_split,
                        eval_max_batches,
                        eval_action_steps,
                    )
                metrics = _evaluate_observed_loader(
                    policy,
                    eval_loader,
                    cfg,
                    device,
                    max_batches=eval_max_batches,
                    eval_action_steps=eval_action_steps,
                    distributed=distributed,
                )
                eval_results[eval_split] = metrics
                if utils.distributed.is_master():
                    tb_writer.add_scalars(
                        _finite_scalar_metrics(f"Eval/{eval_split}", metrics),
                        epoch_idx,
                    )
                if utils.distributed.is_local_master():
                    logging.info(
                        "Eval[%s] samples=%s action_mse=%.6g ee_tracking_mse=%.6g final_object_distance=%.6g final_success_rate=%.4f",
                        eval_split,
                        metrics.get("samples", 0),
                        float(metrics.get("action_mse", math.nan)),
                        float(metrics.get("ee_tracking_mse", math.nan)),
                        float(metrics.get("final_object_distance", math.nan)),
                        float(metrics.get("final_success_rate", math.nan)),
                    )

            primary_metrics = eval_results.get(eval_primary_split)
            if primary_metrics is None:
                if utils.distributed.is_local_master():
                    logging.warning(
                        "TRAIN.EVAL_PRIMARY_SPLIT=%r was not evaluated; skipping best checkpoint selection.",
                        eval_primary_split,
                    )
            else:
                primary_value = primary_metrics.get(eval_primary_metric)
                if isinstance(primary_value, (int, float)) and math.isfinite(float(primary_value)):
                    primary_value = float(primary_value)
                    if primary_value < best_eval_metric:
                        best_eval_metric = primary_value
                        best_eval_improved = True
                        if utils.distributed.is_local_master():
                            logging.info(
                                "New best %s/%s = %.6g at epoch %d.",
                                eval_primary_split,
                                eval_primary_metric,
                                best_eval_metric,
                                epoch_idx + 1,
                            )
                elif utils.distributed.is_local_master():
                    logging.warning(
                        "Eval primary metric %s/%s is not finite (%r); skipping best checkpoint selection.",
                        eval_primary_split,
                        eval_primary_metric,
                        primary_value,
                    )
            optimizer.zero_grad(set_to_none=True)

        # Save the model checkpoint
        if utils.distributed.is_master():
            logging.info("Saving checkpoint to %s ..." % cfg.DIR.CHECKPOINTS)
            utils.helpers.save_checkpoint(
                cfg, policy, cfg.DIR.CHECKPOINTS, epoch_idx + 1
            )
            _save_training_state(
                cfg.DIR.CHECKPOINTS,
                optimizer,
                lr_scheduler,
                epoch_idx + 1,
            )
            if best_eval_improved:
                shutil.copy(
                    os.path.join(cfg.DIR.CHECKPOINTS, "model.safetensors"),
                    os.path.join(cfg.DIR.CHECKPOINTS, best_eval_checkpoint_name),
                )
                logging.info(
                    "Saved best eval checkpoint %s (%s/%s=%.6g).",
                    best_eval_checkpoint_name,
                    eval_primary_split,
                    eval_primary_metric,
                    best_eval_metric,
                )
            if epoch_idx % cfg.TRAIN.CKPT_SAVE_FREQ.EPOCH == 0:
                shutil.copy(
                    os.path.join(cfg.DIR.CHECKPOINTS, "model.safetensors"),
                    os.path.join(
                        cfg.DIR.CHECKPOINTS, "model.epoch%04d.safetensors" % epoch_idx
                    ),
                )

        if distributed:
            torch.distributed.barrier()
