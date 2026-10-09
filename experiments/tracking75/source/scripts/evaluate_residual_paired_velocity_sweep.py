#!/usr/bin/env python
"""Paired counterfactual velocity sweep for residual DynamicVLA.

This benchmark is meant to mirror ``residual_flow_2d.py`` in the full VLA stack:
keep the same DOM observations/actions, change only the velocity condition, and
ask whether the residual policy degrades more gracefully than the baseline under
counterfactual object motion.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

PROJECT_HOME = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_HOME) not in sys.path:
    sys.path.insert(0, str(PROJECT_HOME))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import proactive_residual_offline_compare as offline  # noqa: E402


DEFAULT_CFG = "configs/residual_dynamicvla_dom_paired_velocity.yaml"
DEFAULT_BASELINE_CKPT = "runs/checkpoints/dynamicvla_dom_paired_velocity"
DEFAULT_RESIDUAL_CKPT = "runs/checkpoints/residual_dynamicvla_dom_paired_velocity"
DEFAULT_OUTPUT = "runs/proactive_vla/paired_velocity_sweep/report.json"


def _num_workers(cfg, override: int | None) -> int:
    if override is not None:
        return max(0, int(override))
    return max(0, int(getattr(cfg.CONST, "N_WORKERS", 0)))


def _resolve_episodes(cfg, split: str, args) -> list[int] | None:
    import utils.datasets

    episodes = utils.datasets.resolve_episode_selection(cfg.DATASET, split)
    if episodes is not None:
        episodes = [int(ep_idx) for ep_idx in episodes]

    if args.cached_only:
        episodes, _ = utils.datasets.filter_cached_lerobot_episodes(
            cfg.DATASET.NAME,
            episodes,
            cfg.DATASET.REQUIRED_FEATURES,
            root=getattr(cfg.DATASET, "ROOT", None),
        )
        if not episodes:
            raise RuntimeError(f"No cached episodes found for split={split}.")

    if args.shuffle_episodes and episodes is not None:
        rng = random.Random(int(args.episode_seed))
        episodes = list(episodes)
        rng.shuffle(episodes)

    if args.max_episodes > 0 and episodes is not None:
        episodes = episodes[: int(args.max_episodes)]
    return episodes


def _make_loader(cfg, split: str, args):
    import utils.datasets
    import utils.helpers

    episodes = _resolve_episodes(cfg, split, args)
    dataset = utils.datasets.get_dataset(
        cfg.DATASET.NAME,
        split=split,
        pin_memory=cfg.DATASET.PIN_MEMORY,
        delta_action=cfg.DATASET.USE_DELTA_ACTION,
        required_features=cfg.DATASET.REQUIRED_FEATURES,
        episodes=episodes,
        image_transforms=utils.datasets.ImageTransforms(cfg.DATASET.IMG_SIZE),
        delta_timestamps=utils.helpers.get_delta_timestamps(
            cfg.POLICY,
            cfg.DATASET.DELTA_TIMESTAMPS,
        ),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=_num_workers(cfg, args.num_workers),
        pin_memory=bool(cfg.DATASET.PIN_MEMORY),
    )
    return dataset, loader, episodes


def _clone_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone()
        elif isinstance(value, list):
            out[key] = list(value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _latest_dynamic(batch: Dict[str, Any], key: str) -> torch.Tensor:
    value = batch[key]
    return value[:, -1, :] if value.ndim == 3 else value


def _set_dynamic(batch: Dict[str, Any], key: str, latest: torch.Tensor) -> None:
    current = batch.get(key)
    if isinstance(current, torch.Tensor) and current.ndim == 3:
        updated = current.clone()
        updated[:, -1, : latest.shape[-1]] = latest
        batch[key] = updated
    else:
        batch[key] = latest


def _unit_or_default(velocity: torch.Tensor, fallback_axis: int) -> torch.Tensor:
    norm = torch.linalg.norm(velocity, dim=-1, keepdim=True)
    unit = velocity / torch.clamp(norm, min=1e-6)
    fallback = torch.zeros_like(unit)
    fallback[..., min(fallback_axis, fallback.shape[-1] - 1)] = 1.0
    return torch.where(norm > 1e-6, unit, fallback)


def _counterfactual_batch(
    batch: Dict[str, Any],
    *,
    speed: float,
    direction: str,
    fallback_axis: int,
    fps: float,
) -> Dict[str, Any]:
    out = _clone_batch(batch)
    obj_pos = _latest_dynamic(out, "dynamic_object.position_xyz")
    obj_vel = _latest_dynamic(out, "dynamic_object.velocity_xyz")
    if direction == "observed":
        unit = _unit_or_default(obj_vel, fallback_axis)
    elif direction == "x":
        unit = torch.zeros_like(obj_vel)
        unit[..., 0] = 1.0
    elif direction == "y":
        unit = torch.zeros_like(obj_vel)
        unit[..., 1] = 1.0
    elif direction == "neg_observed":
        unit = -_unit_or_default(obj_vel, fallback_axis)
    else:
        raise ValueError(f"Unknown counterfactual direction: {direction!r}")

    new_vel = unit * float(speed)
    _set_dynamic(out, "dynamic_object.velocity_xyz", new_vel)

    horizon_s = None
    if "dynamic_object.lookahead_horizon_s" in out:
        horizon_s = _latest_dynamic(out, "dynamic_object.lookahead_horizon_s")
        if horizon_s.ndim > 1:
            horizon_s = horizon_s.reshape(horizon_s.shape[0], -1)[:, -1]
    if horizon_s is None:
        horizon_s = torch.ones(new_vel.shape[0], device=new_vel.device, dtype=new_vel.dtype)
    lookahead = obj_pos + new_vel * horizon_s[:, None]
    _set_dynamic(out, "dynamic_object.lookahead_position_xyz", lookahead)

    future_pos = out.get("dynamic_object.future_position_xyz")
    if isinstance(future_pos, torch.Tensor):
        steps = future_pos.shape[1] if future_pos.ndim == 3 else int(getattr(future_pos, "shape", [0])[0])
        t = torch.arange(
            steps,
            device=new_vel.device,
            dtype=new_vel.dtype,
        ) / max(float(fps), 1.0)
        out["dynamic_object.future_position_xyz"] = obj_pos[:, None, :] + new_vel[:, None, :] * t[None, :, None]
    return out


def _ablation_batch(batch: Dict[str, Any], ablation: str) -> Dict[str, Any]:
    out = _clone_batch(batch)
    vel = _latest_dynamic(out, "dynamic_object.velocity_xyz")
    if ablation == "correct":
        return out
    if ablation == "zero":
        new_vel = torch.zeros_like(vel)
    elif ablation == "wrong_sign":
        new_vel = -vel
    elif ablation == "shuffle":
        if vel.shape[0] <= 1:
            new_vel = vel
        else:
            new_vel = vel[torch.roll(torch.arange(vel.shape[0], device=vel.device), shifts=1)]
    else:
        raise ValueError(f"Unknown ablation: {ablation!r}")
    _set_dynamic(out, "dynamic_object.velocity_xyz", new_vel)
    return out


def _final_distance_metrics(
    pred_actions: torch.Tensor,
    batch: Dict[str, Any],
    *,
    fps: float,
    use_delta_action: bool,
    eval_action_steps: int,
) -> Dict[str, float]:
    item = offline._batch_metrics(
        pred_actions,
        batch,
        fps=fps,
        use_delta_action=use_delta_action,
        success_distance_m=0.05,
        success_distance_sweep_m=[0.05, 0.1, 0.15, 0.2, 0.25],
        eval_action_steps=eval_action_steps,
    )
    return {
        "n": item.get("n", 0.0),
        "final_object_distance_sum": item.get("final_object_distance_sum", 0.0),
        "min_object_distance_sum": item.get("min_object_distance_sum", 0.0),
        "action_mse_sum": item.get("action_mse_sum", 0.0),
        "ee_tracking_mse_sum": item.get("ee_tracking_mse_sum", 0.0),
        "final_success_count": item.get("final_success_count", 0.0),
    }


def _merge_items(items: Iterable[Dict[str, float]]) -> Dict[str, float]:
    total: Dict[str, float] = {}
    for item in items:
        for key, value in item.items():
            total[key] = total.get(key, 0.0) + float(value)
    n = total.get("n", 0.0)
    if n <= 0:
        return {"samples": 0}
    return {
        "samples": int(n),
        "final_object_distance": total.get("final_object_distance_sum", 0.0) / n,
        "min_object_distance": total.get("min_object_distance_sum", 0.0) / n,
        "action_mse": total.get("action_mse_sum", 0.0) / n,
        "ee_tracking_mse": total.get("ee_tracking_mse_sum", 0.0) / n,
        "final_success_rate": total.get("final_success_count", 0.0) / n,
    }


@torch.no_grad()
def _evaluate_policy_on_batch(policy, batch, device, cfg, noise_seed: int):
    policy.reset()
    batch = offline._to_device_batch(batch, device)
    noise = offline._sample_noise_for_batch(
        batch,
        chunk_size=int(cfg.POLICY.CHUNK_SIZE),
        action_dim=int(getattr(policy.config, "max_action_dim", 32)),
        device=device,
        seed=noise_seed,
    )
    return policy.predict_action_chunk(batch, noise=noise), batch


def evaluate_sweep(cfg, baseline, residual, loader, device, args) -> Dict[str, Any]:
    eval_steps = offline._resolve_eval_action_steps(
        cfg,
        args.eval_action_steps,
        int(cfg.POLICY.CHUNK_SIZE),
    )
    use_delta = bool(cfg.DATASET.USE_DELTA_ACTION)
    speed_results: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    ablation_results: Dict[str, List[Dict[str, float]]] = {
        name: [] for name in args.residual_ablations
    }

    total_batches = len(loader)
    if args.max_batches > 0:
        total_batches = min(total_batches, int(args.max_batches))
    tick = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        fps = float(getattr(loader.dataset.meta, "fps", 25))
        for speed in args.counterfactual_speeds:
            cf = _counterfactual_batch(
                batch,
                speed=float(speed),
                direction=args.direction,
                fallback_axis=args.fallback_axis,
                fps=fps,
            )
            baseline_pred, baseline_batch = _evaluate_policy_on_batch(
                baseline,
                cf,
                device,
                cfg,
                args.noise_seed,
            )
            residual_pred, residual_batch = _evaluate_policy_on_batch(
                residual,
                cf,
                device,
                cfg,
                args.noise_seed,
            )
            key = f"{float(speed):.3f}mps"
            bucket = speed_results.setdefault(key, {"baseline": [], "residual": []})
            bucket["baseline"].append(
                _final_distance_metrics(
                    baseline_pred,
                    baseline_batch,
                    fps=fps,
                    use_delta_action=use_delta,
                    eval_action_steps=eval_steps,
                )
            )
            bucket["residual"].append(
                _final_distance_metrics(
                    residual_pred,
                    residual_batch,
                    fps=fps,
                    use_delta_action=use_delta,
                    eval_action_steps=eval_steps,
                )
            )

        for ablation in args.residual_ablations:
            ablated = _ablation_batch(batch, ablation)
            pred, eval_batch = _evaluate_policy_on_batch(
                residual,
                ablated,
                device,
                cfg,
                args.noise_seed,
            )
            ablation_results[ablation].append(
                _final_distance_metrics(
                    pred,
                    eval_batch,
                    fps=fps,
                    use_delta_action=use_delta,
                    eval_action_steps=eval_steps,
                )
            )

        seen = batch_idx + 1
        if args.progress_every > 0 and (
            seen == 1 or seen % args.progress_every == 0 or seen == total_batches
        ):
            elapsed = time.perf_counter() - tick
            rate = seen / max(elapsed, 1e-9)
            eta = max(total_batches - seen, 0) / max(rate, 1e-9)
            logging.info(
                "Paired sweep progress: %d/%d batches (%.2f batches/s, ETA %.1f min).",
                seen,
                total_batches,
                rate,
                eta / 60.0,
            )

    merged_speeds = {}
    for speed, policies in speed_results.items():
        baseline_metrics = _merge_items(policies["baseline"])
        residual_metrics = _merge_items(policies["residual"])
        merged_speeds[speed] = {
            "baseline": baseline_metrics,
            "residual": residual_metrics,
            "delta": {
                key: residual_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0)
                for key in (
                    "final_object_distance",
                    "min_object_distance",
                    "action_mse",
                    "ee_tracking_mse",
                    "final_success_rate",
                )
            },
        }
    merged_ablations = {
        name: _merge_items(items) for name, items in ablation_results.items()
    }
    return {
        "counterfactual_speeds": merged_speeds,
        "residual_ablations": merged_ablations,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--residual-checkpoint", default=DEFAULT_RESIDUAL_CKPT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--shuffle-episodes", action="store_true")
    parser.add_argument("--episode-seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--eval-action-steps", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--counterfactual-speeds",
        type=float,
        nargs="+",
        default=[0.02, 0.08, 0.12, 0.18, 0.25, 0.35],
    )
    parser.add_argument(
        "--direction",
        choices=["observed", "neg_observed", "x", "y"],
        default="observed",
    )
    parser.add_argument("--fallback-axis", type=int, default=0)
    parser.add_argument(
        "--residual-ablations",
        nargs="+",
        default=["correct", "zero", "wrong_sign", "shuffle"],
        choices=["correct", "zero", "wrong_sign", "shuffle"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(asctime)s %(message)s")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = offline._load_cfg(args.cfg)
    device = torch.device(args.device)
    dataset, loader, episodes = _make_loader(cfg, args.split, args)
    baseline = offline._build_policy(
        cfg,
        dataset.meta,
        args.baseline_checkpoint,
        False,
        device,
        "baseline",
    )
    residual = offline._build_policy(
        cfg,
        dataset.meta,
        args.residual_checkpoint,
        True,
        device,
        "residual",
    )

    results = evaluate_sweep(cfg, baseline, residual, loader, device, args)
    report = {
        "evaluation_mode": "paired_counterfactual_velocity_sweep",
        "cfg": args.cfg,
        "baseline_checkpoint": args.baseline_checkpoint,
        "residual_checkpoint": args.residual_checkpoint,
        "split": args.split,
        "episodes": None if episodes is None else len(episodes),
        "device": str(device),
        "max_batches": int(args.max_batches),
        "batch_size": int(args.batch_size),
        "direction": args.direction,
        "notes": (
            "This is a paired full-VLA diagnostic: every policy sees the same DOM "
            "observations while only the dynamic-object velocity fields are swept or "
            "ablated. Headline metrics are final/time-aligned distances and MSE, not "
            "min-distance proxy success."
        ),
        "results": results,
    }
    with output.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
