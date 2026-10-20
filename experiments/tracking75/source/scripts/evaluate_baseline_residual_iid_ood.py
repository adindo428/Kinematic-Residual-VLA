#!/usr/bin/env python
"""Observed IID/OOD offline evaluation for paired baseline/residual DynamicVLA.

This evaluates the trained policies on real observed split samples. Unlike the
counterfactual paired velocity sweep, the ground-truth actions and observations
come from the same split, so action_mse is suitable for checkpoint/config
selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
DEFAULT_OUTPUT = "runs/proactive_vla/offline_compare/paired_velocity_iid_ood_report.json"
REQUIRED_CHECKPOINT_FILES = ("config.json", "model.safetensors")


def _project_relative_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    project_candidate = PROJECT_HOME / candidate
    if project_candidate.exists():
        return project_candidate
    return candidate


def _looks_like_remote_checkpoint(path: str) -> bool:
    candidate = Path(path)
    return (
        not candidate.is_absolute()
        and not path.startswith((".", "runs/"))
        and len(path.split("/")) == 2
    )


def _require_checkpoint(path: str, label: str) -> str:
    if _looks_like_remote_checkpoint(path):
        return path
    checkpoint = _project_relative_path(path)
    train_hint = (
        "Train the paired checkpoints first with:\n"
        "  sbatch slurm/train_paired_velocity_pair_h100.sbatch\n"
        "or pass --baseline-checkpoint/--residual-checkpoint to existing checkpoint directories."
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(
            f"Missing {label} checkpoint directory: {path!r}.\n{train_hint}"
        )
    missing = [
        name
        for name in REQUIRED_CHECKPOINT_FILES
        if not (checkpoint / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete {label} checkpoint at {str(checkpoint)!r}; "
            f"missing {', '.join(missing)}.\n{train_hint}"
        )
    return str(checkpoint)


def _num_workers(cfg, override: Optional[int]) -> int:
    if override is not None:
        return max(0, int(override))
    return max(0, int(getattr(cfg.CONST, "N_WORKERS", 0)))


def _resolve_episodes(cfg, split: str, args) -> Optional[List[int]]:
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


def _split_row(split: str, baseline: Dict[str, Any], residual: Dict[str, Any], comparison: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "split": split,
        "samples": baseline.get("samples", residual.get("samples", 0)),
        "baseline_action_mse": baseline.get("action_mse"),
        "residual_action_mse": residual.get("action_mse"),
        "action_mse_delta": comparison.get("action_mse_delta"),
        "action_mse_relative_reduction": comparison.get("action_mse_relative_reduction"),
        "baseline_ee_tracking_mse": baseline.get("ee_tracking_mse"),
        "residual_ee_tracking_mse": residual.get("ee_tracking_mse"),
        "ee_tracking_mse_delta": comparison.get("ee_tracking_mse_delta"),
        "baseline_final_object_distance": baseline.get("final_object_distance"),
        "residual_final_object_distance": residual.get("final_object_distance"),
        "final_object_distance_delta": comparison.get("final_object_distance_delta"),
        "baseline_success_rate": baseline.get("success_rate"),
        "residual_success_rate": residual.get("success_rate"),
        "success_rate_delta": comparison.get("success_rate_delta"),
        "baseline_final_success_rate": baseline.get("final_success_rate"),
        "residual_final_success_rate": residual.get("final_success_rate"),
        "final_success_rate_delta": comparison.get("final_success_rate_delta"),
    }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "samples",
        "baseline_action_mse",
        "residual_action_mse",
        "action_mse_delta",
        "action_mse_relative_reduction",
        "baseline_ee_tracking_mse",
        "residual_ee_tracking_mse",
        "ee_tracking_mse_delta",
        "baseline_final_object_distance",
        "residual_final_object_distance",
        "final_object_distance_delta",
        "baseline_success_rate",
        "residual_success_rate",
        "success_rate_delta",
        "baseline_final_success_rate",
        "residual_final_success_rate",
        "final_success_rate_delta",
    ]
    with output.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", default=DEFAULT_CFG)
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE_CKPT)
    parser.add_argument("--residual-checkpoint", default=DEFAULT_RESIDUAL_CKPT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--splits", nargs="+", default=["iid_eval", "ood_val", "ood_test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--shuffle-episodes", action="store_true")
    parser.add_argument("--episode-seed", type=int, default=1234)
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--eval-action-steps", type=int, default=0)
    parser.add_argument("--success-distance-m", type=float, default=0.05)
    parser.add_argument(
        "--success-distance-sweep-m",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = offline._load_cfg(args.cfg)
    device = torch.device(args.device)
    baseline_checkpoint = _require_checkpoint(args.baseline_checkpoint, "baseline")
    residual_checkpoint = _require_checkpoint(args.residual_checkpoint, "residual")
    success_distance_sweep_m = sorted(set(args.success_distance_sweep_m + [args.success_distance_m]))
    eval_action_steps = offline._resolve_eval_action_steps(
        cfg,
        args.eval_action_steps,
        int(cfg.POLICY.CHUNK_SIZE),
    )

    loaders = {}
    split_episode_counts = {}
    first_dataset = None
    for split in args.splits:
        dataset, loader, episodes = _make_loader(cfg, split, args)
        loaders[split] = loader
        split_episode_counts[split] = None if episodes is None else len(episodes)
        if first_dataset is None:
            first_dataset = dataset

    if first_dataset is None:
        raise ValueError("At least one split is required.")

    baseline = offline._build_policy(
        cfg,
        first_dataset.meta,
        baseline_checkpoint,
        False,
        device,
        "baseline",
    )
    residual = offline._build_policy(
        cfg,
        first_dataset.meta,
        residual_checkpoint,
        True,
        device,
        "residual",
    )

    report_splits = {}
    rows = []
    for split, loader in loaders.items():
        logging.info("Evaluating observed split=%s ...", split)
        gt_oracle_metrics, axis_audit = offline.evaluate_gt_oracle_and_axis_audit(
            loader,
            device,
            cfg,
            args.max_batches,
            args.success_distance_m,
            success_distance_sweep_m,
            eval_action_steps,
            args.progress_every,
        )
        baseline_metrics = offline.evaluate_policy(
            baseline,
            loader,
            device,
            cfg,
            args.max_batches,
            args.success_distance_m,
            success_distance_sweep_m,
            eval_action_steps,
            args.noise_seed,
            f"{split}/baseline",
            args.progress_every,
        )
        residual_metrics = offline.evaluate_policy(
            residual,
            loader,
            device,
            cfg,
            args.max_batches,
            args.success_distance_m,
            success_distance_sweep_m,
            eval_action_steps,
            args.noise_seed,
            f"{split}/residual",
            args.progress_every,
        )
        comparison = offline._comparison_summary(baseline_metrics, residual_metrics)
        report_splits[split] = {
            "episode_count": split_episode_counts[split],
            "gt_oracle": gt_oracle_metrics,
            "axis_velocity_audit": axis_audit,
            "baseline": baseline_metrics,
            "residual": residual_metrics,
            "comparison": comparison,
        }
        rows.append(_split_row(split, baseline_metrics, residual_metrics, comparison))

    report = {
        "evaluation_mode": "observed_iid_ood_offline",
        "primary_metric": "action_mse",
        "baseline_checkpoint": baseline_checkpoint,
        "residual_checkpoint": residual_checkpoint,
        "splits_requested": list(args.splits),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "max_episodes": args.max_episodes,
        "eval_action_steps": eval_action_steps,
        "noise_seed": args.noise_seed,
        "episode_seed": args.episode_seed,
        "success_distance_m": args.success_distance_m,
        "success_distance_sweep_m": success_distance_sweep_m,
        "summary_rows": rows,
        "splits": report_splits,
        "notes": (
            "Observed IID/OOD offline evaluation: action_mse compares policy actions "
            "against the observed ground-truth action chunk for each split. OOD validation "
            "may be used for checkpoint/config selection; ood_test is intended for final reporting. "
            "Counterfactual paired velocity sweep metrics are diagnostic only and are not used here."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    if args.csv_output:
        _write_csv(args.csv_output, rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
