#!/usr/bin/env python
"""One-factor DOM residual-flow ablations.

This script reuses the offline DOM proxy from proactive_residual_offline_compare
and evaluates all residual variants on the same held-out samples. It is meant for
quick calibration sweeps over analytical drift scale, per-axis scale, and
lookahead horizon.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from proactive_residual_offline_compare import (
    PROJECT_HOME,
    _build_dataset,
    _build_policy,
    _comparison_summary,
    _load_cfg,
    _resolve_eval_action_steps,
    evaluate_policy,
)


def _parse_axis_scale(value: str) -> list[float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Axis scale must have three comma-separated values, got {value!r}."
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Axis scale must contain numeric values, got {value!r}."
        ) from exc


def _policy_value(policy_cfg, key: str, default: Any) -> Any:
    return policy_cfg.get(key, default)


def _base_residual_settings(policy_cfg) -> Dict[str, Any]:
    return {
        "RESIDUAL_DRIFT_MODE": _policy_value(policy_cfg, "RESIDUAL_DRIFT_MODE", "trajectory"),
        "RESIDUAL_DRIFT_SCALE": float(_policy_value(policy_cfg, "RESIDUAL_DRIFT_SCALE", 1.0)),
        "RESIDUAL_AXIS_SCALE_XYZ": list(
            _policy_value(policy_cfg, "RESIDUAL_AXIS_SCALE_XYZ", [1.0, 1.0, 1.0])
        ),
        "RESIDUAL_LOOKAHEAD_HORIZON_S": float(
            _policy_value(policy_cfg, "RESIDUAL_LOOKAHEAD_HORIZON_S", 1.0)
        ),
    }


def _variant_name(overrides: Dict[str, Any]) -> str:
    axis = overrides["RESIDUAL_AXIS_SCALE_XYZ"]
    axis_text = ",".join(f"{float(v):g}" for v in axis)
    return (
        f"mode={overrides['RESIDUAL_DRIFT_MODE']}"
        f"_scale={float(overrides['RESIDUAL_DRIFT_SCALE']):g}"
        f"_axis={axis_text}"
        f"_horizon={float(overrides['RESIDUAL_LOOKAHEAD_HORIZON_S']):g}"
    )


def _dedupe_variants(variants: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    seen = set()
    deduped = []
    for variant in variants:
        key = json.dumps(variant["policy_overrides"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def _build_variants(args, cfg) -> list[Dict[str, Any]]:
    base = _base_residual_settings(cfg.POLICY)
    variants: List[Dict[str, Any]] = [
        {
            "name": "configured",
            "sweep": "configured",
            "policy_overrides": copy.deepcopy(base),
        }
    ]

    for scale in args.drift_scales:
        overrides = copy.deepcopy(base)
        overrides["RESIDUAL_DRIFT_SCALE"] = float(scale)
        variants.append(
            {
                "name": f"drift_scale={float(scale):g}",
                "sweep": "drift_scale",
                "policy_overrides": overrides,
            }
        )

    for axis_scale in args.axis_scales:
        overrides = copy.deepcopy(base)
        overrides["RESIDUAL_AXIS_SCALE_XYZ"] = axis_scale
        variants.append(
            {
                "name": f"axis_scale={','.join(f'{v:g}' for v in axis_scale)}",
                "sweep": "axis_scale",
                "policy_overrides": overrides,
            }
        )

    for horizon_s in args.lookahead_horizons:
        overrides = copy.deepcopy(base)
        overrides["RESIDUAL_DRIFT_MODE"] = "lookahead"
        overrides["RESIDUAL_LOOKAHEAD_HORIZON_S"] = float(horizon_s)
        variants.append(
            {
                "name": f"lookahead_horizon={float(horizon_s):g}",
                "sweep": "lookahead_horizon",
                "policy_overrides": overrides,
            }
        )

    for variant in variants:
        variant["label"] = _variant_name(variant["policy_overrides"])
    return _dedupe_variants(variants)


def _apply_policy_overrides(cfg, overrides: Dict[str, Any]):
    variant_cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        variant_cfg.POLICY[key] = value
    return variant_cfg


def _best_variant(variants: list[Dict[str, Any]], metric: str, higher_is_better: bool) -> Dict[str, Any] | None:
    if not variants:
        return None

    def score(variant: Dict[str, Any]) -> float:
        value = variant["residual"].get(metric)
        if value is None:
            return float("-inf") if higher_is_better else float("inf")
        return float(value)

    return max(variants, key=score) if higher_is_better else min(variants, key=score)


def _summarize_best(variants: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any] | None]:
    best_success = _best_variant(variants, "success_rate", higher_is_better=True)
    best_min_dist = _best_variant(variants, "min_object_distance", higher_is_better=False)
    best_final_dist = _best_variant(variants, "final_object_distance", higher_is_better=False)
    best_legacy_final_dist = _best_variant(
        variants, "legacy_final_object_distance", higher_is_better=False
    )

    def pick(variant: Dict[str, Any] | None, metric: str) -> Dict[str, Any] | None:
        if variant is None:
            return None
        return {
            "name": variant["name"],
            "label": variant["label"],
            "sweep": variant["sweep"],
            "metric": metric,
            "value": variant["residual"].get(metric),
            "policy_overrides": variant["policy_overrides"],
        }

    return {
        "success_rate": pick(best_success, "success_rate"),
        "min_object_distance": pick(best_min_dist, "min_object_distance"),
        "final_object_distance": pick(best_final_dist, "final_object_distance"),
        "legacy_final_object_distance": pick(best_legacy_final_dist, "legacy_final_object_distance"),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default=str(PROJECT_HOME / "configs" / "residual_dynamicvla_dom_tiny.yaml"))
    parser.add_argument("--baseline-checkpoint", default="hzxie/dynamic-vla-DOM")
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument(
        "--output",
        default="runs/proactive_vla/offline_compare/residual_ablation_report.json",
    )
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
    )
    parser.add_argument("--drift-scales", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument(
        "--axis-scales",
        type=_parse_axis_scale,
        nargs="+",
        default=[
            [1.0, 1.0, 1.0],
            [0.5, 1.0, 1.0],
            [1.0, 0.5, 1.0],
            [1.0, 1.0, 0.5],
            [0.5, 0.5, 1.0],
            [0.75, 0.75, 1.0],
        ],
        help="One or more XYZ axis scales, each formatted as x,y,z.",
    )
    parser.add_argument("--lookahead-horizons", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--noise-seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = _load_cfg(args.cfg)
    device = torch.device(args.device)
    success_distance_sweep_m = sorted(set(args.success_distance_sweep_m + [args.success_distance_m]))
    eval_action_steps = _resolve_eval_action_steps(cfg, args.eval_action_steps, int(cfg.POLICY.CHUNK_SIZE))

    dataset = _build_dataset(cfg, "test")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg.CONST.N_WORKERS,
        pin_memory=cfg.DATASET.PIN_MEMORY,
    )

    logging.info("Evaluating baseline once on the shared DOM sample slice ...")
    baseline = _build_policy(cfg, dataset.meta, args.baseline_checkpoint, False, device)
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
    )
    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    variants = _build_variants(args, cfg)
    results = []
    for idx, variant in enumerate(variants, start=1):
        logging.info("Evaluating residual variant %d/%d: %s", idx, len(variants), variant["label"])
        variant_cfg = _apply_policy_overrides(cfg, variant["policy_overrides"])
        residual = _build_policy(variant_cfg, dataset.meta, args.residual_checkpoint, True, device)
        residual_metrics = evaluate_policy(
            residual,
            loader,
            device,
            variant_cfg,
            args.max_batches,
            args.success_distance_m,
            success_distance_sweep_m,
            eval_action_steps,
            args.noise_seed,
        )
        del residual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        results.append(
            {
                "name": variant["name"],
                "label": variant["label"],
                "sweep": variant["sweep"],
                "policy_overrides": variant["policy_overrides"],
                "residual": residual_metrics,
                "comparison": _comparison_summary(baseline_metrics, residual_metrics),
            }
        )

    report = {
        "evaluation_mode": "offline_proxy_ablation",
        "simulator_available": False,
        "baseline_checkpoint": args.baseline_checkpoint,
        "residual_checkpoint": args.residual_checkpoint,
        "success_distance_m": args.success_distance_m,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "eval_action_steps": eval_action_steps,
        "noise_seed": args.noise_seed,
        "baseline": baseline_metrics,
        "variants": results,
        "best_by_metric": _summarize_best(results),
        "notes": (
            "All variants are evaluated on the same DOM sample slice. Primary success "
            "uses minimum end-effector distance to the constant-velocity object forecast "
            "over the evaluated action horizon; legacy_* fields retain full-chunk final endpoint scoring."
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
