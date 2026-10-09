#!/usr/bin/env python
"""Compare DA3 runtime velocity estimates with DOM ground-truth object velocity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _stack_column(table, key: str) -> np.ndarray:
    values = table[key].to_pandas().values
    return np.stack(values).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-npz", required=True)
    parser.add_argument("--dom-parquet", required=True)
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--output", default="runs/proactive_vla/da3_dom_velocity_report.json")
    args = parser.parse_args()

    traj = np.load(args.trajectory_npz, allow_pickle=True)
    frame_ids = traj["frame_ids"].astype(np.int64)
    pred_vel = traj["velocity_xyz"][:, args.object_index, :].astype(np.float32)
    valid = traj["valid"][:, args.object_index].astype(bool)

    table = pq.read_table(args.dom_parquet)
    env_state = _stack_column(table, "observation.environment_state")
    gt_vel_all = env_state[:, 6:9]

    in_range = (frame_ids >= 0) & (frame_ids < len(gt_vel_all))
    keep = valid & in_range & np.isfinite(pred_vel).all(axis=1)
    gt_vel = gt_vel_all[frame_ids[keep]]
    pred_vel = pred_vel[keep]

    if pred_vel.size == 0:
        report = {
            "samples": 0,
            "missing_track_rate": 1.0,
            "velocity_l2": None,
            "velocity_direction_cosine": None,
        }
    else:
        pred_norm = np.linalg.norm(pred_vel, axis=1)
        gt_norm = np.linalg.norm(gt_vel, axis=1)
        denom = np.maximum(pred_norm * gt_norm, 1e-8)
        cosine = np.sum(pred_vel * gt_vel, axis=1) / denom
        report = {
            "samples": int(pred_vel.shape[0]),
            "missing_track_rate": float(1.0 - keep.mean()),
            "velocity_l2": float(np.mean(np.linalg.norm(pred_vel - gt_vel, axis=1))),
            "velocity_direction_cosine": float(np.mean(cosine)),
            "mean_da3_velocity_norm": float(np.mean(pred_norm)),
            "mean_dom_velocity_norm": float(np.mean(gt_norm)),
            "coordinate_note": "Velocity magnitudes are only directly comparable after DA3 scale alignment.",
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
