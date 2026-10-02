"""Strict, JSON-serializable configuration and stable provenance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

DEFAULT = Path(__file__).resolve().parents[1] / "configs/pilot.yaml"
PATH_KEYS = ("assets.scenes", "assets.objects", "assets.benchmark", "assets.manifest",
             "scene.layout", "collection.output")


def load_config(path=DEFAULT, overrides=()):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for expression in overrides:
        key, sep, value = expression.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value: {expression}")
        obj = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            obj = obj[part]
        if parts[-1] not in obj:
            raise ValueError(f"Unknown configuration key: {key}")
        obj[parts[-1]] = yaml.safe_load(value)
    for key in PATH_KEYS:
        group, field = key.split(".")
        cfg[group][field] = str((path.parent / cfg[group][field]).resolve())
    robot = cfg["scene"]["robot_usd"]
    if robot and "://" not in robot:
        cfg["scene"]["robot_usd"] = str((path.parent / robot).resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    c, k, t = (cfg[g] for g in ("collection", "controller", "trajectory"))
    if not 0 <= k["exit_speed"] < k["enter_speed"]:
        raise ValueError("Require 0 <= exit_speed < enter_speed")
    for field in ("episodes", "physics_hz", "fps", "max_attempts", "candidates", "future_frames"):
        if type(c[field]) is not int or c[field] < 1:
            raise ValueError(f"collection.{field} must be a positive integer")
    if c["physics_hz"] % c["fps"] or c["seconds"] <= 0 or not float(c["seconds"] * c["fps"]).is_integer():
        raise ValueError("Require an integer physics/capture ratio and whole episode frames")
    if c["raw_format"] not in ("jsonl", "parquet") or c["purpose"] not in ("training", "smoke"):
        raise ValueError("Invalid raw_format or purpose")
    if not 0 < c["balance_tolerance"] < 1/3 or c["warmup_seconds"] < 0:
        raise ValueError("Invalid balancing or warmup configuration")
    for key in ("center", "half_extents"):
        v = np.asarray(t[key], dtype=float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError(f"trajectory.{key} must be a finite 3-vector")
    if min(t["half_extents"]) <= 0:
        raise ValueError("Workspace half extents must be positive")
    if not 0 <= t["workspace_margin"] < min(t["half_extents"]):
        raise ValueError("Workspace margin must be nonnegative and smaller than every half extent")
    if not 0 < t["frequency_range"][0] <= t["frequency_range"][1]:
        raise ValueError("Invalid frequency range")
    for key in ("kp", "orientation_kp", "hover_height", "max_linear_speed", "max_angular_speed", "damping", "joint_margin"):
        if not np.isfinite(k[key]) or k[key] <= 0:
            raise ValueError(f"controller.{key} must be finite and positive")
    if not np.isfinite(k["kd"]) or k["kd"] < 0:
        raise ValueError("Invalid derivative gain")
    if min(t["max_speed"], t["max_acceleration"]) <= 0:
        raise ValueError("Motion bounds must be positive")
    if t["kind"] not in ("random", "stationary", "positive", "negative", "neutral", "switch"):
        raise ValueError("Unknown trajectory kind")
    if c["purpose"] == "training" and t["kind"] != "random":
        raise ValueError("Named test trajectories are for smoke runs only")
    for key in ("observation_offset", "action_offset", "down_quaternion"):
        value = np.asarray(k[key], float)
        expected = 4 if key == "down_quaternion" else 3
        if value.shape != (expected,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}")
    if not np.isclose(np.linalg.norm(k["down_quaternion"]), 1):
        raise ValueError("down_quaternion must be normalized")
    if any(cfg["camera"][x] < 2 or cfg["camera"][x] % 2 for x in ("width", "height")):
        raise ValueError("Video dimensions must be positive even integers")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_fingerprint(cfg):
    value = copy.deepcopy(cfg)
    # Output location is not part of the physical experiment.
    value["collection"].pop("output")
    return fingerprint(value)
