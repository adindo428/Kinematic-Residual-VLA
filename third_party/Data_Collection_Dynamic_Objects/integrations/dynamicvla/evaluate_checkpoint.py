"""Offline saved-DynamicVLA checkpoint evaluation on held-out tracking recordings.

No simulator actions are applied. Predictions are compared with recorded future
pose labels, using the checkpoint's normalizers and absolute/delta convention.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def offsets_from_config(config):
    if config.get("type") != "dynamicvla":
        raise ValueError("This adapter supports saved DynamicVLA baseline/residual checkpoints")
    n_obs, chunk = int(config["n_obs_steps"]), int(config["chunk_size"])
    temporal = config.get("delta_timestamps") or {}
    observation = temporal.get("observation", list(range(1 - n_obs, 1)))
    action = temporal.get("action", list(range(chunk)))
    if len(observation) != n_obs or not observation or observation[-1] != 0:
        raise ValueError("Observation offsets must match n_obs_steps and end at the current frame")
    if any(type(i) is not int or i > 0 for i in observation) or observation != sorted(set(observation)):
        raise ValueError("Observation offsets must be increasing, nonpositive integer frame offsets")
    if action != list(range(chunk)):
        raise ValueError("Only contiguous action chunks beginning at the current frame are supported")
    if config.get("input_features", {}).get("observation.state", {}).get("shape") != [6]:
        raise ValueError("Checkpoint must expect a six-value DOM Euler tool state")
    if config.get("output_features", {}).get("action", {}).get("shape") != [7]:
        raise ValueError("Checkpoint must output seven-value DOM Euler pose/gripper actions")
    if not isinstance(config.get("use_delta_action"), bool):
        raise ValueError("Checkpoint must explicitly specify use_delta_action")
    return observation, action


def errors(prediction, target, current_state, use_delta, valid):
    prediction, target = np.asarray(prediction, dtype=float).copy(), np.asarray(target, dtype=float)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 7:
        raise ValueError("Prediction shape must match the (chunk, 7) target")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite action prediction or label")
    if use_delta:
        prediction[:, :6] += np.asarray(current_state)[:6]
    prediction, target = prediction[valid], target[valid]
    xyz = np.linalg.norm(prediction[:, :3] - target[:, :3], axis=1)
    angle = (Rotation.from_euler("xyz", target[:, 3:6]).inv() *
             Rotation.from_euler("xyz", prediction[:, 3:6])).magnitude()
    return xyz, angle, np.abs(prediction[:, 6] - target[:, 6])


def evaluate(policy, dataset, config, device, max_frames=0, stride=1):
    import torch
    import torchvision.transforms.v2.functional as F

    sums = np.zeros(3)
    maxima = np.zeros(3)
    frames = steps = 0
    per_episode = {}
    for index in range(0, len(dataset), stride):
        if max_frames and frames >= max_frames:
            break
        row = dataset[index]
        metadata = json.loads(row["task"])
        prompt = metadata.get("instruction")
        if metadata.get("task") != "track" or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Expected explicit tracking instruction metadata")
        batch = {"task": [prompt]}
        for key, feature in config["input_features"].items():
            if key not in row:
                raise ValueError(f"Checkpoint input unavailable in dataset: {key}")
            tensor = row[key]
            if key.startswith("observation.images."):
                tensor = F.resize(tensor, feature["shape"][-2:], antialias=True)
            if tuple(tensor.shape[1:]) != tuple(feature["shape"]):
                raise ValueError(f"Incompatible feature {key}: {tuple(tensor.shape)}")
            batch[key] = tensor.unsqueeze(0).to(device)
        # Each anchor has its full recorded history; do not carry an action queue
        # or cached observation across unrelated anchors or episode boundaries.
        policy.reset()
        with torch.inference_mode():
            prediction = policy.predict_action_chunk(batch)[0].detach().cpu().numpy()
        target = row["action"].numpy()
        valid = ~row["action_is_pad"].numpy().astype(bool)
        xyz, angle, gripper = errors(prediction, target, row["observation.state"][-1].numpy(),
                                      config["use_delta_action"], valid)
        values = np.stack([xyz, angle, gripper], axis=1)
        sums += np.square(values).sum(axis=0)
        maxima = np.maximum(maxima, values.max(axis=0))
        frames += 1
        steps += len(values)
        episode = int(row["episode_index"])
        item = per_episode.setdefault(episode, {"anchors": 0, "valid_action_steps": 0, "xyz_squared_sum": 0.0})
        item["anchors"] += 1
        item["valid_action_steps"] += len(values)
        item["xyz_squared_sum"] += float(np.square(xyz).sum())
    if not steps:
        raise ValueError("No unpadded labels evaluated")
    for item in per_episode.values():
        item["position_label_rmse_m"] = (item.pop("xyz_squared_sum") / item["valid_action_steps"]) ** .5
    rmse = np.sqrt(sums / steps)
    return {"anchors": frames, "valid_action_steps": steps,
            "position_label_rmse_m": float(rmse[0]), "max_position_label_error_m": float(maxima[0]),
            "orientation_label_rmse_deg": float(np.rad2deg(rmse[1])),
            "max_orientation_label_error_deg": float(np.rad2deg(maxima[1])),
            "gripper_label_rmse": float(rmse[2]), "per_episode": per_episode,
            "interpretation": "Offline action-label agreement; not closed-loop tracking RMSE or task success."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=Path(__file__).resolve().parents[3] / "DynamicVLA")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory containing config.json and model.safetensors")
    parser.add_argument("--dataset", type=Path, required=True, help="Downloaded, revision-pinned DOM tracking export")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-frames", type=int, default=0, help="0 evaluates every selected frame")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    dataset_root, checkpoint, upstream = args.dataset.resolve(), args.checkpoint.resolve(), args.upstream.resolve()
    if args.output.resolve().is_relative_to(dataset_root):
        parser.error("Keep evaluation reports outside the immutable dataset")
    if args.output.exists():
        parser.error("Output already exists; choose a new report path")
    if args.stride < 1 or args.max_frames < 0:
        parser.error("stride must be positive and max-frames nonnegative")
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint / name).is_file():
            parser.error(f"Missing checkpoint file: {checkpoint / name}")
    config = json.loads((checkpoint / "config.json").read_text())
    observations, actions = offsets_from_config(config)
    if importlib.metadata.version("lerobot") != "0.3.3":
        parser.error("Use the separate DynamicVLA environment with LeRobot 0.3.3")
    if not (upstream / "utils/helpers.py").is_file():
        parser.error(f"DynamicVLA checkout missing: {upstream}")

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; select --device cpu for a CPU check")
    info = json.loads((dataset_root / "meta/info.json").read_text())
    splits = json.loads((dataset_root / "meta/split_assignments.json").read_text())
    episodes = splits[args.split]
    if not episodes or set(episodes) & set(splits["train"]):
        raise ValueError("Evaluation split is empty or overlaps training episodes")
    deltas = {key: [i / info["fps"] for i in observations]
              for key in info["features"] if key.startswith("observation.")}
    deltas["action"] = [i / info["fps"] for i in actions]
    dataset = LeRobotDataset("adi2440/dom-franka-tracking", root=dataset_root,
                             episodes=episodes, delta_timestamps=deltas, video_backend="pyav")
    sys.path.insert(0, str(upstream))
    from utils.helpers import get_policy_cfg, get_policy_class
    policy_cfg = get_policy_cfg(cfg_file=checkpoint / "config.json")
    policy_cfg.enable_streaming = False
    policy_cfg.device = args.device
    policy = get_policy_class(config["type"]).from_pretrained(str(checkpoint), config=policy_cfg)
    policy = policy.to(args.device).eval()
    result = evaluate(policy, dataset, config, args.device, args.max_frames, args.stride)
    result.update(checkpoint=str(checkpoint), checkpoint_sha256={name: sha256(checkpoint / name)
                  for name in ("config.json", "model.safetensors")},
                  dataset=str(dataset_root), dataset_checksums_sha256=sha256(dataset_root / "checksums.json"),
                  split=args.split, selected_episodes=episodes, seed=args.seed, stride=args.stride,
                  observation_frame_offsets=observations, action_frame_offsets=actions,
                  use_delta_action=config["use_delta_action"],
                  upstream_commit=subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip(),
                  policy_source_sha256=sha256(upstream / "policies/dynamicvla/modeling_dynamicvla.py"),
                  versions={name: importlib.metadata.version(name) for name in ("torch", "lerobot", "av")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
