from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from .config import file_hash
from .export import CAMERAS, find_records
from .math3d import dom_euler, rotation
from .storage import read_jsonl, read_records, video_summary


def validate_dataset(root, loader_check=False, repo_id="local/dom-tracking"):
    root = Path(root).resolve()
    checksums = json.loads((root/"checksums.json").read_text())
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "checksums.json" and ".cache" not in p.parts}
    if actual != set(checksums):
        raise ValueError("Dataset file inventory changed since export")
    for path, digest in checksums.items():
        if file_hash(root/path) != digest:
            raise ValueError(f"Checksum mismatch: {path}")
    info = json.loads((root/"meta/info.json").read_text())
    if info["codebase_version"] != "v2.1" or info["robot_type"] != "franka" or info["fps"] != 25:
        raise ValueError("Unexpected DOM metadata version/robot/fps")
    episodes = read_jsonl(root/"meta/episodes.jsonl")
    tasks = read_jsonl(root/"meta/tasks.jsonl")
    stats = read_jsonl(root/"meta/episodes_stats.jsonl")
    collection = json.loads((root/"meta/collection.json").read_text())
    layout = collection["provenance"]["layout"]
    offset = collection["action_horizon_frames"]
    if len(episodes) != info["total_episodes"] or len(stats) != len(episodes) or len(tasks) != info["total_tasks"]:
        raise ValueError("Episode/task/stat counts disagree")
    task_ids = {t["task_index"] for t in tasks}
    total = 0
    for eid, ep in enumerate(episodes):
        if ep["episode_index"] != eid:
            raise ValueError("Noncontiguous episode IDs")
        table = pq.read_table(root/info["data_path"].format(episode_chunk=eid//1000, episode_index=eid))
        n = ep["length"]
        if len(table) != n:
            raise ValueError("Episode length mismatch")
        columns = {name: np.asarray(table[name].to_pylist()) for name in table.column_names}
        for name in ("action", "observation.state", "observation.environment_state"):
            if columns[name].shape != (n, info["features"][name]["shape"][0]) or not np.isfinite(columns[name]).all():
                raise ValueError(f"Invalid tensor: {name}")
            if not pa.types.is_float32(table.schema.field(name).type.value_type):
                raise ValueError(f"Wrong tensor dtype: {name}")
        if not np.array_equal(columns["frame_index"], np.arange(n)) or not np.array_equal(columns["index"], np.arange(total, total+n)):
            raise ValueError("Frame/global indexing mismatch")
        if not np.all(columns["episode_index"] == eid) or not set(columns["task_index"]).issubset(task_ids):
            raise ValueError("Invalid episode/task IDs")
        if not np.allclose(columns["timestamp"], np.arange(n)/info["fps"], atol=1e-6):
            raise ValueError("Timestamp mismatch")
        frames = read_records(find_records(root/"diagnostics"/f"episode_{eid:06d}", "frames"))
        if len(frames) != n+offset:
            raise ValueError("Missing future-label continuation")
        stride = collection["provenance"]["config"]["collection"]["physics_hz"]//info["fps"]
        if [f["frame_index"] for f in frames] != list(range(n+offset)) or [f["control_step"] for f in frames] != list(range(0, (n+offset)*stride, stride)):
            raise ValueError("Raw frame/control alignment mismatch")
        if not np.allclose([f["timestamp"] for f in frames], np.arange(n+offset)/info["fps"], atol=1e-6):
            raise ValueError("Raw continuation timeline mismatch")
        expected = np.array([np.r_[f["ee_position"], dom_euler(f["ee_quaternion"]), 1] for f in frames[offset:]])
        if not np.allclose(columns["action"], expected, atol=1e-6):
            raise ValueError("Action labels are not the documented future observed poses")
        for name in CAMERAS:
            key = f"observation.images.{name}"
            summary = video_summary(root/info["video_path"].format(episode_chunk=eid//1000, video_key=key, episode_index=eid), n)
            if summary["shape"] != info["features"][key]["shape"] or summary["fps"] != info["fps"] or summary["codec"] != "av1":
                raise ValueError("Video/schema mismatch")
            times = [f["cameras"][name]["timestamp"] for f in frames[:n]]
            ids = [f["cameras"][name]["sensor_frame"] for f in frames[:n]]
            if not np.allclose(times, columns["timestamp"], atol=1e-6) or np.any(np.diff(ids) <= 0):
                raise ValueError("Camera synchronization/staleness failure")
            for frame in frames:
                camera = frame["cameras"][name]
                intrinsic = np.asarray(camera["intrinsic"])
                position = np.asarray(camera["position_world"])
                quat = np.asarray(camera["quaternion_world_ros"])
                if intrinsic.shape != (3, 3) or position.shape != (3,) or quat.shape != (4,):
                    raise ValueError("Invalid camera calibration shapes")
                if not all(np.isfinite(v).all() for v in (intrinsic, position, quat)) or not np.isclose(np.linalg.norm(quat), 1, atol=1e-4):
                    raise ValueError("Invalid camera calibration values")
                if name == "wrist_cam":
                    cfg = collection["provenance"]["config"]
                    base = rotation(layout["robot_quaternion"])
                    local = np.asarray(frame["ee_position"])+rotation(frame["ee_quaternion"]).apply(
                        np.array([.065, 0, 0])-cfg["controller"]["observation_offset"])
                    expected = np.array(layout["robot_position"])+base.apply(local)
                    expected_r = base*rotation(frame["ee_quaternion"])*rotation([0, 2**-.5, 2**-.5, 0])*Rotation.from_euler("x", np.pi)
                    if np.linalg.norm(position-expected) > .001 or np.linalg.norm((rotation(quat)*expected_r.inv()).as_rotvec()) > .001:
                        raise ValueError("Wrist extrinsics are inconsistent with the recorded tool pose")
        total += n
    if total != info["total_frames"] or info["total_videos"] != len(episodes)*3:
        raise ValueError("Dataset totals disagree")
    splits = json.loads((root/"meta/split_assignments.json").read_text())
    split_ids = [eid for values in splits.values() for eid in values]
    if sorted(split_ids) != list(range(len(episodes))):
        raise ValueError("Split assignment is incomplete or overlapping")
    groups_path = root/'meta/episode_groups.jsonl'
    if groups_path.exists():
        from .paired_export import validate_groups
        validate_groups(read_jsonl(groups_path), splits, len(episodes))
    result = dict(valid=True, episodes=len(episodes), frames=total, smoke_only=collection["smoke_only"],
                  checksums_sha256=file_hash(root/"checksums.json"), loader_checked=False)
    if loader_check:
        result["loader"] = check_loader(root, repo_id, total)
        result["loader_checked"] = True
    return result


def check_loader(root, repo_id, expected_frames):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
    if CODEBASE_VERSION != "v2.1":
        raise ValueError(f"Installed LeRobot expects {CODEBASE_VERSION}. Use the separate lerobot==0.3.3 loader-check environment; no automatic dataset migration.")
    dataset = LeRobotDataset(repo_id=repo_id, root=root, download_videos=False, video_backend="pyav")
    if len(dataset) != expected_frames:
        raise ValueError("LeRobot reader frame count differs")
    for index in sorted({0, expected_frames//2, expected_frames-1}):
        row = dataset[index]
        for key in ("action", "observation.state", "observation.environment_state", *[f"observation.images.{n}" for n in CAMERAS]):
            if key not in row or not np.isfinite(row[key].numpy()).all():
                raise ValueError(f"Loader failed to decode {key} at row {index}")
    return dict(version=importlib.metadata.version("lerobot"), codebase_version=CODEBASE_VERSION, tested_rows=3)
