"""Explicit LeRobot v2.1 writer, independent of the installed LeRobot writer version."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import file_hash
from .math3d import dom_euler
from .storage import atomic_json, read_records, video_summary, write_jsonl

CAMERAS = ("wrist_cam", "opst_cam", "side_cam")
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
CORE_NAMES = ["ee_pos_x", "ee_pos_y", "ee_pos_z", "ee_rot_x", "ee_rot_y", "ee_rot_z"]


def find_records(episode, name):
    found = [p for ext in ("parquet", "jsonl") if (p := episode/f"{name}.{ext}").is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected one {name} file in {episode}")
    return found[0]


def numeric_stats(array):
    a = np.asarray(array)
    if a.ndim == 1:
        a = a[:, None]
    return dict(min=a.min(axis=0).tolist(), max=a.max(axis=0).tolist(), mean=a.mean(axis=0).tolist(),
                std=a.std(axis=0).tolist(), count=[len(a)])


def aggregate_stats(all_stats):
    result = {}
    for key in all_stats[0]:
        stats = [s[key] for s in all_stats]
        weights = np.array([s["count"][0] for s in stats])
        means = np.array([s["mean"] for s in stats])
        stds = np.array([s["std"] for s in stats])
        mean = np.average(means, axis=0, weights=weights)
        variance = np.average(stds**2+(means-mean)**2, axis=0, weights=weights)
        result[key] = dict(min=np.min([s["min"] for s in stats], axis=0).tolist(),
                           max=np.max([s["max"] for s in stats], axis=0).tolist(),
                           mean=mean.tolist(), std=np.sqrt(variance).tolist(), count=[int(weights.sum())])
    return result


def feature_spec(camera, fps):
    f = {"action": dict(dtype="float32", shape=[7], names=CORE_NAMES+["gripper"]),
         "observation.state": dict(dtype="float32", shape=[6], names=CORE_NAMES),
         "observation.environment_state": dict(dtype="float32", shape=[9], names=[
             "object_pos_x", "object_pos_y", "object_pos_z", "object_rot_x", "object_rot_y", "object_rot_z",
             "object_vel_x", "object_vel_y", "object_vel_z"])}
    for name in CAMERAS:
        video_info = {"has_audio": False, "video.codec": "av1", "video.fps": fps,
                      "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                      "video.height": camera["height"], "video.width": camera["width"], "video.channels": 3,
                      "video.channel": 3}
        f[f"observation.images.{name}"] = dict(dtype="video", shape=[camera["height"], camera["width"], 3],
                                               names=["height", "width", "channel"], info=video_info, video_info=video_info)
    for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        f[name] = dict(dtype="float32" if name == "timestamp" else "int64", shape=[1], names=None)
    return f


def export_dom(run, destination, allow_smoke=False):
    run, destination = Path(run), Path(destination)
    if destination.exists() or destination.with_name(destination.name+".partial").exists():
        raise ValueError("Export destination/partial exists; choose a fresh destination")
    provenance = json.loads((run/"provenance.json").read_text())
    cfg = provenance["config"]
    smoke = cfg["collection"]["purpose"] != "training"
    if smoke and not allow_smoke:
        raise ValueError("Smoke data cannot become a training export; use --allow-smoke only for loader testing")
    report = json.loads((run/"report.json").read_text())
    if not smoke and not (report["complete"] and report["balanced"]):
        raise ValueError("Run has not met collection/balance acceptance gates")
    episodes = sorted((run/"episodes").glob("episode_*"))
    if not episodes:
        raise ValueError("No accepted episodes")
    if cfg["collection"]["fps"] != 25 or (cfg["camera"]["width"], cfg["camera"]["height"]) != (480, 360):
        raise ValueError("DOM export requires 25 FPS and 480x360 RGB; collect with that profile")
    if cfg["camera"]["codec"] != "libsvtav1":
        raise ValueError("DOM export requires AV1 RGB videos")
    tmp = destination.with_name(destination.name+".partial")
    (tmp/"meta").mkdir(parents=True)
    (tmp/"diagnostics").mkdir()
    c = cfg["collection"]
    fps, offset = c["fps"], c["future_frames"]
    task = json.dumps({"task": "track", "objects": [Path(provenance["layout"]["object_usd"]).stem],
                       "containers": [], "instruction": "Hover above the moving object and tilt according to its lateral velocity."})
    meta, episode_stats, camera_rows = [], [], []
    total = 0
    for eid, ep in enumerate(episodes):
        ep_info = json.loads((ep/"episode.json").read_text())
        if ep_info["metrics"]["rejection_reasons"]:
            raise ValueError(f"Rejected episode found in accepted directory: {ep}")
        frames = read_records(find_records(ep, "frames"))
        n = ep_info["frame_count"]
        if len(frames) != n+offset:
            raise ValueError(f"Continuation length mismatch in {ep}")
        if not np.allclose([f["timestamp"] for f in frames], np.arange(n+offset)/fps):
            raise ValueError(f"Nonuniform frame timestamps in {ep}")
        columns = {"observation.state": [], "observation.environment_state": [], "action": []}
        for index, frame in enumerate(frames[:n]):
            future = frames[index+offset]
            columns["observation.state"].append(np.r_[frame["ee_position"], dom_euler(frame["ee_quaternion"])])
            columns["observation.environment_state"].append(np.r_[frame["object_position"], dom_euler(frame["object_quaternion"]), frame["object_velocity"]])
            columns["action"].append(np.r_[future["ee_position"], dom_euler(future["ee_quaternion"]), 1.0])
        for key in list(columns):
            columns[key] = np.asarray(columns[key], dtype=np.float32)
        columns.update(timestamp=np.arange(n, dtype=np.float32)/fps, frame_index=np.arange(n, dtype=np.int64),
                       episode_index=np.full(n, eid, dtype=np.int64), index=np.arange(total, total+n, dtype=np.int64),
                       task_index=np.zeros(n, dtype=np.int64))
        fields, arrays = [], []
        for key, values in columns.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {key} in {ep}")
            dtype = pa.list_(pa.float32(), values.shape[1]) if values.ndim == 2 else (pa.float32() if key == "timestamp" else pa.int64())
            fields.append(pa.field(key, dtype))
            arrays.append(pa.array(values.tolist(), type=dtype))
        data_path = tmp/DATA_PATH.format(episode_chunk=eid//1000, episode_index=eid)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_arrays(arrays, schema=pa.schema(fields)), data_path)
        stats = {key: numeric_stats(value) for key, value in columns.items()}
        for name in CAMERAS:
            key = f"observation.images.{name}"
            summary = video_summary(ep/f"{name}.mp4", n)
            if summary["fps"] != fps or summary["shape"] != [360, 480, 3] or summary["codec"] != "av1":
                raise ValueError(f"Video profile mismatch: {ep}/{name}")
            stats[key] = summary["stats"]
            target = tmp/VIDEO_PATH.format(episode_chunk=eid//1000, video_key=key, episode_index=eid)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ep/f"{name}.mp4", target)
        diag = tmp/"diagnostics"/f"episode_{eid:06d}"
        diag.mkdir()
        for p in (find_records(ep, "frames"), find_records(ep, "controls"), ep/"episode.json"):
            shutil.copy2(p, diag/p.name)
        meta.append(dict(episode_index=eid, tasks=[task], length=n))
        episode_stats.append(dict(episode_index=eid, stats=stats))
        camera_rows.append(dict(episode_index=eid, cameras=frames[0]["cameras"],
                                per_frame_extrinsics=f"diagnostics/episode_{eid:06d}/{find_records(ep, 'frames').name}"))
        total += n
    features = feature_spec(cfg["camera"], fps)
    info = dict(codebase_version="v2.1", robot_type="franka", total_episodes=len(episodes), total_frames=total,
                total_tasks=1, total_videos=len(episodes)*3, total_chunks=(len(episodes)+999)//1000,
                chunks_size=1000, fps=fps, splits={"train": f"0:{len(episodes)}"},
                data_path=DATA_PATH, video_path=VIDEO_PATH, features=features)
    atomic_json(tmp/"meta/info.json", info)
    atomic_json(tmp/"meta/stats.json", aggregate_stats([s["stats"] for s in episode_stats]))
    write_jsonl(tmp/"meta/episodes.jsonl", meta)
    write_jsonl(tmp/"meta/episodes_stats.jsonl", episode_stats)
    write_jsonl(tmp/"meta/camera.jsonl", camera_rows)
    write_jsonl(tmp/"meta/tasks.jsonl", [dict(task_index=0, task=task)])
    ids = np.random.default_rng(c["seed"]).permutation(len(episodes)).tolist()
    train_end, val_end = int(len(ids)*0.8), int(len(ids)*0.9)
    atomic_json(tmp/"meta/split_assignments.json", dict(train=sorted(ids[:train_end]), validation=sorted(ids[train_end:val_end]), test=sorted(ids[val_end:])))
    atomic_json(tmp/"meta/collection.json", dict(provenance=provenance, report=report, smoke_only=smoke,
                action_horizon_frames=offset, action_horizon_seconds=offset/fps,
                action_frame="robot-base axes; observation tool point at hand +0.1034 m",
                actual_commands="diagnostics/*/controls.*", training_loader_status="pending-explicit-check"))
    if not smoke:
        audit_path = run/"asset_manifest.json"
        if not audit_path.is_file():
            raise ValueError("Training run is missing its frozen asset-split manifest")
        shutil.copy2(audit_path, tmp/"meta/asset_manifest.json")
    (tmp/"README.md").write_text(
        "---\ntags:\n- robotics\n- lerobot\n- franka\n- synthetic\n---\n\n"
        "# Franka dynamic-object hover tracking\n\n"
        "DOM-compatible tracking extension, not an official addition to hzxie/DOM. "
        "Three RGB views, one target, and velocity-conditioned orientation modes.\n\n"
        f"{len(episodes)} episodes, {total} frames, {fps} FPS. Smoke-only: {smoke}.\n\n"
        "The `action` field is a future measured observation-tool pose plus gripper +1 (open), "
        f"{offset} frames ahead. It is not the raw command; controls and tool-offset conventions are in diagnostics. "
        "The source DOM pipeline uses future-pose replay; this extension labels its own recorded trajectory directly.\n\n"
        "Use meta/split_assignments.json to select disjoint episode splits; meta/info.json lists all episodes as train for DOM loader compatibility. "
        "Split assets and motion support are shared. No official benchmark-referenced assets were selected. "
        "Camera visibility checks measure frustum inclusion, not segmentation-based occlusion.\n\n"
        "Provenance and configuration: meta/collection.json. Retain DOM and underlying asset attribution; "
        "this exporter does not assign a new license to third-party assets.\n\n"
        "Sources: [DOM](https://huggingface.co/datasets/hzxie/DOM), "
        "[DynamicVLA](https://github.com/hzxie/DynamicVLA).\n")
    atomic_json(tmp/"checksums.json", {str(p.relative_to(tmp)): file_hash(p) for p in sorted(tmp.rglob("*")) if p.is_file()})
    tmp.rename(destination)
    return dict(output=str(destination.resolve()), episodes=len(episodes), frames=total)
