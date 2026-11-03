#!/usr/bin/env python3
"""Freeze full-DOM data, grouped splits, initialization, and training-only stats."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from proactive_vla.phase_gate import PhaseGateConfig, replay_prefix_states
from utils.dom_experiment import atomic_json, digest, file_digest, grouped_stratified_split, source_digest


def repo_file_matches(path, entry):
    """Validate cached bytes against the pinned Git/LFS object, without HEAD."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size != entry.size:
        return False
    lfs = getattr(entry, "lfs", None)
    if lfs:
        expected = lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256
        return file_digest(path) == expected
    checksum = hashlib.sha1()
    checksum.update(f"blob {entry.size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2*1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest() == entry.blob_id


def prefetch_dom(data, revision):
    """Enumerate pinned objects once; fetch only missing/corrupt local bytes."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    counts = {"verified_cached": 0, "downloaded": 0, "downloaded_bytes": 0}
    def fetch(entry):
        path = Path(data)/entry.path
        hf_hub_download("hzxie/DOM", entry.path, repo_type="dataset", revision=revision,
                        local_dir=str(data), force_download=path.exists())
        if not repo_file_matches(path, entry):
            raise ValueError(f"Downloaded bytes differ from pinned DOM object: {entry.path}")
        return entry.size
    def selected(path):
        return (path.startswith("meta/") or
                path.startswith("data/") and path.endswith(".parquet") or
                path.startswith("videos/") and path.endswith(".mp4") and any(
                    f"/{camera}/" in path for camera in ("observation.images.opst_cam", "observation.images.wrist_cam")))
    pending = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for entry in HfApi().list_repo_tree("hzxie/DOM", repo_type="dataset", revision=revision, recursive=True):
            if not hasattr(entry, "size") or not selected(entry.path):
                continue
            if repo_file_matches(Path(data)/entry.path, entry):
                counts["verified_cached"] += 1
            else:
                pending.append(entry)
            if len(pending) == 64:
                counts["downloaded_bytes"] += sum(pool.map(fetch, pending))
                counts["downloaded"] += len(pending); pending = []
                print("Prefetch "+json.dumps(counts), flush=True)
        counts["downloaded_bytes"] += sum(pool.map(fetch, pending))
        counts["downloaded"] += len(pending)
    print("Prefetch complete "+json.dumps(counts), flush=True)
    return counts


def balanced_sample(records, count, seed=42):
    pools = defaultdict(list)
    rng = np.random.default_rng(seed)
    for r in records:
        pools[(r["task"], r["speed_bin"])].append(r["episode_index"])
    for pool in pools.values():
        rng.shuffle(pool)
    result = []
    # Rotate tasks first, then speeds within each task, so even small replay
    # suites cover every manipulation mode rather than the first sorted bins.
    tasks = sorted({key[0] for key in pools})
    cursors = {task: 0 for task in tasks}
    task_keys = {task: sorted(key for key in pools if key[0] == task) for task in tasks}
    while len(result) < count and any(pools.values()):
        for task in tasks:
            keys = task_keys[task]
            for _ in keys:
                key = keys[cursors[task] % len(keys)]; cursors[task] += 1
                if pools[key] and len(result) < count:
                    result.append(pools[key].pop()); break
    if len(result) != count:
        raise ValueError(f"Need {count} episodes, only {len(result)} available")
    return sorted(result)


def prepare(args):
    from huggingface_hub import HfApi, snapshot_download
    out, data = args.family_root.resolve(), args.dataset_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"manifest.json").exists():
        raise FileExistsError("Family is already frozen; choose a new family root to rebuild")
    model_revision = HfApi().model_info("hzxie/dynamic-vla-DOM", revision=args.model_revision).sha
    dataset_revision = HfApi().dataset_info("hzxie/DOM", revision=args.dataset_revision).sha
    if args.prefetch:
        atomic_json(out/"prefetch.json", {"revision": dataset_revision, **prefetch_dom(data, dataset_revision)})
    initialization = Path(snapshot_download("hzxie/dynamic-vla-DOM", revision=model_revision,
        allow_patterns=["config.json", "model.safetensors"], local_dir=str(out/"initialization")))
    # Constructors need only this backbone's config/tokenizer, not its weights.
    model_cfg = json.loads((initialization/"config.json").read_text())
    snapshot_download(model_cfg.get("vlm_model_name", "HuggingFaceTB/SmolLM2-360M"),
                      allow_patterns=["*.json", "*.model", "tokenizer*", "vocab*", "merges*"])
    info = json.loads((data/"meta/info.json").read_text())
    if info["fps"] != 25 or info["features"]["action"]["shape"] != [7]:
        raise ValueError("DOM FPS/action representation differs from the approved recipe")
    tasks = {int(r["task_index"]): json.loads(r["task"]) if isinstance(r["task"], str) else r["task"]
             for r in map(json.loads, (data/"meta/tasks.jsonl").read_text().splitlines())}
    episodes = {int(r["episode_index"]): r for r in map(json.loads, (data/"meta/episodes.jsonl").read_text().splitlines())}
    records, files, phase_files, videos = [], {}, {}, {}
    out.joinpath("phase_cache").mkdir(exist_ok=True)
    phase_cfg = PhaseGateConfig()
    for ep in range(info["total_episodes"]):
        parquet = data/info["data_path"].format(episode_chunk=ep//info["chunks_size"], episode_index=ep)
        table = pq.read_table(parquet, columns=["observation.state", "observation.environment_state", "action", "timestamp", "task_index"])
        arrays = {key: np.asarray(table[key].to_pylist()) for key in table.column_names}
        for camera in ("observation.images.opst_cam", "observation.images.wrist_cam"):
            video = data/info["video_path"].format(episode_chunk=ep//info["chunks_size"], episode_index=ep, video_key=camera)
            if not video.is_file() or video.stat().st_size == 0:
                raise FileNotFoundError(video)
            videos[str(video.relative_to(data))] = {"sha256": file_digest(video), "bytes": video.stat().st_size}
        velocities = arrays["observation.environment_state"][:, 6:9]
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError(f"Nonfinite numeric data in episode {ep}")
        mean_speed = float(np.linalg.norm(velocities, axis=-1).mean())
        content = hashlib.sha256()
        for key in ("observation.state", "observation.environment_state", "action"):
            content.update(np.asarray(arrays[key], dtype=np.float32).tobytes())
        # Prefer original demonstration IDs when published; otherwise detect
        # identical numeric trajectories (independent of language paraphrase).
        metadata = episodes[ep]
        duplicate = next((str(metadata[k]) for k in ("source_episode_id", "demonstration_id", "original_demonstration_id") if metadata.get(k) is not None), content.hexdigest())
        task = str(tasks[int(arrays["task_index"][0])]["task"])
        records.append({"episode_index": ep, "length": len(table), "task": task,
            "speed_bin": int(np.searchsorted([.05, .10, .18, .30], mean_speed, side="right")),
            "mean_speed": mean_speed, "duplicate_id": duplicate})
        files[str(ep)] = file_digest(parquet)
        compact, phases = replay_prefix_states(arrays["observation.state"], arrays["observation.environment_state"],
                                                arrays["action"], arrays["timestamp"].reshape(-1), phase_cfg)
        np.savez_compressed(out/"phase_cache"/f"{ep:06d}.npz", state=compact, phase=phases)
        phase_files[str(ep)] = file_digest(out/"phase_cache"/f"{ep:06d}.npz")
        if ep % 1000 == 0:
            print(f"Prepared {ep}/{info['total_episodes']} episodes", flush=True)
    split = grouped_stratified_split(records)
    if sum(r["length"] for r in records) != info["total_frames"]:
        raise ValueError("Episode lengths do not cover all published DOM frames")
    if not split["validation"] or not split["test"]:
        raise ValueError("Empty held-out split")
    by_id = {r["episode_index"]: r for r in records}
    pilot = balanced_sample([by_id[i] for i in split["train"]], 4096)
    sums, squares, count = np.zeros(3), np.zeros(3), 0
    for ep in split["train"]:
        path = data/info["data_path"].format(episode_chunk=ep//info["chunks_size"], episode_index=ep)
        values = np.asarray(pq.read_table(path, columns=["observation.environment_state"])[0].to_pylist(), dtype=np.float64)[:, 6:9]
        sums += values.sum(0); squares += np.square(values).sum(0); count += len(values)
    mean = sums/count
    std = np.sqrt(np.maximum(squares/count-mean**2, 1e-12))
    atomic_json(out/"split.json", {"splits": split, "seed": 42, "ratios": [.98, .01, .01]})
    atomic_json(out/"episodes.json", records)
    atomic_json(out/"pilot_episodes.json", pilot)
    # Cache one immutable validation anchor list shared by every model/profile.
    total_validation = sum(by_id[i]["length"] for i in split["validation"])
    if total_validation < 8192:
        raise ValueError("Need at least 8192 validation anchors")
    anchors = sorted(np.random.default_rng(42).choice(total_validation, 8192, replace=False).tolist())
    atomic_json(out/"validation_anchors.json", anchors)
    atomic_json(out/"validation_cases.json", balanced_sample([by_id[i] for i in split["validation"]], 64))
    atomic_json(out/"test_cases.json", balanced_sample([by_id[i] for i in split["test"]], 256))
    manifest = {"version": 1, "dataset": "hzxie/DOM", "dataset_revision": dataset_revision,
        "dataset_root": str(data), "model_revision": model_revision, "initialization": str(initialization),
        "weights_sha256": file_digest(initialization/"model.safetensors"),
        "source_sha256": source_digest(Path(__file__).resolve().parents[1]),
        "split_sha256": file_digest(out/"split.json"), "parquets_sha256": digest(files),
        "frames": info["total_frames"], "episodes": info["total_episodes"],
        "train_frames": sum(by_id[i]["length"] for i in split["train"]),
        "velocity_mean": mean.tolist(), "velocity_std": std.tolist(), "fps": info["fps"],
        "phase_config": vars(phase_cfg), "normalization": "pretrained_state_action",
        "counts": {k: len(v) for k,v in split.items()}}
    atomic_json(out/"parquet_hashes.json", files)
    atomic_json(out/"video_hashes.json", videos)
    atomic_json(out/"phase_hashes.json", phase_files)
    manifest["artifacts"] = {name: file_digest(out/name) for name in (
        "episodes.json", "pilot_episodes.json", "validation_anchors.json", "validation_cases.json", "test_cases.json",
        "parquet_hashes.json", "video_hashes.json", "phase_hashes.json", "initialization/config.json")}
    manifest["fingerprint"] = digest(manifest)
    atomic_json(out/"manifest.json", manifest)
    print(json.dumps({"prepared": str(out), "counts": manifest["counts"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM"))
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--prefetch", action="store_true", help="Fetch missing data and both cameras at the pinned revision")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
