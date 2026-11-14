#!/usr/bin/env python3
"""Re-extract exact frames and provenance from the local DOM dataset (CPU only)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent
SELECTIONS = {5881: [3, 22, 60, 99], 103899: [3, 23, 43, 64, 83, 105],
              3265: [3, 47, 93, 136, 181, 226], 5224: [3, 100, 197]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT.parents[1] /
        "scratch/dom45/cache/huggingface/lerobot/hzxie/DOM")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    info = json.loads((root / "meta/info.json").read_text())
    episodes = {}
    with (root / "meta/episodes.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["episode_index"] in SELECTIONS:
                episodes[row["episode_index"]] = row
    manifest = {"dataset": "hzxie/DOM", "dataset_root": str(root),
        "source_kind": "Released simulation demonstrations; not KRFM policy rollouts",
        "metadata_info_sha256": sha(root / "meta/info.json"), "fps": info["fps"],
        "image_processing": "Exact decoded RGB frames. No recoloring, inpainting, sharpening, "
            "or synthesized content. Figure crops and vector annotations are separate.",
        "episodes": [], "frames": []}
    (ROOT / "assets").mkdir(exist_ok=True)
    for ep, indices in SELECTIONS.items():
        rel = info["data_path"].format(episode_chunk=ep//info["chunks_size"], episode_index=ep)
        table = pq.read_table(root / rel)
        timestamps = table["timestamp"].to_pylist()
        manifest["episodes"].append({"episode_index": ep, "metadata": episodes[ep],
            "parquet": rel, "parquet_sha256": sha(root/rel),
            "initial_velocity_stored_xyz": table["observation.environment_state"].to_pylist()[3][6:9]})
        cameras = ["opst_cam", "wrist_cam"] if ep == 103899 else ["opst_cam"]
        for camera in cameras:
            key = "observation.images." + camera
            vid = info["video_path"].format(episode_chunk=ep//info["chunks_size"],
                episode_index=ep, video_key=key)
            height, width, channels = info["features"][key]["shape"]
            data = subprocess.check_output(["ffmpeg", "-v", "error", "-c:v", "libdav1d",
                "-threads", "1", "-i", str(root/vid), "-f", "rawvideo", "-pix_fmt",
                "rgb24", "-threads", "1", "-"])
            frames = np.frombuffer(data, np.uint8).reshape(-1,height,width,channels)
            assert len(frames) == len(timestamps), (ep, camera, len(frames), len(timestamps))
            video_sha = sha(root/vid)
            for idx in indices:
                name = f"assets/ep{ep:06d}_{camera}_f{idx:03d}.png"
                Image.fromarray(frames[idx]).save(ROOT/name)
                manifest["frames"].append({"file": name, "episode_index": ep,
                    "camera": camera, "frame_index_zero_based": idx,
                    "timestamp_seconds": timestamps[idx], "source_video": vid,
                    "source_video_sha256": video_sha, "frame_sha256": sha(ROOT/name),
                    "size": [width,height]})
    (ROOT/"frame_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Extracted {len(manifest['frames'])} frames from {len(SELECTIONS)} episodes")


if __name__ == "__main__":
    main()
