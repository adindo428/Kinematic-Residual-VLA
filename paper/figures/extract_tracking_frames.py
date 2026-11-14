#!/usr/bin/env python3
"""Extract two authentic expert tracking frames and calibrated annotations.

The pinned episode was downloaded from adi2440/dom-franka-tracking. This
helper is offline: it verifies retained source hashes, decodes on the CPU,
and projects recorded points with the recorded camera calibration. It does
not run a simulation or alter scene content. Requires NumPy, Pillow,
PyArrow, and FFmpeg with libdav1d.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

FIGURES = Path(__file__).resolve().parent
REPO = FIGURES.parents[1]
DEFAULT_SOURCE = REPO / "scratch/dom45/results/recovered-tracking-controller-b5549db/tracking_demo"
REVISION = "5c46a9f26b5f85edafa53885187cc03fa8706219"
IDS = (0, 150)
CAMERA = "opst_cam"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(q):
    """Quaternion w,x,y,z to a local-to-parent rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def extract(source):
    download = json.loads((source / "download_manifest.json").read_text())
    assert download["dataset"] == "adi2440/dom-franka-tracking"
    assert download["revision"] == REVISION
    checksums = json.loads((source / "checksums.json").read_text())
    for item in download["files"]:
        actual = sha256(source / item["path"])
        assert actual == item["sha256"], item["path"]
        if item["path"] in checksums:
            assert actual == checksums[item["path"]], item["path"]
    info = json.loads((source / "meta/info.json").read_text())
    collection = json.loads((source / "meta/collection.json").read_text())
    split = json.loads((source / "meta/split_assignments.json").read_text())
    assert 8 in split["test"]
    assert info["fps"] == 25
    layout = collection["provenance"]["layout"]
    base_rotation = rotation(layout["robot_quaternion"])
    base_translation = np.asarray(layout["robot_position"])
    frames = pq.read_table(source / "diagnostics/episode_000008/frames.parquet").to_pylist()
    controls = pq.read_table(source / "diagnostics/episode_000008/controls.parquet").to_pylist()
    labels = pq.read_table(source / "data/chunk-000/episode_000008.parquet")
    video_relative = f"videos/chunk-000/observation.images.{CAMERA}/episode_000008.mp4"
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-c:v", "libdav1d", "-threads", "1",
        "-i", str(source / video_relative), "-f", "rawvideo", "-pix_fmt",
        "rgb24", "-threads", "1", "-",
    ])
    frame_bytes = 480 * 360 * 3
    assert len(raw) % frame_bytes == 0
    assert len(raw) // frame_bytes == len(labels) == 250
    # Diagnostics retain the five additional frames needed for future-pose labels.
    assert len(frames) == 255 and len(controls) == 2040

    def project(position, camera):
        world = base_rotation @ np.asarray(position) + base_translation
        local = rotation(camera["quaternion_world_ros"]).T @ (
            world - np.asarray(camera["position_world"]))
        assert local[2] > 0
        homogeneous = np.asarray(camera["intrinsic"]) @ local
        return (homogeneous[:2] / homogeneous[2]).tolist()

    extracted = []
    for idx in IDS:
        record = frames[idx]
        control = controls[record["control_step"]]
        assert record["timestamp"] == control["timestamp"] == idx / 25
        camera = record["cameras"][CAMERA]
        target = np.asarray(control["target"]["position"])
        path = FIGURES / f"assets/tracking_ep000008_{CAMERA}_f{idx:03d}.png"
        Image.frombytes("RGB", (480, 360), raw[idx*frame_bytes:(idx+1)*frame_bytes]).save(path)
        extracted.append({
            "file": str(path.relative_to(FIGURES)),
            "frame_sha256": sha256(path),
            "episode_index": 8,
            "frame_index_zero_based": idx,
            "timestamp_seconds": record["timestamp"],
            "camera": CAMERA,
            "camera_calibration": camera,
            "projection": {
                "apple_center_px": project(record["object_position"], camera),
                "apple_top_px": project(target - [0, 0, 0.1], camera),
                "hover_target_px": project(target, camera),
                "end_effector_px": project(record["ee_position"], camera),
            },
            "object_velocity_mps": record["object_velocity"],
            "hover_target_path_pixels": [
                project(controls[row["control_step"]]["target"]["position"], camera)
                for row in frames[:250]
            ],
        })
    try:
        retained_source = str(source.relative_to(REPO))
    except ValueError:
        retained_source = str(source)
    manifest = {
        "dataset": download["dataset"],
        "revision": REVISION,
        "episode_index": 8,
        "split": "test",
        "fps": 25,
        "resolution_pixels": [480, 360],
        "source_root": retained_source,
        "source_files": download["files"],
        "source_video": video_relative,
        "description": "Rule-based expert hover-tracking demonstration; not a DOM or KR policy rollout and not an image from the CR/DA/LS evaluation.",
        "hover_target": "10 cm above the apple's visual top; recorded target position from control telemetry.",
        "projection": "Robot-base to world using collection layout; inverse recorded ROS camera pose; recorded pinhole intrinsics. All pixels refer to full uncropped 480x360 image.",
        "image_processing": "Lossless PNG output of decoded AV1 video; no scene alteration. Figure crops and vector overlays are applied separately.",
        "annotation_semantics": "Hover target is a recorded reference point. apple_center_px is the projected object origin, not a segmented centroid; apple_top_px is the calibrated visual-top reference. A projected circle is a marker, not a metrically projected 3 cm tolerance boundary. The full reference trajectory describes this expert episode only.",
        "robot_position_world": layout["robot_position"],
        "robot_quaternion_world": layout["robot_quaternion"],
        "frames": extracted,
    }
    (FIGURES / "tracking_frame_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Verified source hashes; extracted expert episode 8 at 0 and 6 seconds with calibrated projections.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    extract(parser.parse_args().source_root.resolve())
