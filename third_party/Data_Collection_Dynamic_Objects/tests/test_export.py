import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from dom_tracking.config import file_hash, load_config
from dom_tracking.export import export_dom
from dom_tracking.storage import VideoWriter, atomic_json, read_records, video_summary, write_records
from dom_tracking.validate import validate_dataset


@pytest.fixture
def tiny_run(tmp_path):
    run = tmp_path/"run"
    ep = run/"episodes/episode_000000"
    ep.mkdir(parents=True)
    cfg = load_config()
    cfg["collection"].update(purpose="smoke", episodes=1, seconds=.4)
    atomic_json(run/"provenance.json", dict(config=cfg, layout={"object_usd": "apple00.usd", "robot_position": [0,0,0], "robot_quaternion": [1,0,0,0]}))
    atomic_json(run/"report.json", dict(complete=True, balanced=False))
    atomic_json(ep/"episode.json", dict(frame_count=10, metrics={"rejection_reasons": []}))
    frames = []
    for i in range(15):
        frames.append(dict(timestamp=i/25, frame_index=i, control_step=8*i, ee_position=[.4+i*.001, 0, .3], ee_quaternion=[0, 1, 0, 0],
                           object_position=[.4, 0, .1], object_quaternion=[1, 0, 0, 0], object_velocity=[.01, 0, 0],
                           cameras={name: dict(timestamp=i/25, sensor_frame=i+1, intrinsic=[[240,0,240],[0,240,180],[0,0,1]],
                                               position_world=[0,0,1], quaternion_world_ros=[1,0,0,0])
                                    for name in ("wrist_cam", "opst_cam", "side_cam")}))
        frames[-1]["cameras"]["wrist_cam"]["position_world"] = [.465+i*.001, 0, .4034]
        frames[-1]["cameras"]["wrist_cam"]["quaternion_world_ros"] = [0, 2**-.5, -2**-.5, 0]
    write_records(ep/"frames", frames, "parquet")
    write_records(ep/"controls", [{"timestamp": 0., "command": {"mode": 0}}], "jsonl")
    for name in ("wrist_cam", "opst_cam", "side_cam"):
        writer = VideoWriter(ep/f"{name}.mp4", 25, 480, 360, "libsvtav1")
        try:
            for i in range(10):
                image = np.full((360, 480, 3), i*20, np.uint8)
                image[100:200, 100+i*2:200+i*2, 0] = 255
                writer.add(image)
        finally:
            writer.close()
    return run


def test_export_roundtrip_labels_videos_and_corruption(tiny_run, tmp_path):
    destination = tmp_path/"dataset"
    result = export_dom(tiny_run, destination, allow_smoke=True)
    assert result["frames"] == 10
    assert validate_dataset(destination)["valid"]
    table = pq.read_table(destination/"data/chunk-000/episode_000000.parquet")
    assert table["action"][0].as_py()[0] == pytest.approx(.405)
    assert table["action"][-1].as_py()[0] == pytest.approx(.414)
    assert table["observation.state"][0].as_py()[0] == pytest.approx(.4)
    with pytest.raises(ValueError, match="destination"):
        export_dom(tiny_run, destination, allow_smoke=True)
    (destination/"meta/tasks.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        validate_dataset(destination)


def test_smoke_export_requires_explicit_flag(tiny_run, tmp_path):
    with pytest.raises(ValueError, match="Smoke"):
        export_dom(tiny_run, tmp_path/"dataset")


def test_stale_wrist_extrinsics_rejected_even_with_valid_checksums(tiny_run, tmp_path):
    destination = tmp_path/"dataset"
    export_dom(tiny_run, destination, allow_smoke=True)
    path = destination/"diagnostics/episode_000000/frames.parquet"
    frames = read_records(path)
    # Simulate a cached wrist pose while the tool continues moving.
    frames[10]["cameras"]["wrist_cam"] = frames[0]["cameras"]["wrist_cam"].copy()
    frames[10]["cameras"]["wrist_cam"].update(timestamp=10/25, sensor_frame=11)
    write_records(path.with_suffix(""), frames, "parquet")
    manifest = json.loads((destination/"checksums.json").read_text())
    manifest[str(path.relative_to(destination))] = file_hash(path)
    atomic_json(destination/"checksums.json", manifest)
    with pytest.raises(ValueError, match="Wrist"):
        validate_dataset(destination)


def test_video_statistics_match_decoded_pixels(tiny_run):
    import av
    path = tiny_run/"episodes/episode_000000/wrist_cam.mp4"
    result = video_summary(path, 10)
    with av.open(str(path)) as container:
        pixels = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]).astype(float)/255
    for key, function in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
        expected = function(pixels, axis=(0, 1, 2))
        np.testing.assert_allclose(np.asarray(result["stats"][key]).reshape(3), expected, atol=1e-10)
    assert result["stats"]["count"] == [10]
