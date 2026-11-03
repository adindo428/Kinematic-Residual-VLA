"""Cached-only selection must retain complete observations and real epoch sizes."""
import pytest

from scripts.prepare_dom100 import coverage_summary, episode_paths, inventory_cached_episodes


def test_cached_selection_excludes_missing_camera_empty_video_and_missing_parquet(tmp_path):
    info = {"chunks_size": 2, "total_episodes": 6,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    metadata = {i: {"length": 10+i} for i in range(6)}
    for ep in range(6):
        for path in episode_paths(tmp_path, info, ep):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"cached")
    episode_paths(tmp_path, info, 1)[1].unlink()
    episode_paths(tmp_path, info, 2)[2].write_bytes(b"")
    episode_paths(tmp_path, info, 4)[0].unlink()
    # A downloader's partial file must never count as a completed recording.
    missing = episode_paths(tmp_path, info, 1)[1]
    missing.with_suffix(".mp4.incomplete").write_bytes(b"partial")
    selected, excluded = inventory_cached_episodes(tmp_path, info, metadata)
    assert selected == [0, 3, 5]
    assert [row["episode_index"] for row in excluded] == [1, 2, 4]
    assert [row["length"] for row in excluded] == [11, 12, 14]
    assert all(len(row["missing_or_empty_files"]) == 1 for row in excluded)
    assert inventory_cached_episodes(tmp_path, info, metadata) == (selected, excluded)
    missing.write_bytes(b"newly downloaded")
    # The frozen selection is unchanged even if a recording arrives later.
    assert selected == [0, 3, 5]
    assert inventory_cached_episodes(tmp_path, info, metadata)[0] == [0, 1, 3, 5]


def test_coverage_uses_actual_frames_and_exposes_nonrandom_exclusions():
    selected = [{"length": 100, "task": "grasp", "speed_bin": 0},
                {"length": 200, "task": "place", "speed_bin": 4}]
    excluded = [{"length": 500, "task": "place", "speed_bin": 4,
                 "missing_or_empty_files": ["overhead", "wrist"]}]
    report = coverage_summary(selected, excluded, 3, 800)
    assert report["retained_episodes"] == 2 and report["retained_frames"] == 300
    assert report["excluded_episodes"] == 1 and report["excluded_frames"] == 500
    assert report["episode_fraction"] == 2/3 and report["frame_fraction"] == 300/800
    assert report["missing_or_empty_files"] == 2
    assert report["task_speed_coverage"]['["place", 4]'] == {"retained": 1, "excluded": 1}
    assert report["task_speed_coverage"]['["grasp", 0]'] == {"retained": 1, "excluded": 0}
    with pytest.raises(ValueError, match="published dataset"):
        coverage_summary(selected, excluded, 4, 800)
    with pytest.raises(ValueError, match="published dataset"):
        coverage_summary(selected, excluded, 3, 1000)
