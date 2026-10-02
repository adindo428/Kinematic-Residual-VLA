"""Check the pinned DynamicVLA reader against a local tracking export; no training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent


def check_sources(upstream):
    manifest = json.loads((HERE / "upstream.json").read_text())
    commit = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != manifest["commit"]:
        raise ValueError("Upstream commit differs from the audited revision")
    hashes = {}
    for name, original in manifest["files"].items():
        actual = hashlib.sha256((upstream / name).read_bytes()).hexdigest()
        allowed = {original}
        if name == "utils/instruction_generator.py":
            allowed = {manifest["patched_instruction_sha256"]}
        elif name == "utils/memcached.py":
            allowed.add(manifest["patched_memcached_sha256"])
        if actual not in allowed:
            raise ValueError(f"Unexpected source or missing tracking patch: {name}")
        hashes[name] = actual
    return commit, hashes


def check_instructions(upstream, dataset):
    spec = importlib.util.spec_from_file_location(
        "audited_instructions", upstream / "utils/instruction_generator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generate = module.InstructionGenerator.generate_instruction
    tasks = [json.loads(line) for line in (dataset / "meta/tasks.jsonl").read_text().splitlines()]
    instructions = {}
    for task in tasks:
        metadata = json.loads(task["task"])
        if metadata["task"] != "track":
            raise ValueError("This check expects a tracking export")
        prompt = generate(task["task"])
        assert prompt == metadata["instruction"].strip()
        instructions[task["task_index"]] = prompt
    # Regression checks: existing tasks still work and invalid tracking labels fail.
    for name in ("pick", "place", "long-horizon"):
        prompt = generate({"task": name, "objects": ["apple"], "containers": ["bowl"]})
        assert "apple" in prompt
        if name != "pick":
            assert "bowl" in prompt
    for value in (None, "", "  ", 3):
        try:
            generate({"task": "track", "instruction": value, "containers": []})
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid tracking instruction was accepted")
    return instructions


def check_dataset(upstream, dataset, instructions):
    import importlib.metadata
    import numpy as np
    import pyarrow.parquet as pq
    import torch

    if importlib.metadata.version("lerobot") != "0.3.3":
        raise ValueError("Use the isolated LeRobot 0.3.3 reader environment")
    sys.path.insert(0, str(upstream))
    from utils.datasets import LeRobotDataset

    splits = json.loads((dataset / "meta/split_assignments.json").read_text())
    info = json.loads((dataset / "meta/info.json").read_text())
    selected = [episode for key in ("train", "validation", "test") for episode in splits[key]]
    assert len(set(selected)) == len(selected) == info["total_episodes"]
    assert set(selected) == set(range(info["total_episodes"]))
    camera_keys = [key for key in info["features"] if key.startswith("observation.images.")]
    checked = []
    for split in ("train", "validation", "test"):
        for delta in (False, True):
            reader = LeRobotDataset(
                "local/dom-tracking", root=dataset, episodes=splits[split],
                required_features=list(info["features"]), pin_memory=False,
                delta_action=delta,
                delta_timestamps={"observation": [-2, -1, 0], "action": [0, 1, 2, 3, 4]},
            )
            assert reader.episodes == splits[split]
            episode = splits[split][0]
            table = pq.read_table(dataset / reader.meta.get_data_file_path(episode))
            for frame in (0, len(table) // 2, len(table) - 1):
                item = reader[frame]
                expected = np.asarray([
                    table["action"][min(frame + offset, len(table) - 1)].as_py()
                    for offset in range(5)
                ], dtype=np.float32)
                if delta:
                    expected[:, :6] -= np.asarray(table["observation.state"][frame].as_py(), dtype=np.float32)
                np.testing.assert_allclose(item["action"].numpy(), expected, atol=1e-6)
                assert item["observation.state"].shape == (3, 6)
                assert item["observation.environment_state"].shape == (3, 9)
                assert item["task"] == instructions[table["task_index"][frame].as_py()]
                for key in camera_keys:
                    image = item[key]
                    assert image.shape == (3, 3, 360, 480), (key, image.shape)
                    assert torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
                checked.append({"split": split, "episode": episode, "frame": frame, "delta_action": delta})
    return {"samples": checked, "versions": {
        name: importlib.metadata.version(name)
        for name in ("lerobot", "datasets", "torch", "torchvision", "torchcodec")
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--instructions-only", action="store_true")
    args = parser.parse_args()
    upstream, dataset, report = args.upstream.resolve(), args.dataset.resolve(), args.report.resolve()
    if report.is_relative_to(dataset):
        raise ValueError("Keep reports outside the immutable dataset")
    commit, hashes = check_sources(upstream)
    instructions = check_instructions(upstream, dataset)
    result = {"upstream_commit": commit, "source_sha256": hashes,
              "dataset": str(dataset), "instructions": instructions,
              "checksums_sha256": hashlib.sha256((dataset / "checksums.json").read_bytes()).hexdigest(),
              "full_loader_checked": not args.instructions_only}
    if not args.instructions_only:
        result.update(check_dataset(upstream, dataset, instructions))
    result["valid"] = True
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
