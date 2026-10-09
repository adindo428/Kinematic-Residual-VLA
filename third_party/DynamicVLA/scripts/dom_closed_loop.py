#!/usr/bin/env python3
"""Bind held-out DOM episodes to their simulator scenes, replay, and evaluate."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dom_experiment import VARIANTS, atomic_json, digest, file_digest
from utils.dom_sim_metrics import summarize_cases

PROJECT = Path(__file__).resolve().parents[1]


def scene_asset_paths(assets, scene):
    """Use the same scene/object layout as the existing simulator loader."""
    assets = Path(assets)
    paths = []
    for name, node in scene.items():
        if not isinstance(node, dict) or "spawn" not in node or "usd_path" not in node["spawn"]:
            continue
        usd = Path(node["spawn"]["usd_path"])
        if name == "house":
            paths.append(assets/"scenes"/usd.name)
        elif node.get("class_type") == "isaaclab.assets.rigid_object.rigid_object:RigidObject":
            paths.append(assets/"objects"/usd.parent.name/usd.name)
    return paths


def validate_cases(root):
    from core.dom_training import load_family
    root = Path(root)
    binding = json.loads((root/"simulator/binding.json").read_text())
    check = dict(binding); expected = check.pop("fingerprint")
    if digest(check) != expected or binding["manifest"] != load_family(root)["fingerprint"]:
        raise ValueError("Simulator case binding differs from this frozen family")
    for name, sha in binding["files"].items():
        if file_digest(name) != sha:
            raise ValueError(f"Simulator asset/configuration changed: {name}")
    for split, count in (("validation", 64), ("test", 256)):
        cases = json.loads((root/f"simulator/{split}.json").read_text())
        if [r["episode_index"] for r in cases] != json.loads((root/f"{split}_cases.json").read_text()) or len(cases) != count:
            raise ValueError("Simulator suite does not match the reserved episode split")
    return binding


def validate_integration(root):
    binding = validate_cases(root)
    report = json.loads((Path(root)/"simulator/integration.json").read_text())
    if report.get("binding") != binding["fingerprint"] or not report.get("complete"):
        raise ValueError("Real demonstration/controller replay must pass before production")
    if not report.get("rotation_roundtrip") or not report.get("decoded_controller_smoke"):
        raise ValueError("Simulator integration verification is incomplete")
    return report


def bind(root, assets, case_map=None):
    """Never substitute unrelated benchmark scenes for reserved DOM episodes."""
    import pyarrow.parquet as pq
    from core.dom_training import load_family
    from scripts.prepare_dom100 import balanced_sample
    root, assets = Path(root), Path(assets).resolve()
    manifest = load_family(root)
    destination = root/"simulator"
    if (destination/"binding.json").exists():
        return validate_cases(root)
    destination.mkdir(exist_ok=True)
    mapping = json.loads(Path(case_map).read_text()) if case_map else {}
    # Released scene bundles may retain the dataset episode identity. If they
    # do not, an explicit exported episode->scene map is required.
    if not mapping:
        for path in sorted((assets/"tests").rglob("*.json")):
            cfg = json.loads(path.read_text())
            idx = cfg.get("episode_index", cfg.get("dom_episode_index"))
            if idx is not None:
                if str(idx) in mapping:
                    raise ValueError(f"Ambiguous simulator configuration for episode {idx}")
                mapping[str(idx)] = str(path)
    selected = {split: json.loads((root/f"{split}_cases.json").read_text()) for split in ("validation", "test")}
    pilot = set(json.loads((root/"pilot_episodes.json").read_text()))
    records = json.loads((root/"episodes.json").read_text())
    selected["replay"] = balanced_sample([r for r in records if r["episode_index"] in pilot], 6)
    required = set(sum(selected.values(), []))
    missing = sorted(required-{int(k) for k in mapping})
    if missing:
        raise ValueError(f"No exact episode-to-scene mapping for {len(missing)} reserved/replay episodes; first IDs: {missing[:10]}. Supply --case-map exported from the DOM scene bundle.")
    info = json.loads((Path(manifest["dataset_root"])/"meta/info.json").read_text())
    files = {}
    for split, indices in selected.items():
        rows = []
        for ep in indices:
            path = Path(mapping[str(ep)])
            if not path.is_absolute():
                path = assets/path
            path = path.resolve()
            cfg = json.loads(path.read_text())
            if cfg.get("episode_index", cfg.get("dom_episode_index", ep)) != ep:
                raise ValueError("Mapped configuration has a different episode identity")
            cfg["seed"] = int(cfg["seed"])
            files[str(path)] = file_digest(path)
            for asset in scene_asset_paths(assets, cfg["scene"]):
                files[str(asset)] = file_digest(asset)
            row = {"episode_index": ep, "env_cfg": str(path), "phase_config": manifest["phase_config"]}
            if split == "replay":
                parquet = Path(manifest["dataset_root"])/info["data_path"].format(episode_chunk=ep//info["chunks_size"], episode_index=ep)
                table = pq.read_table(parquet, columns=["action", "observation.state"])
                replay = destination/f"replay-{ep:06d}.npz"
                np.savez_compressed(replay, actions=np.asarray(table["action"].to_pylist(), dtype=np.float32),
                    states=np.asarray(table["observation.state"].to_pylist(), dtype=np.float32))
                row["replay"] = str(replay)
                files[str(replay)] = file_digest(replay)
            rows.append(row)
        atomic_json(destination/f"{split}.json", rows)
        files[str(destination/f"{split}.json")] = file_digest(destination/f"{split}.json")
    sif = Path(os.environ.get("SIF_FILE", "/scratch/aparame/isaaclab.sif")).resolve()
    files[str(sif)] = file_digest(sif)
    binding = {"manifest": manifest["fingerprint"], "assets": str(assets), "sif": str(sif), "files": files,
        "label_timing": {"generation_lookahead_frames": 5, "fps": 25, "analytic_action_grid_offset": 0},
        "collision_definition": "robot links 1-7 and hand/fingers above 1 N, excluding target contact on hand/fingers",
        "replay": selected["replay"]}
    binding["fingerprint"] = digest(binding)
    atomic_json(destination/"binding.json", binding)
    return binding


def server_command(root, binding, suite, output, ports, replay=False):
    assets = Path(binding["assets"])
    deps = Path(os.environ.get("ISAAC_DEPS", "/scratch/aparame/Dynamic_VLA/isaac_eval_deps/python"))
    sim_data = Path(os.environ.get("ISAAC_SIM_DATA", "/scratch/aparame/isaac-sim"))
    for path in (deps, assets/"scenes", assets/"objects"):
        if not path.exists():
            raise FileNotFoundError(path)
    mounts = [PROJECT, Path(root).resolve(), assets, deps]
    command = ["apptainer", "exec", "--cleanenv", "--nv"]
    for path in dict.fromkeys(mounts):
        command += ["-B", f"{path}:{path}"]
    for source, target in (("kit_data", "/isaac-sim/kit/data"), ("cache/kit", "/isaac-sim/kit/cache"), ("tmp", "/tmp")):
        path = sim_data/source; path.mkdir(parents=True, exist_ok=True)
        command += ["-B", f"{path}:{target}"]
    rows = json.loads(Path(suite).read_text())
    command += ["--env", f"PYTHONPATH={deps}:{PROJECT}", "--env", "OMNI_KIT_ACCEPT_EULA=YES", "--env", "TERM=xterm",
        "--pwd", str(assets), binding["sif"], "/opt/IsaacLab/isaaclab.sh", "-p", str(PROJECT/"simulations/evaluate.py"),
        "--headless", "--enable_cameras", "--device", "cuda:0", "--num_envs", "1", "--scene_dir", str(assets/"scenes"),
        "--object_dir", str(assets/"objects"), "--env_cfg", rows[0]["env_cfg"], "--output_dir", str(output),
        "--dom-instrument", "--replay-manifest" if replay else "--dom-case-manifest", str(suite),
        "-n", "1", "--img_port", str(ports[0]), "--act_port", str(ports[1])]
    return command


def free_ports():
    sockets = [socket.socket() for _ in range(2)]
    try:
        for sock in sockets:
            sock.bind(("127.0.0.1", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def run_suite(root, checkpoint, suite, output, binding):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    ports = free_ports()
    with (output/"server.log").open("w") as log, (output/"client.log").open("w") as client_log:
        server = subprocess.Popen(server_command(root, binding, suite, output, ports), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            client = subprocess.Popen([sys.executable, str(PROJECT/"scripts/inference.py"), "-p", str(checkpoint),
                "-a", "dom100-eval", "-r", "euler", "-d", "--img_port", str(ports[0]), "--act_port", str(ports[1]),
                "-o", str(output/"inference")], stdout=client_log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic()+23*3600
            try:
                while client.poll() is None:
                    if server.poll() is not None:
                        # Client needs a short interval to consume the final suite message.
                        try:
                            client.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            raise RuntimeError("Simulator exited before the client finished")
                    if time.monotonic() > deadline:
                        raise TimeoutError("Manipulation suite exceeded its allocation budget")
                    time.sleep(2)
                if client.returncode != 0 or server.wait(timeout=60) != 0:
                    raise RuntimeError("Manipulation client/server failed; inspect their logs")
            finally:
                if client.poll() is None:
                    os.killpg(client.pid, 15); client.wait(timeout=30)
        finally:
            if server.poll() is None:
                os.killpg(server.pid, 15); server.wait(timeout=30)
    rows = json.loads((output/"case_results.json").read_text())
    expected = json.loads(Path(suite).read_text())
    if [r["env_cfg"] for r in rows] != [r["env_cfg"] for r in expected]:
        raise ValueError("Simulator skipped or reordered a reserved case")
    return {"complete": True, "summary": summarize_cases(rows), "cases": rows,
            "binding": binding["fingerprint"], "checkpoint_sha256": file_digest(Path(checkpoint)/"model.safetensors")}


def integrate(root):
    from core.dom_training import build_policy, load_family
    from utils.helpers import get_quaternion, get_rotation_vector
    root = Path(root); binding = validate_cases(root)
    output = root/"simulator/integration"
    output.mkdir(exist_ok=True)
    with (output/"replay.log").open("w") as log:
        subprocess.run(server_command(root, binding, root/"simulator/replay.json", output, free_ports(), replay=True),
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=6*3600)
    rows = json.loads((output/"replay_results.json").read_text())
    if len(rows) != len(binding["replay"]) or any(not r["completion"] or r["replay_position_rmse_m"] > .05 for r in rows):
        raise ValueError("Demonstration replay did not complete or differed by more than 5 cm RMS")
    angles = np.array([[3.4, -.2, 5.9], [.2, .3, .4]], np.float32)
    quaternions = get_quaternion(angles, "euler")
    back = get_quaternion(get_rotation_vector(quaternions, "euler"), "euler")
    if not np.allclose(np.abs(np.sum(quaternions*back, axis=-1)), 1, atol=1e-5):
        raise ValueError("Euler/wxyz rotation roundtrip failed")
    checkpoint = output/"controller"
    policy = build_policy(root, "krvp100", device="cpu")
    policy.save_pretrained(str(checkpoint)); del policy
    cfg = json.loads((checkpoint/"config.json").read_text())
    cfg.update(delta_timestamps={"observation": [-2, 0]}, use_delta_action=True)
    atomic_json(checkpoint/"config.json", cfg)
    # One training replay scene exercises decoded observations, live oracle
    # velocity, applied-command gating, frames, and the actual controller.
    smoke_suite = output/"controller-case.json"
    atomic_json(smoke_suite, json.loads((root/"simulator/replay.json").read_text())[:1])
    controller = run_suite(root, checkpoint, smoke_suite, output/"controller-result", binding)
    report = {"complete": True, "binding": binding["fingerprint"], "replays": rows,
        "manifest": load_family(root)["fingerprint"], "rotation_roundtrip": True,
        "decoded_controller_smoke": controller["complete"], "controller_outcome": controller["summary"]}
    atomic_json(root/"simulator/integration.json", report)
    import shutil
    shutil.rmtree(checkpoint)
    return report


def dispatch(root, request):
    from core.dom_checkpoint import run_lock
    root, request = Path(root), Path(request)
    with run_lock(root/"evaluation_dispatch"):
        item = json.loads(request.read_text())
        if item.get("slurm_job"):
            return item["slurm_job"]
        validate_integration(root)
        job = subprocess.check_output(["sbatch", "--parsable", "--export", "ALL",
            str(PROJECT/"slurm/evaluate_dom100.sbatch"), str(root), str(request)], text=True).strip().split(";")[0]
        item["slurm_job"] = job
        atomic_json(request, item)
        return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "integration", "evaluate", "dispatch", "complete"))
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, default=Path("/scratch/aparame/Dynamic_VLA/dom_benchmark"))
    parser.add_argument("--case-map", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args(); root = args.family_root.resolve()
    if args.command == "bind":
        result = bind(root, args.asset_root, args.case_map)
    elif args.command == "integration":
        result = integrate(root)
    elif args.command == "dispatch":
        requests = [args.request] if args.request else sorted((root/"runs").glob("*/evaluation-*.json"))
        result = [dispatch(root, path) for path in requests]
    elif args.command == "evaluate":
        item = json.loads(args.request.read_text())
        binding = validate_cases(root)
        output = args.request.parent/"closed_loop"/args.request.stem
        from core.dom_checkpoint import run_lock
        with run_lock(output):
            result = run_suite(root, item["checkpoint"], root/f"simulator/{item['split']}.json", output, binding)
            result.update(variant=item["variant"], epoch=item["epoch"], split=item["split"])
            atomic_json(output/"results.json", result)
    else:
        result = {}
        for variant in VARIANTS:
            run = root/"runs"/variant
            training = json.loads((run/"verified_training.json").read_text())
            rows = {p.parent.name: json.loads(p.read_text()) for p in (run/"closed_loop").glob("*/results.json")}
            required = {f"evaluation-epoch-{ep:03d}" for ep in (1, 5, *range(10, 101, 10))}|{"evaluation-test-best", "evaluation-test-final"}
            if not training.get("complete") or not required <= rows.keys() or any(not rows[k].get("complete") for k in required):
                raise ValueError(f"Training or manipulation evaluations remain incomplete for {variant}")
            result[variant] = {"training_samples_per_second": training["samples_per_second"],
                "evaluations": {name: rows[name]["summary"] for name in sorted(required)}}
        atomic_json(root/"comparison.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
