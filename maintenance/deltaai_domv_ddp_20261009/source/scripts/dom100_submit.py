#!/usr/bin/env python3
"""Verify prerequisites and submit six independent production allocations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.dom_checkpoint import run_lock
from core.dom_training import load_family
from scripts.dom_closed_loop import validate_integration
from utils.dom_experiment import VARIANTS, atomic_json, digest, simulator_evaluation_mode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-root", type=Path, required=True)
    parser.add_argument("--gpu", choices=("a100", "h100", "h200"), default="a100")
    parser.add_argument("--constraint", choices=("gpu_a100_40gb", "gpu_a100_80gb"), help="Optional Palmetto A100 VRAM feature")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); root = args.family_root.resolve()
    manifest = load_family(root)
    recipe = json.loads((root/"recipe.json").read_text()); expected = recipe.pop("fingerprint")
    if digest(recipe) != expected or recipe["manifest"] != manifest["fingerprint"]:
        raise ValueError("Shared production recipe differs from the frozen family")
    if simulator_evaluation_mode(root, recipe) == "during_training":
        validate_integration(root)
    approved = []
    for path in (root/"profiles").glob("*.json"):
        profile = json.loads(path.read_text())
        if args.gpu.upper() not in profile["hardware"]["name"].upper() or profile["manifest"] != manifest["fingerprint"]:
            continue
        if args.constraint and args.gpu == "a100":
            requested = 40 if args.constraint == "gpu_a100_40gb" else 80
            if (40 if profile["hardware"]["total_memory"] < 60*1024**3 else 80) != requested:
                continue
        check = dict(profile); expected_profile = check.pop("fingerprint")
        if digest(check) != expected_profile:
            raise ValueError("GPU profile was modified")
        if profile["runtime"]["precision"] != recipe["precision"] or not profile.get("all_variants_verified"):
            continue
        if any(not (root/"smoke"/f"{variant}-{profile['gpu_key']}.json").is_file() for variant in VARIANTS):
            continue
        for variant in VARIANTS:
            smoke = json.loads((root/"smoke"/f"{variant}-{profile['gpu_key']}.json").read_text())
            if not smoke.get("complete") or smoke["manifest"] != manifest["fingerprint"] or smoke["profile"] != expected_profile:
                raise ValueError("Every variant needs a decoded backward/reload test on the requested GPU class")
        approved.append(profile)
    if not approved:
        raise ValueError("Requested GPU class has no approved profile at the shared precision")
    if args.gpu == "a100":
        memory_classes = {40 if p["hardware"]["total_memory"] < 60*1024**3 else 80 for p in approved}
        if args.constraint:
            requested = 40 if args.constraint == "gpu_a100_40gb" else 80
            if requested not in memory_classes:
                raise ValueError("The requested A100 memory class has not been profiled")
        elif len(memory_classes) == 1:
            args.constraint = f"gpu_a100_{next(iter(memory_classes))}gb"
    elif args.constraint:
        raise ValueError("A100 memory constraints cannot select H100/H200 nodes")
    project = Path(__file__).resolve().parents[1]
    commands = []
    for idx, variant in enumerate(VARIANTS, 1):
        command = ["sbatch", "--parsable", "--chdir", str(project), "--export", "ALL", f"--gpus={args.gpu}:1"]
        if args.constraint:
            command.append("--constraint="+args.constraint)
        command.append(str(project/"slurm"/f"{idx:02d}_{variant}.sbatch"))
        commands.append((variant, command))
    if args.dry_run:
        print(json.dumps({"commands": [c for _, c in commands], "dependencies": []}, indent=2)); return
    # Persist each successful submission before proceeding, so a partial
    # submission failure cannot silently create duplicate writers on retry.
    with run_lock(root/"submissions"):
        path = root/"submissions/production.json"
        jobs = json.loads(path.read_text()) if path.exists() else {}
        for variant, command in commands:
            if variant in jobs:
                continue
            jobs[variant] = subprocess.check_output(command, text=True).strip().split(";")[0]
            atomic_json(path, jobs)
        print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()
