"""Experiment manifests, grouped splits, and explicit six-method identities."""
import hashlib
import json
import math
from pathlib import Path
import random

VARIANTS = {
    "dom100": (False, False, "constant"),
    "kr100": (True, False, "constant"),
    "domv100": (False, True, "constant"),
    "krv100": (True, True, "constant"),
    "krp100": (True, False, "causal_rules"),
    "krvp100": (True, True, "causal_rules"),
}
MICROBATCHES = (2, 4, 8, 16, 32, 44, 88, 176, 352)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    tmp.replace(path)


def source_digest(root):
    root = Path(root)
    files = sorted(p for sub in ("core", "utils", "policies", "proactive_vla", "scripts", "simulations")
                   for p in (root/sub).rglob("*.py"))
    return digest({str(p.relative_to(root)): file_digest(p) for p in files})


def simulator_evaluation_mode(root, recipe=None):
    path = Path(root)/"evaluation_plan.json"
    plan = json.loads(path.read_text()) if path.exists() else {"mode": "during_training"}
    mode = plan["mode"]
    if mode not in ("during_training", "after_training"):
        raise ValueError("Unknown simulator evaluation timing")
    if recipe is not None:
        if recipe.get("simulator_evaluation_mode", "during_training") != mode:
            raise ValueError("Simulator evaluation timing differs from the frozen recipe")
        expected = recipe.get("evaluation_plan_sha256")
        if expected != (file_digest(path) if path.exists() else None):
            raise ValueError("Simulator evaluation plan changed after recipe selection")
    return mode


def grouped_stratified_split(records, seed=42):
    """Group duplicate demos before splitting; balanced 98/1/1 allocation.

    records have episode_index, task, speed_bin, duplicate_id. Mixed-stratum
    duplicate groups use their sorted set of strata as their own stratum.
    """
    groups = {}
    for record in records:
        groups.setdefault(str(record["duplicate_id"]), []).append(record)
    strata = {}
    for key, group in groups.items():
        label = tuple(sorted({(r["task"], r["speed_bin"]) for r in group}))
        strata.setdefault(label, []).append((key, group))
    splits = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for label in sorted(strata):
        entries = sorted(strata[label])
        rng.shuffle(entries)
        # Balance episode counts, not just duplicate-group counts. Whole
        # groups remain indivisible; tiny strata may have no held-out group.
        n = sum(len(g) for _, g in entries)
        targets = {"train": .98*n, "validation": .01*n, "test": .01*n}
        counts = dict.fromkeys(targets, 0)
        entries.sort(key=lambda x: -len(x[1]))
        for _, group in entries:
            split = max(targets, key=lambda k: targets[k]-counts[k])
            splits[split].extend(int(r["episode_index"]) for r in group)
            counts[split] += len(group)
    for split in splits:
        splits[split].sort()
    if len(records) != sum(map(len, splits.values())) or len(set(sum(splits.values(), []))) != len(records):
        raise ValueError("Episodes are duplicated or missing from split")
    return splits


def choose_sampling(pilots, block_speedup):
    indexed = {(r["variant"], float(r["lr"]), r["sampling"]): r for r in pilots}
    block_ok = block_speedup >= 1.10
    for variant in ("dom100", "kr100"):
        frame = indexed.get((variant, 3e-5, "frames"), {})
        block = indexed.get((variant, 3e-5, "blocks"), {})
        if not stable_pilot(frame) or not stable_pilot(block) or block["final_mse"] > 1.05*frame["final_mse"]:
            block_ok = False
    return "blocks" if block_ok else "frames"


def stable_pilot(row):
    return row.get("complete") and row.get("stable", True) and all(
        math.isfinite(float(row[k])) and float(row[k]) > 0 for k in ("initial_mse", "final_mse"))


def select_recipe(pilots, block_speedup):
    """Score completed stable pairs; a diverged candidate cannot win."""
    indexed = {(r["variant"], float(r["lr"]), r["sampling"]): r for r in pilots}
    sampling = choose_sampling(pilots, block_speedup)
    scores = []
    for lr in (1e-5, 3e-5, 1e-4):
        rows = [indexed.get((v, lr, sampling), {}) for v in ("dom100", "kr100")]
        if not all(r.get("complete") for r in rows):
            raise ValueError(f"Missing pilots for {lr}/{sampling}")
        if not all(stable_pilot(r) for r in rows):
            continue
        scores.append((sum(r["final_mse"]/r["initial_mse"] for r in rows)/2, lr))
    if not scores:
        raise ValueError("No learning rate was stable on both architectures")
    score, lr = min(scores)
    return {"lr": lr, "sampling": sampling, "effective_batch": 352, "epochs": 100,
            "seed": 42, "pilot_score": score, "block_speedup": block_speedup}
