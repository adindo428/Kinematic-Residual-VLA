"""Conservative split policy: reserve ALL assets used by ANY supplied DOM test."""
from __future__ import annotations

import json
from pathlib import Path

from .config import file_hash, fingerprint
from .storage import atomic_json

OBJECT_CATEGORIES = {"apple", "avocado", "beer", "bottle", "can", "cup", "egg", "kiwi", "lemon",
                     "lime", "onion", "orange", "peach", "potato", "tangerine", "tomato"}


def audit_assets(cfg):
    a = cfg["assets"]
    scenes, objects, benchmark = (Path(a[k]) for k in ("scenes", "objects", "benchmark"))
    manifest_path = benchmark / "test-envs.txt"
    if not manifest_path.is_file():
        raise ValueError("Official test-envs.txt is required to reserve benchmark assets")
    names = [Path(s.strip()).name for s in manifest_path.read_text().splitlines() if s.strip() and not s.startswith("#")]
    files = sorted((benchmark / "tests").glob("*.json"))
    if set(names) != {f.name for f in files} or len(names) != len(set(names)):
        raise ValueError("Test manifest/config inventory mismatch; finish extracting DOM-Test")
    reserved_scenes, reserved_objects, external = set(), set(), set()
    for path in files:
        d = json.loads(path.read_text())
        for value in d["scene"].values():
            if not isinstance(value, dict):
                continue
            usd = (value.get("spawn") or {}).get("usd_path", "")
            if "/scenes/" in usd:
                reserved_scenes.add(Path(usd).name)
            elif "/objects/" in usd:
                reserved_objects.add(usd.split("/objects/", 1)[1])
            elif usd:
                external.add(usd)
    all_scenes = {p.name: str(p.resolve()) for p in scenes.glob("*.usd")}
    all_objects = {str(p.relative_to(objects)): str(p.resolve()) for p in objects.rglob("*.usd")}
    missing = sorted(s for s in reserved_scenes if s not in all_scenes and not (benchmark/"scenes"/s).exists())
    missing += sorted(o for o in reserved_objects if o not in all_objects)
    if missing:
        raise ValueError(f"Benchmark references missing assets: {missing[:10]}")
    eligible_scenes = {k: v for k, v in all_scenes.items() if k not in reserved_scenes}
    eligible_objects = {k: v for k, v in all_objects.items()
                        if k not in reserved_objects and Path(k).parts[0] in OBJECT_CATEGORIES}
    report = dict(policy="exclude-all-benchmark-referenced-assets-and-unseen-v1",
                  benchmark_manifest_sha256=file_hash(manifest_path),
                  benchmark_configs={p.name: file_hash(p) for p in files},
                  reserved_scenes=sorted(reserved_scenes), reserved_objects=sorted(reserved_objects),
                  eligible_scenes=eligible_scenes, eligible_objects=eligible_objects,
                  external_dependencies=sorted(external),
                  counts=dict(scenes=len(all_scenes), objects=len(all_objects), test_configs=len(files),
                              eligible_scenes=len(eligible_scenes), eligible_objects=len(eligible_objects)))
    report["audit_sha256"] = fingerprint(report)
    if not eligible_scenes or not eligible_objects:
        raise ValueError("No training-eligible assets remain under the conservative split policy")
    atomic_json(a["manifest"], report)
    return report


def check_layout(cfg, audit):
    path = Path(cfg["scene"]["layout"])
    if not path.is_file():
        raise ValueError("Run inspect-scene --assemble first to select assets and a table anchor")
    layout = json.loads(path.read_text())
    for key, group in (("scene_usd", "eligible_scenes"), ("object_usd", "eligible_objects")):
        p = str(Path(layout[key]).resolve())
        if p not in audit[group].values():
            raise ValueError(f"Layout asset is not training eligible: {p}")
        if file_hash(p) != layout[key+"_sha256"]:
            raise ValueError(f"Asset changed since layout inspection: {p}")
    if layout.get("audit_sha256") != audit["audit_sha256"]:
        raise ValueError("Asset audit changed; reassemble the layout")
    return layout
