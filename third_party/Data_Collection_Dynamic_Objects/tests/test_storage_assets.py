import json
from pathlib import Path

import pytest

from dom_tracking.assets import audit_assets
from dom_tracking.storage import RunStore, atomic_json


def test_recovery_skips_attempt_and_preserves_commit(tmp_path):
    store = RunStore(tmp_path, {"seed": 42})
    attempt, folder = store.begin()
    store.commit(folder, {"attempt": attempt}, True)
    store.begin()  # emulate interruption
    store.close()
    resumed = RunStore(tmp_path, {"seed": 42})
    assert len(resumed.completed()) == 1
    assert len(list((tmp_path/"interrupted").iterdir())) == 1
    assert resumed.begin()[0] == 2
    resumed.close()
    with pytest.raises(ValueError, match="Resume refused"):
        RunStore(tmp_path, {"seed": 43})


def test_asset_audit_reserves_all_benchmark_assets(tmp_path):
    scenes, objects, tests = tmp_path/"scenes", tmp_path/"objects", tmp_path/"benchmark"
    for p in [scenes, objects/"apple", objects/"unseen", tests/"tests"]:
        p.mkdir(parents=True)
    for p in [scenes/"held.usd", scenes/"train.usd", objects/"apple/held.usd", objects/"apple/train.usd", objects/"unseen/extra.usd"]:
        p.write_text("fixture")
    (tests/"test-envs.txt").write_text("../tests/case.json\n")
    atomic_json(tests/"tests/case.json", {"scene": {"house": {"spawn": {"usd_path": "./scenes/held.usd"}},
                                                     "object": {"spawn": {"usd_path": "./objects/apple/held.usd"}}}})
    cfg = {"assets": dict(scenes=str(scenes), objects=str(objects), benchmark=str(tests), manifest=str(tmp_path/"audit.json"))}
    report = audit_assets(cfg)
    assert list(report["eligible_scenes"]) == ["train.usd"]
    assert list(report["eligible_objects"]) == ["apple/train.usd"]
    (tests/"test-envs.txt").write_text("../tests/missing.json\n")
    with pytest.raises(ValueError, match="mismatch"):
        audit_assets(cfg)
