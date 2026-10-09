import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/hub_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("hub_checkpoints", SCRIPT)
hub_checkpoints = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub_checkpoints)


def make_checkpoint(path, epoch):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps({"epoch": epoch}))
    (path / "model.safetensors").write_bytes(b"model")
    (path / "training_state.pt").write_bytes(b"state")


def test_upload_creates_private_repo_and_publishes_marker_last(tmp_path, monkeypatch):
    make_checkpoint(tmp_path, 3)
    calls = []

    class FakeApi:
        def __init__(self, token):
            calls.append(("token", token))

        def create_repo(self, *args, **kwargs):
            calls.append(("create", args, kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("folder", kwargs))

        def upload_file(self, **kwargs):
            calls.append(("marker", kwargs))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setenv("HF_TOKEN", "test-token")
    hub_checkpoints.upload("example/dom-vla-training", "baseline", tmp_path)

    assert [call[0] for call in calls] == ["token", "create", "folder", "marker"]
    assert calls[1][2]["private"] is True
    assert calls[2][1]["path_in_repo"] == "baseline/epoch-0003"
    assert calls[3][1]["path_in_repo"] == "baseline/latest.json"
    assert json.loads(calls[3][1]["path_or_fileobj"]) == {"epoch": 3}


def test_restore_keeps_newer_local_checkpoint(tmp_path, monkeypatch):
    make_checkpoint(tmp_path, 4)
    marker = tmp_path / "remote-latest.json"
    marker.write_text('{"epoch": 3}')

    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, **kwargs):
            return {}

    def forbidden_snapshot(**kwargs):
        raise AssertionError("Must not download an older checkpoint")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            HfApi=FakeApi,
            hf_hub_download=lambda **kwargs: str(marker),
            snapshot_download=forbidden_snapshot,
        ),
    )
    hub_checkpoints.restore("example/dom-vla-training", "residual", tmp_path)
    assert hub_checkpoints.checkpoint_epoch(tmp_path) == 4


def test_restore_rejects_partial_local_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "model.safetensors").write_bytes(b"partial")
    marker = tmp_path / "remote-latest.json"
    marker.write_text('{"epoch": 3}')
    snapshot = tmp_path / "snapshot"
    make_checkpoint(snapshot / "baseline/epoch-0003", 3)

    class FakeApi:
        def __init__(self, token):
            pass

        def repo_info(self, **kwargs):
            return {}

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            HfApi=FakeApi,
            hf_hub_download=lambda **kwargs: str(marker),
            snapshot_download=lambda **kwargs: str(snapshot),
        ),
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        hub_checkpoints.restore("example/dom-vla-training", "baseline", tmp_path)
