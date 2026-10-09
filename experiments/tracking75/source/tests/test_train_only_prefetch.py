import importlib
import sys
import types
from types import SimpleNamespace


def _import_run_with_fake_nvml(monkeypatch):
    fake_pynvml = types.SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda idx: idx,
        nvmlDeviceGetName=lambda handle: b"fake-gpu",
        nvmlDeviceGetCpuAffinity=lambda handle, elements: [0],
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    sys.modules.pop("utils.distributed", None)
    sys.modules.pop("core", None)
    sys.modules.pop("run", None)
    return importlib.import_module("run")


def test_prefetch_only_skips_test_split_when_train_eval_disabled(monkeypatch):
    run = _import_run_with_fake_nvml(monkeypatch)
    resolved = []
    prefetched = []

    def fake_resolve(dataset_cfg, split):
        resolved.append(split)
        return [1, 2] if split == "train" else [9]

    def fake_prefetch(repo_id, episodes, required_features):
        prefetched.append((repo_id, episodes, required_features))

    monkeypatch.setattr(run.utils.datasets, "resolve_episode_selection", fake_resolve)
    monkeypatch.setattr(run.utils.datasets, "ensure_lerobot_episodes_cached", fake_prefetch)

    cfg = SimpleNamespace(
        DATASET=SimpleNamespace(NAME="hzxie/DOM", REQUIRED_FEATURES=["action"]),
        TRAIN={"EVAL_TEST_DURING_TRAIN": False},
    )

    run.prefetch_configured_dataset(cfg)

    assert resolved == ["train"]
    assert prefetched == [("hzxie/DOM", [1, 2], ["action"])]
