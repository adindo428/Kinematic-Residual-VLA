import importlib.machinery
import sys
import types


def _stub_torchcodec_decoder_import():
    torchcodec = types.ModuleType("torchcodec")
    torchcodec.__path__ = []
    torchcodec.__spec__ = importlib.machinery.ModuleSpec("torchcodec", None)
    decoders = types.ModuleType("torchcodec.decoders")
    decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", None)
    decoders.VideoDecoder = type("VideoDecoder", (), {})
    torchcodec.decoders = decoders
    sys.modules.setdefault("torchcodec", torchcodec)
    sys.modules.setdefault("torchcodec.decoders", decoders)


_stub_torchcodec_decoder_import()

import json
import random

import utils.datasets


def test_episode_split_file_takes_precedence_over_speed_split(tmp_path):
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"train": [3, "4"], "test": [9]}), encoding="utf-8")

    cfg = {
        "NAME": "hzxie/DOM",
        "EPISODE_SPLIT_FILE": str(split_file),
        "TRAIN_EPISODES": [1, 2],
        "TRAIN_EPISODE_RANGE": [1, 512],
        "SPEED_SPLIT": {"ENABLED": True, "TRAIN_MAX_MPS": 0.0},
    }

    assert utils.datasets.resolve_episode_selection(cfg, "train") == [3, 4]
    assert utils.datasets.resolve_episode_selection(cfg, "test") == [9]


def test_episode_split_file_missing_split_fails_clearly(tmp_path):
    split_file = tmp_path / "split.json"
    split_file.write_text(json.dumps({"train": [1]}), encoding="utf-8")

    cfg = {"NAME": "hzxie/DOM", "EPISODE_SPLIT_FILE": str(split_file)}

    try:
        utils.datasets.resolve_episode_selection(cfg, "test")
    except ValueError as exc:
        assert "does not contain split 'test'" in str(exc)
    else:
        raise AssertionError("Expected missing split to raise ValueError")


def test_derived_episode_split_file_samples_and_excludes_deterministically(tmp_path):
    source_file = tmp_path / "source_split.json"
    source_file.write_text(
        json.dumps(
            {
                "splits": {
                    "train": list(range(10)),
                    "test": list(range(20, 32)),
                }
            }
        ),
        encoding="utf-8",
    )
    derived_file = tmp_path / "derived_split.json"
    derived_file.write_text(
        json.dumps(
            {
                "source_file": source_file.name,
                "splits": {
                    "train": {"source_split": "train"},
                    "iid_eval": {"source_split": "train", "sample": 4, "seed": 1234},
                    "ood_val": {"source_split": "test", "sample": 5, "seed": 1234},
                    "ood_test": {"source_split": "test", "exclude_splits": ["ood_val"]},
                    "test": {"source_file": None, "source_split": "ood_test"},
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = {"NAME": "hzxie/DOM", "EPISODE_SPLIT_FILE": str(derived_file)}

    iid_eval = utils.datasets.resolve_episode_selection(cfg, "iid_eval")
    assert iid_eval == utils.datasets.resolve_episode_selection(cfg, "iid_eval")
    assert iid_eval == sorted(random.Random(1234).sample(list(range(10)), 4))

    source_test = set(range(20, 32))
    ood_val = set(utils.datasets.resolve_episode_selection(cfg, "ood_val"))
    ood_test = set(utils.datasets.resolve_episode_selection(cfg, "ood_test"))
    assert ood_val.isdisjoint(ood_test)
    assert ood_val | ood_test == source_test
    assert utils.datasets.resolve_episode_selection(cfg, "test") == sorted(ood_test)
