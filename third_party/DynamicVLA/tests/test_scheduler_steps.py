import importlib.machinery
import sys
import types


def test_scheduler_steps_match_optimizer_updates():
    torchcodec = types.ModuleType("torchcodec")
    torchcodec.__path__ = []
    torchcodec.__spec__ = importlib.machinery.ModuleSpec("torchcodec", None)
    decoders = types.ModuleType("torchcodec.decoders")
    decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", None)
    decoders.VideoDecoder = type("VideoDecoder", (), {})
    torchcodec.decoders = decoders
    sys.modules.setdefault("torchcodec", torchcodec)
    sys.modules.setdefault("torchcodec.decoders", decoders)

    from core.train import _accumulation_group_size, _scheduler_training_steps

    assert _scheduler_training_steps(10, 3, 4, "optimizer") == 9
    assert _scheduler_training_steps(10, 3, 4, "legacy") == 30
    assert [_accumulation_group_size(batch, 10, 4) for batch in range(10)] == [
        4, 4, 4, 4, 4, 4, 4, 4, 2, 2
    ]
