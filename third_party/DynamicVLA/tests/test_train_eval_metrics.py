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

from types import SimpleNamespace

import torch

from core.train import (
    _empty_eval_totals,
    _evaluate_observed_loader,
    _merge_eval_totals,
    _metrics_from_eval_totals,
)


class _ToyLoader:
    def __init__(self, batch):
        self.batch = batch
        self.dataset = SimpleNamespace(meta=SimpleNamespace(fps=10))

    def __iter__(self):
        yield self.batch

    def __len__(self):
        return 1


class _ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.config = SimpleNamespace(chunk_size=3, max_action_dim=4)
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, batch, noise=None):
        assert noise is not None
        return batch["action"] * 0.5 + self.weight * 0.0


def test_training_observed_eval_uses_no_grad_and_restores_train_mode():
    batch = {
        "action": torch.ones(2, 3, 4),
        "observation.state": torch.zeros(2, 2, 4),
        "dynamic_object.position_xyz": torch.zeros(2, 2, 3),
        "dynamic_object.velocity_xyz": torch.zeros(2, 2, 3),
        "sample_index": torch.tensor([10, 11]),
        "task": ["pick", "pick"],
    }
    cfg = SimpleNamespace(
        DATASET=SimpleNamespace(USE_DELTA_ACTION=True),
        POLICY=SimpleNamespace(CHUNK_SIZE=3, N_ACTION_STEPS=2),
        TRAIN=SimpleNamespace(EVAL_NOISE_SEED=1234),
    )
    policy = _ToyPolicy().train()

    metrics = _evaluate_observed_loader(
        policy,
        _ToyLoader(batch),
        cfg,
        torch.device("cpu"),
        max_batches=1,
        eval_action_steps=2,
    )

    assert policy.training is True
    assert policy.weight.grad is None
    assert policy.reset_calls == 1
    assert metrics["samples"] == 2
    assert metrics["batches"] == 1
    assert metrics["eval_action_steps"] == 2
    assert metrics["action_mse"] == 0.25


def test_eval_metric_totals_match_concatenated_per_rank_samples():
    rank0 = _empty_eval_totals()
    rank0.update(
        samples=2.0,
        batches=1.0,
        eval_action_steps_sum=4.0,
        action_mse_sum=4.0,
        action_mse_sq_sum=10.0,
        ee_tracking_mse_sum=8.0,
        ee_tracking_mse_sq_sum=40.0,
        final_object_distance_count=2.0,
        final_object_distance_sum=3.0,
        final_object_distance_sq_sum=5.0,
        final_success_count=1.0,
    )
    rank1 = _empty_eval_totals()
    rank1.update(
        samples=1.0,
        batches=1.0,
        eval_action_steps_sum=2.0,
        action_mse_sum=5.0,
        action_mse_sq_sum=25.0,
        ee_tracking_mse_sum=7.0,
        ee_tracking_mse_sq_sum=49.0,
        final_object_distance_count=1.0,
        final_object_distance_sum=6.0,
        final_object_distance_sq_sum=36.0,
        final_success_count=0.0,
    )

    merged = _merge_eval_totals(rank0.copy(), rank1)
    metrics = _metrics_from_eval_totals(merged)

    assert metrics["samples"] == 3
    assert metrics["batches"] == 2
    assert metrics["eval_action_steps"] == 2
    assert metrics["action_mse"] == 3.0
    assert metrics["ee_tracking_mse"] == 5.0
    assert metrics["final_object_distance"] == 3.0
    assert metrics["final_ee_object_proximity_rate_5cm"] == 1.0 / 3.0
    assert "final_success_rate" not in metrics
