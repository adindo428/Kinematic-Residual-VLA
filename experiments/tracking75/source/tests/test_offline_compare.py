import torch

from scripts.proactive_residual_offline_compare import (
    _merge_metrics,
    _sample_noise_for_batch,
)


def test_shared_noise_is_deterministic_per_sample_index():
    batch_a = {
        "action": torch.zeros(2, 5, 7),
        "sample_index": torch.tensor([10, 11]),
    }
    batch_b = {
        "action": torch.zeros(2, 5, 7),
        "sample_index": torch.tensor([11, 10]),
    }

    noise_a = _sample_noise_for_batch(
        batch_a,
        chunk_size=5,
        action_dim=7,
        device=torch.device("cpu"),
        seed=1234,
    )
    noise_b = _sample_noise_for_batch(
        batch_b,
        chunk_size=5,
        action_dim=7,
        device=torch.device("cpu"),
        seed=1234,
    )

    assert torch.allclose(noise_a[0], noise_b[1])
    assert torch.allclose(noise_a[1], noise_b[0])


def test_merge_metrics_handles_empty_all_valid_anchor_result():
    merged = _merge_metrics(
        [{"n": 0.0, "eval_action_steps_sum": 0.0, "full_chunk_steps_sum": 0.0}],
        [0.05],
    )

    assert merged["samples"] == 0
    assert merged["success_rate"] == 0.0
    assert merged["success_rate_by_threshold_m"]["0.050m"] == 0.0
