import importlib.util
from pathlib import Path

import numpy as np
import pytest

_path = Path(__file__).parents[1] / "integrations/dynamicvla/evaluate_checkpoint.py"
_spec = importlib.util.spec_from_file_location("checkpoint_eval", _path)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)


def test_checkpoint_temporal_and_feature_contract():
    config = dict(type="dynamicvla", n_obs_steps=3, chunk_size=4, use_delta_action=True,
                  input_features={"observation.state": {"shape": [6]}},
                  output_features={"action": {"shape": [7]}},
                  delta_timestamps={"observation": [-8, -4, 0]})
    assert evaluation.offsets_from_config(config) == ([-8, -4, 0], [0, 1, 2, 3])
    config["delta_timestamps"]["observation"] = [-4, 0, 4]
    with pytest.raises(ValueError, match="current frame"):
        evaluation.offsets_from_config(config)
    config["delta_timestamps"]["observation"] = [-8, -4, 0]
    config["output_features"]["action"]["shape"] = [8]
    with pytest.raises(ValueError, match="seven-value"):
        evaluation.offsets_from_config(config)


def test_geodesic_rotation_and_padding_do_not_inflate_errors():
    target = np.zeros((2, 7))
    prediction = target.copy()
    prediction[0, 3] = 2 * np.pi
    prediction[1, :3] = 100
    xyz, angle, gripper = evaluation.errors(prediction, target, np.zeros(6), False, [True, False])
    np.testing.assert_allclose([xyz, angle, gripper], 0, atol=1e-14)
    prediction[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluation.errors(prediction, target, np.zeros(6), False, [True, False])


@pytest.mark.parametrize("delta", [False, True])
def test_evaluator_uses_observations_without_label_leakage(delta):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import json

    target = torch.tensor([[.4, .1, .5, 0, 0, 0, 1.]]).repeat(3, 1)
    state = target[0, :6].repeat(2, 1)
    prediction = target.clone()
    if delta:
        prediction[:, :6] -= state[-1]
    prediction[2, :3] = 1000  # Padded step must not contribute.
    row = {"action": target, "observation.state": state, "episode_index": torch.tensor(6),
           "action_is_pad": torch.tensor([False, False, True]),
           "task": json.dumps({"task": "track", "instruction": "Hover over the apple."})}

    class Policy:
        resets = 0

        def reset(self):
            self.resets += 1

        def predict_action_chunk(self, batch):
            assert set(batch) == {"task", "observation.state"}
            assert batch["task"] == ["Hover over the apple."]
            assert batch["observation.state"].shape == (1, 2, 6)
            return prediction.unsqueeze(0)

    policy = Policy()
    result = evaluation.evaluate(policy, [row, row],
                                 {"input_features": {"observation.state": {"shape": [6]}},
                                  "use_delta_action": delta}, "cpu")
    assert policy.resets == result["anchors"] == 2
    assert result["valid_action_steps"] == 4
    assert result["position_label_rmse_m"] == 0
    assert result["gripper_label_rmse"] == 0
