import copy

import numpy as np
import pytest

from dom_tracking.config import load_config
from dom_tracking.metrics import episode_metrics


def example():
    controls = [dict(timestamp=i/200, command=dict(position_error=[.001, 0, 0], mode=0,
                                                 orientation_error=0.01, saturated=False),
                     contact_force=0., joint_limit_violation=False) for i in range(400)]
    frames = [dict(mode=0, object_velocity=[0., .01, 0.], cameras={name: {"target_in_frame": True}
              for name in ("wrist_cam", "opst_cam", "side_cam")}) for _ in range(50)]
    return controls, frames


def test_quality_rejects_collision_and_out_of_frame():
    controls, frames = example()
    cfg = load_config()
    result = episode_metrics(controls, frames, cfg)
    assert not result["rejection_reasons"]
    assert result["rmse"] == pytest.approx(.001)
    assert result["mode_counts"] == [0, 50, 0]
    controls[-1]["contact_force"] = 2.
    for f in frames[:5]:
        f["cameras"]["wrist_cam"]["target_in_frame"] = False
    result = episode_metrics(controls, frames, cfg)
    assert set(result["rejection_reasons"]) == {"collision", "target_out_of_frame"}


def test_nonfinite_errors_are_not_accepted():
    controls, frames = example()
    controls[0]["command"]["position_error"] = [np.nan, 0, 0]
    with pytest.raises(ValueError, match="nonfinite"):
        episode_metrics(controls, frames, load_config())
