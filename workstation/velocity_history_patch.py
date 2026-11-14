"""Install explicit, checkpoint-compatible observation-clip velocity conditioning."""
from pathlib import Path


def patch_source(source):
    source = Path(source)
    config = source/'policies/dynamicvla/configuration_dynamicvla.py'
    text = config.read_text()
    anchor = '    condition_on_object_velocity: bool = False\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '''    # Old checkpoints retain current-only conditioning. History uses the same
    # causal timestamps as the observation clip, newest first in state padding.
    velocity_conditioning_mode: str = "current"
''')
    anchor = '    def validate_residual_formulation(self) -> None:\n'
    text = text.replace(anchor, anchor + '''        if self.velocity_conditioning_mode not in {"current", "history"}:
            raise ValueError("velocity_conditioning_mode must be current or history")
        if self.condition_on_object_velocity and self.velocity_conditioning_mode == "history":
            if self.n_obs_steps < 1 or self.max_state_dim < 6 + 3 * self.n_obs_steps:
                raise ValueError("State padding cannot hold the observation velocity clip")
''')
    config.write_text(text)
    model = source/'policies/dynamicvla/modeling_dynamicvla.py'
    text = model.read_text()
    anchor = '            velocity = torch.as_tensor(value, device=state.device, dtype=state.dtype)\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '''            mode = getattr(self.config, "velocity_conditioning_mode", "current")
            if mode not in {"current", "history"}:
                raise ValueError("Unknown velocity conditioning mode")
            if mode == "history":
                expected = (state.shape[0], self.config.n_obs_steps, 3)
                if velocity.shape != expected or not torch.isfinite(velocity).all():
                    raise ValueError("Velocity history must match the finite observation clip [batch, n_obs_steps, 3]")
                if batch[OBS_STATE].ndim != 3 or batch[OBS_STATE].shape[1] != self.config.n_obs_steps:
                    raise ValueError("Robot and velocity observation clips must align")
                if self.config.max_state_dim < 6 + 3 * self.config.n_obs_steps:
                    raise ValueError("State padding cannot hold the observation velocity clip")
                mean = state.new_tensor(self.config.velocity_input_mean)
                std = state.new_tensor(self.config.velocity_input_std)
                if mean.shape != (3,) or std.shape != (3,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                    raise ValueError("Invalid velocity normalization")
                # Keep the existing current-velocity slots 6:9; append older
                # samples in reverse chronological order, normalized per axis.
                history = ((velocity - mean) / std).flip(1).reshape(state.shape[0], -1)
                return pad_vector(torch.cat((state, history), dim=-1), self.config.max_state_dim)
''')
    model.write_text(text)
    training = source/'core/dom_training.py'
    text = training.read_text()
    anchor = '        "condition_on_object_velocity": velocity, "residual_lambda_type": "constant",\n'
    assert text.count(anchor) == 1
    text = text.replace(anchor, anchor + '''        "velocity_conditioning_mode": manifest.get("velocity_conditioning_mode", "current"),
''')
    training.write_text(text)


if __name__ == '__main__':
    import sys
    patch_source(sys.argv[1])
