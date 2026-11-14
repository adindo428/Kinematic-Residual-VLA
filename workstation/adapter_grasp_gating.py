"""Versioned evaluation-only gating and training-consistent adapter sampling.

These interventions do not change weights or the frozen training recipe.
The gate multiplies the centered physical displacement, not measured velocity.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType


def install_corrected_prediction(model, gated=False):
    if model.method != 'vlash':
        raise ValueError('The corrected prompt intervention is scoped to VLASH')

    def predict(self, batch):
        b, correction = self.prepare(batch)
        p = self.policy
        images, masks = p.prepare_images(b)
        state = p.prepare_state(b)
        # Training supplies these exact prepared tokens to model.forward.
        # Avoid native predict_action_chunk's incompatible retokenization.
        sampled = p.model.sample_actions(images, masks,
            b['observation.language.tokens'], b['observation.language.attention_mask'], state)
        physical_residual = self.affine(sampled[..., :7], 'action', inverse=True)
        if gated:
            gate = self._evaluation_gate
            if gate.shape != (*correction.shape[:2], 1):
                raise ValueError('Gate does not match the generated chunk')
            correction = correction * gate
        return physical_residual + correction

    model.predict = MethodType(predict, model)


def gate_module(path):
    name = 'adapter_frozen_causal_phase_gate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def update_live_gate(front, observation, rotation_vector):
    if not getattr(front.config, 'adapter_grasp_gate', False):
        return
    # Upstream sends the instruction as a separate control packet before the
    # first physical observation. It must reset the tracker, not be scored as
    # an observation with missing telemetry.
    if 'task' in observation and 'observation.state' not in observation:
        return
    import numpy as np
    if any(k not in observation for k in ['timestamp', 'previous_gripper_command', 'observation.environment_state']):
        raise ValueError('Gated inference requires causal simulator telemetry')
    module = gate_module(front.config.adapter_gate_source)
    tracker = getattr(front, '_adapter_gate_tracker', None)
    if tracker is None:
        tracker = module.CausalPhaseGate(module.PhaseGateConfig())
        front._adapter_gate_tracker = tracker
    obj = observation['observation.environment_state']['object']
    ee = observation['observation.state']['end_effector']
    commands = observation['previous_gripper_command']
    rotation = rotation_vector(np.asarray(ee['quat']), 'euler')
    compact = tracker.observe(observation['timestamp'], np.asarray(ee['pos']).reshape(-1, 3)[0],
        np.asarray(obj['pos']).reshape(-1, 3)[0], np.asarray(obj['velocity']).reshape(-1, 3)[0],
        None if commands is None else np.asarray(commands).reshape(-1)[0],
        ee_rotation_xyz=np.asarray(rotation).reshape(-1, 3)[0])
    gate = module.gate_from_state(compact, 20, .04)[None]
    observation['evaluation.phase_gate'] = gate
    if front.log:
        with Path(front.log).with_name('gate_trace.jsonl').open('a') as f:
            f.write(json.dumps({'index': int(observation['index']), 'time_s': float(observation['timestamp']),
                'phase': tracker.phase, 'gate_first': float(gate[0, 0, 0]), 'gate_last': float(gate[0, -1, 0])})+'\n')
