"""The gated ablation changes centered reconstruction, never learned inputs."""
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter_grasp_gating import install_corrected_prediction, gate_module, update_live_gate


def fake_model(gated):
    tokens = torch.tensor([[11, 22, 33]])
    correction = torch.zeros(1, 20, 7)
    correction[..., :3] = torch.tensor([.2, -.1, .05])
    residual = torch.ones(1, 20, 32)
    calls = []
    def sample(images, masks, received, attention, state):
        assert received is tokens  # Native retokenization must never run.
        calls.append(received)
        return residual
    batch = {'observation.language.tokens': tokens,
             'observation.language.attention_mask': torch.ones_like(tokens, dtype=torch.bool)}
    inner = SimpleNamespace(prepare_images=lambda b: ([], []), prepare_state=lambda b: torch.zeros(1, 32),
                            model=SimpleNamespace(sample_actions=sample))
    model = SimpleNamespace(method='vlash', policy=inner, prepare=lambda b: (batch, correction),
                            affine=lambda x, key, inverse: x*2+3,
                            _evaluation_gate=torch.ones(1, 20, 1))
    install_corrected_prediction(model, gated)
    return model, correction, calls


def test_gated_one_matches_ungated_and_zero_only_removes_centered_xyz():
    ungated, correction, calls = fake_model(False)
    gated, _, _ = fake_model(True)
    full = ungated.predict({})
    torch.testing.assert_close(full, gated.predict({}), rtol=0, atol=0)
    gated._evaluation_gate.zero_()
    zero = gated.predict({})
    torch.testing.assert_close(full-zero, correction)
    torch.testing.assert_close(zero[..., 3:], full[..., 3:], rtol=0, atol=0)
    assert len(calls) == 1


def test_causal_gate_requires_attachment_then_fades_and_resets():
    source = Path(__file__).resolve().parents[2]/'scratch/dom45/training/krv100-full-deltaai-20261106/source/proactive_vla/phase_gate.py'
    module = gate_module(source)
    gate = module.CausalPhaseGate()
    # A closed command by itself is insufficient; without lift it stays active.
    for i in range(15):
        compact = gate.observe(i*.04, [0, 0, .1], [0, 0, .05], [0, 0, 0], -1)
    assert module.gate_from_state(compact, 20, .04)[0, 0] == 1
    # Attached object and EE rise together with stable relative geometry.
    for i in range(15, 35):
        z = (i-14)*.004
        compact = gate.observe(i*.04, [0, 0, .1+z], [0, 0, .05+z], [0, 0, .1], -1)
    assert gate.phase == 'carry'
    assert not module.gate_from_state(compact, 20, .04).any()
    gate.reset()
    assert gate.phase == 'approach' and gate.last_time is None


def test_instruction_only_packet_does_not_require_physical_telemetry():
    front = SimpleNamespace(config=SimpleNamespace(adapter_grasp_gate=True))
    update_live_gate(front, {'task': 'Pick up the cup.'}, None)


def test_identical_kr_weights_share_one_worker_and_retain_variant_gate_flags():
    from resident_adapter_table1 import shared_worker_specs
    def spec(key, digest, gate):
        return {'key': key, 'entry': {'checkpoint_sha256': digest, 'code_root': digest,
            'recipe': digest, 'identity': digest}, 'grasp_gate': gate,
            'corrected_prompt_contract': True, 'share_checkpoint_worker': True}
    original = [spec('baseline', 'b', False), spec('kr', 'k', False), spec('gated', 'k', True)]
    physical, aliases = shared_worker_specs(original)
    assert len(physical) == 2 and aliases == {'baseline': 'baseline', 'kr': 'kr', 'gated': 'kr'}
    assert physical[1]['grasp_gate']
    assert original[1]['grasp_gate'] is False


def test_shared_worker_ungated_request_explicitly_supplies_unit_gate():
    import queue
    from resident_adapter_table1 import Submitter
    front = SimpleNamespace(config=SimpleNamespace(adapter_gate_capable=True, adapter_grasp_gate=False),
                            inbox={}, output=queue.Queue())
    Submitter(front).submit(None, {'index': 7}, 3)
    obs, generation = front.inbox['obs']
    assert generation == 3 and obs['index'] == 7
    assert obs['evaluation.phase_gate'].shape == (1, 1, 20, 1)
    assert obs['evaluation.phase_gate'].eq(1).all()
