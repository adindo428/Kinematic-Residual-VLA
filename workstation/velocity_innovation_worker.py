"""Native streaming worker with per-observation noise and chunk-origin metadata."""
import ast
import inspect
from pathlib import Path
import textwrap
from types import MethodType

_source = Path(__file__).with_name('resident_table1_worker.py').read_text()
_source = _source.replace('        dummy.update(policy.extract_residual_batch_fields(dummy))',
    "        if hasattr(policy, 'extract_residual_batch_fields'): dummy.update(policy.extract_residual_batch_fields(dummy))")
_source = _source.replace("cfg.residual_phase_gate_mode=='causal_rules'", "getattr(cfg, 'residual_phase_gate_mode', 'constant')=='causal_rules'")
_needle = "                index=int(batch['index']);state=batch['observation.state'][:,-1:,:].clone()"
assert _source.count(_needle) == 1
_source = _source.replace(_needle, _needle+'''
                episode_seed = batch.pop('_episode_seed', None)
                if noise is None and episode_seed is not None:
                    generator = torch.Generator(device='cuda').manual_seed(int(episode_seed)*1000003+index)
                    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device='cuda', generator=generator)
''')
exec(compile(_source, str(Path(__file__).with_name('resident_table1_worker.py')), 'exec'))

_BasePool = ResidentPool


def streaming_with_origin(method, namespace):
    source = textwrap.dedent(inspect.getsource(method))
    edits = {
        '        skip_n_actions = batch["index"] - actions["index"]':
            '        origin = int(actions["index"])\n        skip_n_actions = batch["index"] - actions["index"]',
        '{"index": actions["index"] + i, "action": a}':
            '{"index": actions["index"] + i, "action": a, "origin": origin}',
        '        self._action_index = action["index"]':
            '        self._action_index = action["index"]\n        self.chunk_origin_index = action["origin"]',
        '    self.q_in.update({"obs": (batch, noise)})':
            '    batch = dict(batch, _episode_seed=self._episode_seed)\n    self.q_in.update({"obs": (batch, noise)})',
    }
    for old, new in edits.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    local = dict(namespace)
    exec(compile(source, '<streaming_chunk_origin>', 'exec'), local)
    return local['_get_streaming_action']


class ResidentPool(_BasePool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
        module = __import__(DynamicVLAPolicy.__module__, fromlist=['*'])
        method = streaming_with_origin(DynamicVLAPolicy._get_streaming_action, vars(module))
        for front in self.fronts.values():
            front._episode_seed = 0
            front.chunk_origin_index = None
            front.select_action = MethodType(method, front)
