"""Create isolated, auditable diagnostic copies of the released evaluator."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys

ALIGNED_METHOD = '''    @torch.no_grad()
    def _get_streaming_action(self, batch, noise=None):
        # Diagnostic: select only the command indexed for the current observation.
        current = int(batch["index"])
        self.q_in.update({"obs": (batch, noise)})
        if not self.q_out.empty():
            chunk = self.q_out.get_nowait()
            start = int(chunk["index"])
            actions = chunk["actions"]
            assert actions.size(0) == self.config.n_action_steps
            remaining = [
                {"index": start + i, "action": action}
                for i, action in enumerate(actions)
                if start + i >= current
            ]
            # A fully expired chunk must not erase valid queued commands.
            if remaining:
                self._queues[ACTION].clear()
                self._queues[ACTION].extend(remaining)
        queue = self._queues[ACTION]
        while queue and queue[0]["index"] < current:
            queue.popleft()
        if not queue or queue[0]["index"] != current:
            return None
        action = queue.popleft()
        self._action_index = action["index"]
        return action["action"]

'''

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(base, stage, root, protocol_path=None, scripts_dir=None):
    stage.mkdir(parents=True, exist_ok=False)
    root.mkdir(parents=True, exist_ok=False)
    (stage/'scripts').mkdir()
    (stage/'checkpoint').symlink_to(base/'checkpoint', target_is_directory=True)
    for name in ('prepare.py', 'report.py', 'probe_policy.py'):
        shutil.copy2((scripts_dir or base/'scripts')/name, stage/'scripts'/name)
    for name in ('run.py', 'observer.py'):
        shutil.copy2(Path(__file__).parent/name, stage/'scripts'/name)
    variants = {}
    for label, gripper, aligned in (
        ('original', False, False), ('gripper', True, False),
        ('gripper_aligned', True, True),
    ):
        source = stage/'sources'/label
        shutil.copytree(base/'source', source, ignore=shutil.ignore_patterns('.git', '__pycache__'))
        file = source/'simulations/robots/franka.py'
        text = file.read_text()
        if gripper:
            assert text.count('velocity_limit_sim=0.05,') == 1
            assert text.count('friction=0.2,') == 1
            text = text.replace('velocity_limit_sim=0.05,', 'velocity_limit_sim=0.2,')
            # Known SIF is 5.1; explicit 5.x settings from upstream PR #30.
            text = text.replace('friction=0.2,', 'friction=0.0,\n            dynamic_friction=0.0,\n            viscous_friction=0.0,')
            file.write_text(text)
        model = source/'policies/dynamicvla/modeling_dynamicvla.py'
        if aligned:
            text = model.read_text()
            start = text.index('    @torch.no_grad()\n    def _get_streaming_action(')
            end = text.index('    @torch.no_grad()\n    def _get_non_streaming_action(', start)
            model.write_text(text[:start] + ALIGNED_METHOD + text[end:])
        # Identical instrumentation and policy noise seeds in every arm.
        client = source/'scripts/inference.py'
        text = client.read_text()
        needle = '            vla_model.reset()'
        assert text.count(needle) == 1
        text = text.replace(needle, needle + '\n            setattr(_get_action, "episode", n_tests + 1)')
        needle = '    return action\n\n\ndef _get_transformed_observations'
        assert text.count(needle) == 1
        trace = '''    import json as _json
    with open(os.environ["DOM_CLIENT_TRACE"], "a") as _trace:
        _trace.write(_json.dumps({
            "episode": getattr(_get_action, "episode", 0),
            "observation_index": int(observations["index"]),
            "action_index": int(vla_model._action_index),
            "queue_length": len(vla_model._queues["action"]),
            "wall_time": time.perf_counter(),
        }) + "\\n")
'''
        text = text.replace(needle, trace + needle)
        client.write_text(text)
        model_text = model.read_text()
        seed_needle = '            actions = vla_model._get_action_chunk(batch, noise)'
        assert model_text.count(seed_needle) == 1
        # Stable noise for the same observed index, independent of worker scheduling.
        model_text = model_text.replace(seed_needle,
            '            torch.manual_seed(20261009 + int(index))\n' + seed_needle)
        model.write_text(model_text)
        changed = {}
        for file in source.rglob('*.py'):
            ast.parse(file.read_text())
            relative = file.relative_to(source)
            upstream = base/'source'/relative
            if file.read_bytes() != upstream.read_bytes():
                changed[str(relative)] = {'upstream_sha256': sha(upstream), 'diagnostic_sha256': sha(file)}
        variants[label] = {'gripper_fix': gripper, 'align_actions': aligned,
                           'source': str(source), 'changed_files': changed}
    protocol_path = protocol_path or Path('/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_table1/20261009/protocol.json')
    protocol = json.loads(protocol_path.read_text())
    selected = [
        next(t for t in protocol['tests'] if t['name'].startswith('1-1_place_franka_apple')),
        next(t for t in protocol['tests'] if t['name'].startswith('1-1_place_franka_wbottle')),
        next(t for t in protocol['tests'] if t['name'].startswith('1-2_place_franka_potato')),
    ]
    protocol.update(benchmark='DOM grasp/timing diagnostic, not Table 1 reproduction',
                    expected_trials=6, trials_per_configuration=2, tests=selected,
                    variants=variants, unchanged_dt_scale_sleep=True,
                    policy_noise_seed='20261009 + observation index in every arm',
                    limitations=['Three configurations and two repeats per arm are diagnostic only.',
                                 'Original shared-L40S baseline is rerun with identical telemetry.',
                                 'No continuity regularizer, retraining, target-pose input, or physics timestep change.'])
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (root/'test-envs.txt').write_text('\n'.join(t['path'] for t in selected)+'\n')
    print(json.dumps({'stage':str(stage), 'root':str(root), 'variants':list(variants)}, indent=2))

if __name__ == '__main__':
    prepare(*map(Path, sys.argv[1:]))
