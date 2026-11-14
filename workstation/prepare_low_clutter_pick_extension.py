"""Prepare (not certify) ten picking cases paired with frozen CR scenes.

This adds a task goal using the upstream picking termination. No scene physics
are edited. Simulator reset and success checks remain required before scoring.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    text = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Refusing to change frozen artifact {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def build(source, generator, output):
    spec = importlib.util.spec_from_file_location('frozen_instructions', generator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = json.loads(source.read_text())
    original = sorted((t for t in protocol['tests'] if t['dimension'] == 'CR'), key=lambda t: t['name'])
    assert len(original) == 10
    tests = []
    for entry in original:
        src = Path(entry['path'])
        assert sha(src) == entry['config_sha256']
        config = json.loads(src.read_text())
        assert config['instruction']['task'] == 'place'
        objects = [k for k, v in config['scene'].items() if k.startswith('object')
                   and isinstance(v, dict) and v.get('class_type', '').endswith(':RigidObject')]
        assert 1 <= len(objects) <= 2 and 'object' in objects
        candidate = copy.deepcopy(config)
        candidate['instruction']['task'] = 'pick'
        placement = candidate['terminations'].pop('objects_placed')
        candidate['terminations']['object_picked'] = {
            'func': 'configs.termination_cfg:is_object_picked',
            'params': {'goal_position': placement['params']['goal_position'],
                       'objects': ['object'], 'tolerance': placement['params']['tolerance']},
            'time_out': False}
        name = 'pick_extension__' + entry['name']
        candidate['evaluation_case_id'] = name
        prompts = []
        for repeat in range(20):
            state = random.getstate()
            random.seed(int(hashlib.sha256(f'low-clutter-pick:42:{name}:{repeat}'.encode()).hexdigest()[:8], 16))
            try:
                prompts.append(module.InstructionGenerator.generate_instruction(candidate['instruction']))
            finally:
                random.setstate(state)
        candidate['evaluation_prompts'] = prompts
        assert candidate['scene'] == config['scene']
        for key in config:
            if key not in {'instruction', 'terminations', 'evaluation_case_id', 'evaluation_prompts'}:
                assert candidate[key] == config[key]
        destination = output / 'configs' / (name + '.json')
        freeze(destination, candidate)
        tests.append({'name': name, 'path': str(destination.relative_to(output)), 'config_sha256': sha(destination),
                      'dimension': 'PICK', 'task': 'pick', 'task_objects': len(objects),
                      'seed': config['seed'], 'horizon_seconds': config['episode_length_s'],
                      'paired_source_case': entry['name'], 'source_config_sha256': sha(src),
                      'modified_scene_physics': False})
    freeze(output / 'protocol.candidate.json', {
        'benchmark': 'Low-clutter picking extension candidates',
        'status': 'static_checks_passed; simulator_validation_required; not_approved_for_scoring',
        'source_protocol_sha256': sha(source), 'instruction_generator_sha256': sha(generator),
        'selection': 'All ten frozen CR cases, independent of policy outcomes.',
        'tests': tests, 'path_base': 'protocol directory', 'repeats': 20, 'candidate_trials': 200,
        'training_task_family': 'pick', 'task_object_cap': 2,
        'success': 'Upstream is_object_picked: grasp proximity and robot-relative goal proximity.',
        'validation_required': ['Asset loading and deterministic resets',
                                'No-action negative control', 'Expert picking positive control',
                                'Terminal success reporting and prompt/target correspondence'],
        'reporting': 'Keep picking extension separate from the existing 90-case Table 1 scores.'})
    lists = output / 'test_lists'
    lists.mkdir(exist_ok=True)
    (lists / 'PICK.txt').write_text(''.join(t['path'] + '\n' for t in tests))
    print(json.dumps({'candidate_cases': len(tests), 'candidate_trials': 200,
                      'simulator_validated': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'generator', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.generator, args.output)
