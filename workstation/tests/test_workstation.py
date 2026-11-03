import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location('workstation_'+name, HERE/(name+'.py'))
    item = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(item)
    return item


def test_preflight_failure_is_durable_and_returns_failure(tmp_path, monkeypatch):
    check = module('preflight')
    def fail():
        raise RuntimeError('CUDA wheel cannot execute this architecture')
    monkeypatch.setattr(check, 'run_checks', fail)
    output = tmp_path/'checks/policy.json'
    monkeypatch.setattr(sys, 'argv', ['preflight', '--output', str(output)])
    with pytest.raises(SystemExit) as error:
        check.main()
    assert error.value.code == 1
    result = json.loads(output.read_text())
    assert result['validated'] is False and 'architecture' in result['error']


def frozen_fixture(tmp_path):
    diagnostic = module('diagnostic')
    stage = tmp_path/'stage'; (stage/'checkpoint').mkdir(parents=True)
    weights = stage/'checkpoint/model.safetensors'; weights.write_bytes(b'weight-fixture')
    config = stage/'checkpoint/config.json'; config.write_text('{}')
    source = stage/'sources/original/model.py'; source.parent.mkdir(parents=True); source.write_text('value = 1\n')
    protocol = {'checkpoint_weights_sha256': diagnostic.sha256(weights),
                'checkpoint_config_sha256': diagnostic.sha256(config),
                'source_hashes': {str(source.relative_to(stage)): diagnostic.sha256(source)},
                'driver_hashes': {}, 'asset_sha256': {}, 'tests': []}
    return diagnostic, SimpleNamespace(stage=stage), protocol, source


def test_changed_diagnostic_source_is_rejected_before_launch(tmp_path):
    diagnostic, args, protocol, source = frozen_fixture(tmp_path)
    diagnostic.validate(args, protocol)
    source.write_text('value = 2\n')
    with pytest.raises(ValueError, match='source changed'):
        diagnostic.validate(args, protocol)


def test_changed_checkpoint_config_is_rejected(tmp_path):
    diagnostic, args, protocol, source = frozen_fixture(tmp_path)
    (args.stage/'checkpoint/config.json').write_text('{"chunk_size": 10}')
    with pytest.raises(ValueError, match='configuration changed'):
        diagnostic.validate(args, protocol)


def test_dom_wrapper_preserves_spaces_and_extra_arguments(tmp_path):
    spy = tmp_path/'policy interpreter'
    spy.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    spy.chmod(0o755)
    env = dict(os.environ, KRVLA_REPO=str(HERE.parent), POLICY_PYTHON=str(spy),
               STAGE=str(tmp_path/'staged source'), ASSETS=str(tmp_path/'all assets'),
               RESULTS=str(tmp_path/'result folder'), ISAACLAB=str(tmp_path/'sim launcher'))
    output = subprocess.check_output(['bash', str(HERE/'dom.sh'), 'full', '--dimension', 'CR', '--dry-run'], env=env, text=True)
    argv = json.loads(output)
    assert argv[argv.index('--stage')+1] == env['STAGE']
    assert argv[argv.index('--root')+1] == env['RESULTS']
    assert argv[argv.index('--policy-python')+1] == str(spy)
    assert argv[-3:] == ['--dimension', 'CR', '--dry-run']


def test_sim_wrapper_selects_environment_without_shell_interpolation(tmp_path):
    conda = tmp_path/'conda'
    conda.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    conda.chmod(0o755)
    env = dict(os.environ, PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],
               ISAACLAB_ROOT=str(tmp_path/'Isaac Lab'), ISAACLAB_ENV='env_sim')
    output = subprocess.check_output(['bash', str(HERE/'isaaclab.sh'), '-p', 'a file.py', '--headless'], env=env, text=True)
    argv = json.loads(output)
    assert argv == ['run', '--no-capture-output', '-n', 'env_sim', 'bash',
                    env['ISAACLAB_ROOT']+'/isaaclab.sh', '-p', 'a file.py', '--headless']


@pytest.mark.parametrize('contact_sweep', [False, True])
def test_portable_preparation_uses_local_protocol_and_preserves_baseline(tmp_path, monkeypatch, contact_sweep):
    diagnostic = module('diagnostic')
    base = tmp_path/'official'; source = base/'source'
    for directory in ('simulations/robots', 'policies/dynamicvla', 'scripts'):
        (source/directory).mkdir(parents=True, exist_ok=True)
    franka = source/'simulations/robots/franka.py'
    franka.write_text('cfg = dict(velocity_limit_sim=0.05, friction=0.2,)\n')
    inference = source/'scripts/inference.py'
    inference.write_text('def reset():\n    if True:\n            vla_model.reset()\n\ndef _get_action():\n    return action\n\n\ndef _get_transformed_observations():\n    pass\n')
    model_file = source/'policies/dynamicvla/modeling_dynamicvla.py'
    model_file.write_text('class Model:\n    @torch.no_grad()\n    def _get_streaming_action(self, batch, noise=None):\n        pass\n\n    @torch.no_grad()\n    def _get_non_streaming_action(self, batch, noise=None):\n        pass\n\ndef worker():\n    if True:\n            actions = vla_model._get_action_chunk(batch, noise)\n')
    (base/'checkpoint').mkdir()
    weight = base/'checkpoint/model.safetensors'; weight.write_bytes(b'fixture')
    config = base/'checkpoint/config.json'; config.write_text('{}')
    names = ['1-1_place_franka_apple_fixture', '1-1_place_franka_wbottle_fixture', '1-2_place_franka_potato_fixture']
    official_root = tmp_path/'official results'; official_root.mkdir()
    (official_root/'protocol.json').write_text(json.dumps({
        'checkpoint': str(base/'checkpoint'), 'checkpoint_weights_sha256': diagnostic.sha256(weight),
        'checkpoint_config_sha256': diagnostic.sha256(config), 'asset_sha256': {},
        'tests': [{'name': name, 'path': str(tmp_path/(name+'.json'))} for name in names]}))
    before = {str(p): p.read_bytes() for p in source.rglob('*.py')}
    monkeypatch.setattr(diagnostic.local, 'WEIGHTS_SHA256', diagnostic.sha256(weight))
    monkeypatch.setattr(diagnostic.subprocess, 'check_output', lambda argv, **kw:
                        diagnostic.local.SOURCE_COMMIT+'\n' if 'rev-parse' in argv else '')
    args = SimpleNamespace(base=base, stage=tmp_path/'diagnostic', root=tmp_path/'diagnostic results', official_root=official_root)
    if contact_sweep:
        args.finger_friction = [1.0, 2.0]
    diagnostic.prepare(args)
    assert all(Path(p).read_bytes() == value for p, value in before.items())
    protocol = json.loads((args.root/'protocol.json').read_text())
    assert len(protocol['tests']) == 3 and len(protocol['variants']) == (5 if contact_sweep else 3)
    if contact_sweep:
        assert protocol['variants']['contact_mu2']['finger_contact_coefficient'] == 2.0
        assert 'apply_contact_friction' in (args.stage/'scripts/observer.py').read_text()
    assert (args.root/'test-envs.txt').read_text().splitlines() == [str(tmp_path/(n+'.json')) for n in names]
    assert Path(protocol['checkpoint']).resolve() == (base/'checkpoint').resolve()
    for relative, expected in protocol['source_hashes'].items():
        assert diagnostic.sha256(args.stage/relative) == expected
    with pytest.raises(FileExistsError):
        diagnostic.prepare(args)
