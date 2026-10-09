#!/usr/bin/env python3
"""Portable six-trial grasp diagnostics using the same three Palmetto variants."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO/'evaluation/official_dom/scripts'
sys.path.insert(0, str(SCRIPTS))
import local
from prepare import sha256, write_json
from report import verify_folder
from run_suite import stop_group


def load_diagnostic(name):
    path = REPO/'evaluation/official_dom/diagnostics/palmetto_20261009'/f'{name}.py'
    spec = importlib.util.spec_from_file_location('diagnostic_'+name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    # This uses a previously validated official protocol; never modifies its source.
    protocol = json.loads((args.official_root/'protocol.json').read_text())
    if Path(protocol['checkpoint']).resolve() != (args.base/'checkpoint').resolve():
        raise ValueError('Official protocol and baseline checkpoint do not match')
    if sha256(args.base/'checkpoint/model.safetensors') != local.WEIGHTS_SHA256:
        raise ValueError('Released checkpoint hash mismatch')
    source = args.base/'source'
    if subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != local.SOURCE_COMMIT:
        raise ValueError('Wrong upstream source revision')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Baseline source has edits; use a clean pinned source')
    if args.stage.exists() or args.root.exists():
        raise FileExistsError('Choose new diagnostic stage and result paths')
    load_diagnostic('stage').prepare(args.base, args.stage, args.root,
                                     args.official_root/'protocol.json', SCRIPTS)
    protocol = json.loads((args.root/'protocol.json').read_text())
    protocol['checkpoint'] = str(args.stage/'checkpoint')
    protocol['gpu_label'] = 'Local workstation shared GPU diagnostic'
    protocol['limitations'] = ['Three configurations, two repeats per variant; diagnostic only.',
                               'Sim 5.1.0 / Lab 2.3.2 differ from the paper Sim 4.5 / Lab 2.2.1.',
                               'Simulation and policy share one visible GPU.',
                               'Observation-index noise seeding and telemetry are shared across all variants.']
    protocol['source_hashes'] = {
        str(path.relative_to(args.stage)): sha256(path)
        for path in (args.stage/'sources').rglob('*') if path.is_file()
    }
    protocol['driver_hashes'] = {
        str(path.relative_to(args.stage)): sha256(path)
        for path in (args.stage/'scripts').glob('*.py')
    }
    write_json(args.root/'protocol.json', protocol)


def validate(args, protocol):
    if sha256(args.stage/'checkpoint/model.safetensors') != protocol['checkpoint_weights_sha256']:
        raise ValueError('Diagnostic checkpoint changed')
    if sha256(args.stage/'checkpoint/config.json') != protocol['checkpoint_config_sha256']:
        raise ValueError('Diagnostic checkpoint configuration changed')
    for field in ('source_hashes', 'driver_hashes'):
        for path, expected in protocol[field].items():
            if sha256(args.stage/path) != expected:
                raise ValueError(f'Diagnostic source changed: {path}')
    for path, expected in protocol['asset_sha256'].items():
        if sha256(path) != expected:
            raise ValueError(f'Benchmark asset changed: {path}')
    for test in protocol['tests']:
        if sha256(test['path']) != test['config_sha256']:
            raise ValueError('Test JSON changed')


def run(args):
    protocol = json.loads((args.root/'protocol.json').read_text())
    validate(args, protocol)
    if not args.isaaclab.is_file() or not os.access(args.isaaclab, os.X_OK):
        raise ValueError('Executable simulator launcher required')
    for label in args.labels:
        folder = args.root/label
        if folder.exists():
            raise FileExistsError(f'Preserve existing evidence and use a new root: {folder}')
        source = args.stage/'sources'/label
        server, client = local.commands(args, args.root/'test-envs.txt', folder, 2)
        server[2] = str(args.stage/'scripts/observer.py')
        client[2] = str(source/'scripts/inference.py')
        environment = dict(os.environ, DOM_OFFICIAL_SOURCE=str(source),
                           DOM_TRIAL_RESULTS=str(folder/'trial_results.jsonl'),
                           DOM_CLIENT_TRACE=str(folder/'client_trace.jsonl'), TOKENIZERS_PARALLELISM='false')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                       '--format=csv,noheader'], text=True)
        folder.mkdir()
        write_json(folder/'commands.json', {'server': server, 'client': client})
        write_json(folder/'runtime.json', {'gpu': gpu, 'cuda_visible_devices': environment.get('CUDA_VISIBLE_DEVICES'),
                                          'shared_gpu': True, 'variant': label})
        simulation = policy = None
        try:
            subprocess.run([str(args.policy_python), str(SCRIPTS/'probe_policy.py'), '--source', str(source),
                            '--checkpoint', str(args.stage/'checkpoint'), '--output', str(folder/'policy_probe.json')],
                           cwd=source, env=environment, check=True, timeout=1200)
            with (folder/'server.log').open('w') as slog, (folder/'client.log').open('w') as clog:
                simulation = subprocess.Popen(server, cwd=source, env=environment, stdout=slog,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                policy = subprocess.Popen(client, cwd=source, env=environment, stdout=clog,
                                          stderr=subprocess.STDOUT, start_new_session=True)
                progress = time.monotonic(); previous = -1
                while policy.poll() is None:
                    if simulation.poll() is not None:
                        raise RuntimeError('Simulator exited early; inspect server.log')
                    path = folder/'trial_results.jsonl'
                    size = path.stat().st_size if path.exists() else 0
                    if size != previous:
                        progress = time.monotonic(); previous = size
                    if time.monotonic()-progress > 1200:
                        raise RuntimeError('No completed trial for 20 minutes')
                    time.sleep(5)
                if policy.returncode:
                    raise RuntimeError('Policy failed; inspect client.log')
            verify_folder(folder, args.root, args.root/'test-envs.txt', 2)
        except BaseException as error:
            write_json(folder/'failure.json', {'error': str(error)})
            raise
        finally:
            stop_group(policy); stop_group(simulation)
        load_diagnostic('run').summarize(args.root, args.labels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prepare_parser = sub.add_parser('prepare')
    for name in ('base', 'official-root', 'stage', 'root'):
        prepare_parser.add_argument('--'+name, type=Path, required=True)
    run_parser = sub.add_parser('run')
    for name in ('stage', 'root', 'assets', 'isaaclab', 'policy-python'):
        run_parser.add_argument('--'+name, type=Path, required=True)
    run_parser.add_argument('--labels', nargs='+', choices=['original', 'gripper', 'gripper_aligned'],
                            default=['original', 'gripper', 'gripper_aligned'])
    run_parser.add_argument('--port', type=int, default=28000)
    args = p.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    if args.command == 'prepare':
        prepare(args)
    else:
        if not 1024 <= args.port <= 65534:
            p.error('--port must be 1024..65534')
        run(args)

if __name__ == '__main__':
    main()
