"""Build a training-only deployment without modifying either Palmetto source."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / 'bundle'
if BUNDLE.exists():
    shutil.rmtree(BUNDLE)
BUNDLE.mkdir(exist_ok=True)
changes = []

def replace(path, old, new):
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f'Expected source fragment missing from {path}: {old[:90]}')
    path.write_text(text.replace(old, new))
    changes.append({'path': str(path.relative_to(BUNDLE)), 'reason': new.strip()[:160]})

for label in ('tracking', 'dom100'):
    target = BUNDLE / label
    if target.exists():
        raise RuntimeError('Bundle already exists; use a fresh transfer directory')
    shutil.copytree(HERE / 'staging' / label, target)
    ref = target / 'reference_palmetto'
    ref.mkdir()
    if (target / 'slurm').exists():
        shutil.move(str(target / 'slurm'), str(ref / 'slurm'))
    for name in ('dom100_submit.py', 'submit_dom100.sh', 'train_paired_velocity_pair.sh'):
        if (target / 'scripts' / name).exists():
            (target / 'scripts' / name).unlink()
            changes.append({'path': f'{label}/scripts/{name}', 'reason': 'Omitted legacy submission/evaluation coupling; use direct training entry points.'})
    for path in ref.rglob('*.sbatch'):
        if 'dom_closed_loop' in path.read_text() or 'integration' in path.name:
            path.unlink()
    script = target / 'scripts/train_full_finetune_speed_ood_pair.sh'
    if script.exists():
        old = 'source "${SCRIPT_DIR}/palmetto_env.sh"'
        if old not in script.read_text():
            old = 'source "${DVLA_ENV_SCRIPT:-${SCRIPT_DIR}/palmetto_env.sh}"'
        replace(script, old, 'source "${DVLA_ENV_SCRIPT:-${SCRIPT_DIR}/deltaai_env.sh}"')
        text = script.read_text()
        marker = 'if [[ "${RUN_OFFLINE_COMPARE}" == "1" ]]; then'
        if marker in text:
            text = text[:text.index(marker)] + 'echo "[full-finetune-speed-ood] Training complete. Simulator evaluation is excluded on DeltaAI."\n'
            script.write_text(text)
            changes.append({'path': f'{label}/scripts/{script.name}', 'reason': 'Removed optional post-training comparison stage; in-training recorded-data validation stays intact.'})
    # Both environments use the existing DeltaAI module/venv adapter.
    delta_env = target / 'scripts/deltaai_env.sh'
    if not delta_env.exists():
        shutil.copy2(HERE / 'staging/dom100/scripts/deltaai_env.sh', delta_env)
        changes.append({'path': f'{label}/scripts/deltaai_env.sh', 'reason': 'Added existing DeltaAI ARM module/venv adapter from DOM100 snapshot.'})

dom = BUNDLE / 'dom100'
train = dom / 'core/dom_training.py'
replace(train, ', source_digest, simulator_evaluation_mode', ', source_digest')
replace(train, '    if production:\n        simulator_evaluation_mode(root, recipe)\n', '')
replace(train, '            if milestone:\n                enqueue_evaluations(root, destination, variant, progress, best)\n', '')
replace(train, '        if production:\n            enqueue_evaluations(root, destination, variant, progress, best)\n', '')
replace(train, '            enqueue_evaluations(root, destination, variant, progress, best)\n', '')
text = train.read_text(); marker = '\n\ndef enqueue_evaluations('
if marker not in text: raise RuntimeError('Simulator dispatch function not found')
train.write_text(text[:text.index(marker)] + '\n')
changes.append({'path': 'dom100/core/dom_training.py', 'reason': 'Removed simulator request/dispatch function; optimizer, epochs, recorded-data validation and checkpoint logic unchanged.'})

runner = dom / 'scripts/dom100.py'
replace(runner, ', choose_sampling, simulator_evaluation_mode', ', choose_sampling')
replace(runner, '        from scripts.dom_closed_loop import validate_integration\n', '')
replace(runner, '        if simulator_evaluation_mode(root, recipe) == "during_training":\n            validate_integration(root)\n', '')
replace(runner, 'result["simulator_evaluation_mode"] = simulator_evaluation_mode(root)', 'result["simulator_evaluation_mode"] = "disabled_on_deltaai"')
replace(runner, 'result["evaluation_plan_sha256"] = file_digest(root/"evaluation_plan.json") if (root/"evaluation_plan.json").exists() else None', 'result["evaluation_plan_sha256"] = None')

experiment = dom / 'utils/dom_experiment.py'
replace(experiment, '"scripts", "simulations"', '"scripts"')
text = experiment.read_text()
start = text.index('\n\ndef simulator_evaluation_mode(')
end = text.index('\n\ndef grouped_stratified_split(', start)
experiment.write_text(text[:start] + text[end:])
changes.append({'path': 'dom100/utils/dom_experiment.py', 'reason': 'Removed simulator plan dependency; source fingerprint covers the transferred training-only tree.'})

(dom / 'scripts/dom100_env.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
export DVLA_PROJECT_DIR=${DVLA_PROJECT_DIR:-$PWD}
source "${DVLA_ENV_SCRIPT:-${DVLA_PROJECT_DIR}/scripts/deltaai_env.sh}"
export DOM100_ROOT=${DOM100_ROOT:-${DVLA_WORK_ROOT}/dom100}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=4 PYTHONUNBUFFERED=1
''')
run = dom / 'scripts/run_dom100.sh'
text = run.read_text()
start = text.index('    integration)')
end = text.index('    smoke)', start)
run.write_text(text[:start] + text[end:])
changes.append({'path': 'dom100/scripts/run_dom100.sh', 'reason': 'Removed Isaac controller integration command; retained preparation/profile/pilots/smoke/train runner and signal handling.'})
changes.append({'path': 'dom100/scripts/dom100_env.sh', 'reason': 'Routes to DeltaAI environment and project work storage instead of Palmetto conda/scratch.'})

for label in ('tracking', 'dom100'):
    for path in (BUNDLE / label / 'reference_palmetto').rglob('*.sbatch'):
        # Reference scripts remain documented, but are not active launchers.
        pass
    for path in (BUNDLE / label).rglob('*.py'):
        text = path.read_text()
        ast.parse(text, filename=str(path))
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or '']
            else: continue
            for name in names:
                if name.startswith(('isaac', 'omni', 'simulations', 'scripts.dom_closed_loop')):
                    raise RuntimeError(f'Forbidden simulator import in {path}: {name}')

created = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
snapshot_name = 'training-' + created
record = {'captured_utc': created, 'deltaai_root': '/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/' + snapshot_name, 'changes': changes, 'scope': 'code and small configuration/split metadata only; no dataset, weights, secrets, simulator, generated logs, or environment binaries', 'files': {}}
readme = '''# Current Palmetto training source for DeltaAI

This directory contains two separate current Palmetto training sources:

- `dom100/`: active matched full-DOM 100-epoch family, from `/scratch/aparame/Dynamic_VLA/third_party/dom100-20261007`, including DOM baseline, kinematic residual, neural velocity and causal phase-gate variants.
- `tracking/`: current home working tree from `/home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA`, including uncommitted October 6–8 model/trainer updates. The three original baseline/residual/coordinate recipes were recovered from the active run's provenance, alongside the current coordinate+velocity recipe.

`PALMETTO_SOURCE_MANIFEST.json` in each tree records original hashes, source location, and excluded paths. `TRANSFER_MANIFEST.json` records deployed hashes and the narrow platform/exclusion edits. Model weights, flow equations, optimizer behavior, training epoch counts and validation mathematics were not changed by this transfer. Dataset-based training validation (`core/test.py` and the DOM validation loop) does not use Isaac Sim and is retained.

Simulator modules, Isaac controllers, scene/asset generation, rollouts, simulator evaluation dispatch, datasets, videos, checkpoints, environments, caches, credentials and generated logs are excluded. DOM100 simulator checks/dispatch were removed from its training runner; selected recipes are explicitly tagged `disabled_on_deltaai`. Offline validation and GPU profile/smoke/reload prerequisites remain. Palmetto batch recipes are retained as reference under `reference_palmetto/slurm`, not as DeltaAI launchers. Old destination code is untouched.

The transfer does not submit jobs or install packages. Do not source `palmetto_env.sh` or install `requirements_rocky8.txt` on DeltaAI: those are x86/Palmetto references. The included `deltaai_env.sh` uses the existing ARM module/venv adapter. DeltaAI Python environment, current Slurm billing account, actual project storage, GPU profile and dataset/video decoding must be configured separately before training. No A100/H100/H200 profiling evidence is transplanted as GH200 approval.

Next step: obtain data on DeltaAI. DOM100 uses `hzxie/DOM`; tracking uses `adi2440/dom-franka-tracking`; initialization uses `hzxie/dynamic-vla-DOM`. Preserve immutable dataset/model revisions when preparing the new family. Rebuild DeltaAI family metadata/source fingerprints after code selection. Several historical tracking configs still point to a Palmetto normalization-reference checkpoint; that exact normalization information must be restored or prepared explicitly before running those variants. Do not silently substitute normalization statistics. Set `DVLA_OUTPUT_ROOT` to the DeltaAI work directory when using historical configs.
'''
(BUNDLE / 'README.md').write_text(readme)
for path in sorted(BUNDLE.rglob('*')):
    if path.is_file():
        data = path.read_bytes()
        record['files'][str(path.relative_to(BUNDLE))] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
(BUNDLE / 'TRANSFER_MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
archive = HERE / 'deltaai-training-only.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(BUNDLE, arcname=snapshot_name)
(HERE / 'deployment.json').write_text(json.dumps({'snapshot_name': snapshot_name, 'archive': str(archive), 'destination': record['deltaai_root'], 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2) + '\n')
print(json.dumps({'destination': record['deltaai_root'], 'files': len(record['files']) + 1, 'archive_bytes': archive.stat().st_size, 'adapted_entries': len(changes)}))
