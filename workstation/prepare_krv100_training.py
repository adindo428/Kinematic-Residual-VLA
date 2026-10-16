#!/usr/bin/env python3
"""Wait for the baseline/full data/login, then prepare and verify KRVLA+Velocity."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

WORK = Path(os.environ['KRVLA_WORK'])
ROOT = WORK/'training/krv100-full-20261010'
SOURCE = ROOT/'source'
FAMILY = ROOT/'family'
DATA = Path(os.environ['HF_LEROBOT_HOME'])/'hzxie/DOM'
PYTHON = WORK/'envs/train/bin/python'
STATUS = ROOT/'training_status.json'


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def status(phase, **details):
    record = {'phase':phase, 'variant':'krv100', 'family_root':str(FAMILY),
              'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), **details}
    write_json(STATUS, record)
    print(json.dumps(record), flush=True)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run(command, name, environment=None, timeout=None):
    with (ROOT/f'{name}.log').open('a') as log:
        process = subprocess.Popen([str(item) for item in command], cwd=SOURCE,
            env=environment or os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        finally:
            # Include launcher descendants, even if their parent already exited.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def main():
    request = read_json(ROOT/'request.json')
    if not request.get('user_authorized_training'):
        raise ValueError('Missing authorized training request')
    if not request.get('profiling_required'):
        raise ValueError('The user requires throughput profiling before production training')
    sys.path.insert(0, str(SOURCE))
    from utils.dom_experiment import source_digest, file_digest
    if source_digest(SOURCE) != request['source_sha256']:
        raise ValueError('Training source snapshot changed after the request was prepared')
    if request.get('readiness_adapter_sha256'):
        adapter = Path(__file__).with_name('readiness_krv100.py')
        if file_digest(adapter) != request['readiness_adapter_sha256']:
            raise ValueError('Readiness adapter differs from the reviewed recovery request')
    if request.get('training_launcher_sha256'):
        if file_digest(Path(__file__).with_name('train_krv100.py')) != request['training_launcher_sha256']:
            raise ValueError('Training launcher differs from the reviewed recovery request')
    if request.get('family_revision'):
        if file_digest(Path(__file__).with_name('revise_dom_family.py')) != request['family_revision']['adapter_sha256']:
            raise ValueError('Family revision adapter differs from the approved recovery request')
    if request.get('metadata_compaction'):
        for name, expected in request['metadata_compaction']['files_sha256'].items():
            if file_digest(Path(__file__).with_name(name)) != expected:
                raise ValueError('Worker metadata adapter differs from the recovery request')
    if request.get('epoch_limit'):
        for name, expected in request['epoch_limit']['files_sha256'].items():
            path = ROOT/name if name=='training_control.json' else Path(__file__).with_name(name)
            if file_digest(path)!=expected:
                raise ValueError('Authorized epoch-limit controller changed')
    exclusion_request = request.get('approved_exclusions')
    if exclusion_request:
        exclusions_path = Path(exclusion_request['path'])
        if file_digest(exclusions_path) != exclusion_request['sha256']:
            raise ValueError('Approved dataset exclusions changed after authorization')
        approved = read_json(exclusions_path)
        if (not approved.get('user_authorized_exclusions')
                or approved['dataset_revision'] != request['dataset_revision']
                or sorted(row['episode_index'] for row in approved['episodes']) != exclusion_request['episode_ids']):
            raise ValueError('Dataset exclusions differ from the exact authorized episodes')
    baseline = Path(request['baseline_results'])
    previous = None
    while True:
        summary = read_json(baseline/'summary.json')
        suite = read_json(baseline/'suite_status.json')
        download = read_json(WORK/'training/dom-full/download_status.json')
        missing = []
        if suite.get('state')!='complete' or not summary.get('complete') or summary.get('verified_trials')!=1800:
            missing.append('verified complete 1800-trial baseline evaluation')
        if (not download.get('complete') or not download.get('inventory_complete')
                or download.get('revision')!=request['dataset_revision']
                or download.get('completion_scope')!='all_episodes_training_cameras'
                or download.get('selected_cameras')!=request['required_cameras']
                or not download.get('checksum_verification')):
            missing.append('all DOM episodes, metadata and both training-camera video streams with pinned checksums')
        if request.get('palmetto_backfill_required'):
            backfill = read_json(WORK/'training/dom-full/palmetto_backfill_status.json')
            if (backfill.get('phase')!='complete' or not backfill.get('remote_checksum_verified')
                    or backfill.get('revision')!=request['dataset_revision']
                    or backfill.get('selected_cameras')!=request['required_cameras']):
                missing.append('verified two-camera dataset backfill to Palmetto')
        if not Path(os.environ['NETRC']).is_file():
            missing.append('your W&B login in the SSD credential file')
        if not missing:
            break
        if missing != previous:
            status('waiting', required=missing)
            previous = missing
        time.sleep(30)

    status('checking_wandb')
    import wandb
    api = wandb.Api(timeout=30)
    user = api.viewer
    entity = os.environ.get('DVLA_WANDB_ENTITY') or api.default_entity
    if entity not in user.teams and entity != user.username:
        raise ValueError('Selected W&B entity does not belong to the logged-in account')
    # The user requested W&B confirmation. A labeled setup run verifies write
    # access; its logs/configuration stay on the SSD, without exposing a key.
    previous_wandb = read_json(ROOT/'wandb_verified.json')
    if (previous_wandb.get('authenticated') and previous_wandb.get('write_access_verified')
            and previous_wandb.get('username')==user.username and previous_wandb.get('entity')==entity
            and previous_wandb.get('project')=='Kinematic-Residual-VLA'):
        print(json.dumps({'phase':'reusing_wandb_write_verification','username':user.username,
                          'readiness_url':previous_wandb['readiness_url']}),flush=True)
    else:
        with wandb.init(project='Kinematic-Residual-VLA', entity=entity, job_type='readiness',
                        name='krv100-ssd-readiness', dir=str(ROOT),
                        config={'variant':'krv100', 'purpose':'verify logging before training',
                                'dataset_revision':request['dataset_revision']} ) as readiness:
            readiness.log({'Setup/dataset_checksum_verified':1, 'Setup/baseline_trials':1800})
            wandb_url = readiness.url
        write_json(ROOT/'wandb_verified.json', {'authenticated':True, 'write_access_verified':True,
            'username':user.username, 'entity':entity, 'project':'Kinematic-Residual-VLA', 'readiness_url':wandb_url})
    environment = os.environ.copy()
    environment.update(DVLA_WANDB_ENTITY=entity, CUDA_VISIBLE_DEVICES='0,1')

    status('checking_training_environment' if PYTHON.exists() else 'preparing_training_environment')
    if not PYTHON.exists():
        if run([os.environ['CONDA_EXE'], 'create', '--prefix', PYTHON.parents[1],
                '--clone', WORK/'envs/policy', '-y'], 'environment', environment)!=0:
            raise RuntimeError('Training environment clone failed; inspect environment.log')
    environment['PATH'] = str(PYTHON.parent)+os.pathsep+environment['PATH']
    environment['CONDA_PREFIX'] = str(PYTHON.parents[1])
    if run([PYTHON,'-m','pip','check'], 'environment', environment)!=0:
        raise RuntimeError('Training dependencies failed pip check')
    import psutil
    memory = psutil.virtual_memory()
    budget_path = ROOT/'host_memory_budget.json'
    budget = read_json(budget_path)
    if not budget:
        budget = {'machine_total_bytes':memory.total,
                  'limit_mib':min(memory.available,int(memory.total*.90))//1024**2}
        write_json(budget_path,budget)
    if budget['machine_total_bytes']!=memory.total:
        raise ValueError('Host memory differs from the frozen profiling budget')
    environment['SLURM_MEM_PER_NODE'] = str(budget['limit_mib'])

    status('reusing_prepared_dataset_and_statistics' if (FAMILY/'manifest.json').exists()
           else 'freezing_dataset_and_training_statistics',
           approved_excluded_episodes=exclusion_request['episode_ids'] if exclusion_request else [])
    FAMILY.mkdir(parents=True, exist_ok=True)
    if not (FAMILY/'manifest.json').exists():
        if request.get('family_revision'):
            preparation = [PYTHON,Path(__file__).with_name('revise_dom_family.py'),
                           '--source',SOURCE,'--prior-family',request['family_revision']['prior_family'],
                           '--family-root',FAMILY,'--approved-exclusions',exclusions_path]
        else:
            preparation = [PYTHON,SOURCE/'scripts/prepare_dom100.py','--family-root',FAMILY,
                       '--dataset-root',DATA,'--dataset-revision',request['dataset_revision'],
                       '--model-revision',request['model_revision']]
        if exclusion_request and not request.get('family_revision'):
            preparation.extend(['--approved-exclusions',exclusions_path])
        if request.get('phase_cache_reuse_family') and not request.get('family_revision'):
            preparation.extend(['--reuse-phase-family',request['phase_cache_reuse_family']])
        code = run(preparation, 'dataset-preparation', environment)
        if code:
            raise RuntimeError('Full-dataset preparation failed; inspect dataset-preparation.log')
    manifest = read_json(FAMILY/'manifest.json')
    expected_ids = set(exclusion_request['episode_ids']) if exclusion_request else set()
    expected_frames = exclusion_request['frames'] if exclusion_request else 0
    mode = 'full_minus_approved_invalid_episodes' if exclusion_request else 'full'
    selection = read_json(FAMILY/'dataset_selection.json')
    actual_ids = [row['episode_index'] for row in selection['excluded']]
    if (manifest['episodes'] != download['episodes']-len(expected_ids)
            or manifest['dataset_selection_mode'] != mode
            or len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids
            or set(selection['episode_ids']) != set(range(download['episodes']))-expected_ids
            or manifest['coverage']['excluded_episodes'] != len(expected_ids)
            or manifest['coverage']['excluded_frames'] != expected_frames
            or manifest['frames'] != download['published_frames']-expected_frames
            or manifest['source_sha256'] != request['source_sha256']):
        raise ValueError('Training family differs from the full dataset minus exact approved exclusions')
    if exclusion_request and manifest['artifacts'].get('approved_exclusions.json') != exclusion_request['sha256']:
        raise ValueError('Training family does not preserve the approved exclusion evidence')
    if manifest['weights_sha256'] != request['weights_sha256']:
        raise ValueError('Training initialization differs from the released checkpoint')
    sys.path.insert(0, str(SOURCE))
    from utils.dom_experiment import digest, file_digest
    plan = {'mode':'after_training', 'reason':'Complete baseline first; defer trained-policy simulator evaluation until training finishes'}
    write_json(FAMILY/'evaluation_plan.json', plan)
    recipe = {**request['recipe'], 'manifest':manifest['fingerprint'],
              'simulator_evaluation_mode':'after_training',
              'evaluation_plan_sha256':file_digest(FAMILY/'evaluation_plan.json'),
              'selection':'Existing DOM100 fixed-rate recipe; capacity is measured on this workstation'}
    recipe['fingerprint'] = digest(recipe)
    old_recipe = read_json(FAMILY/'recipe.json')
    if old_recipe and old_recipe != recipe:
        raise ValueError('Existing recipe differs; prepare a fresh training family')
    write_json(FAMILY/'recipe.json', recipe)

    status('profiling_two_gpu_training', profiling_root=str(ROOT/'profiling'))
    supervisor = Path(__file__).resolve().with_name('profile_krv100.py')
    if file_digest(supervisor) != request['profiler_sha256']:
        raise ValueError('Profiler differs from the frozen training request')
    profiling_command = [PYTHON,supervisor,'--family-root',FAMILY,'--source',SOURCE,'--output',ROOT/'profiling']
    if request.get('profiling_revalidation'):
        prior_selection = Path(request['profiling_revalidation']['prior_selection'])
        if file_digest(prior_selection) != request['profiling_revalidation']['prior_selection_sha256']:
            raise ValueError('Prior profiling selection changed during recovery')
        profiling_command.extend(['--prior-selection',prior_selection])
    if run(profiling_command, 'profiling', environment):
        raise RuntimeError('Throughput profiling failed; inspect profiling.log and profiling/trials')
    measured = read_json(ROOT/'profiling/selected_runtime.json')
    copy = dict(measured);fingerprint = copy.pop('fingerprint')
    if (digest(copy)!=fingerprint or not measured['complete'] or not measured['safe']
            or not measured['correctness_checks_passed'] or measured['manifest']!=manifest['fingerprint']):
        raise ValueError('Missing complete measured runtime approval')
    from profile_krv100 import runtime_arguments
    selected = measured['runtime']
    rate = measured['samples_per_second']
    estimate = {'preliminary':True, 'measurement':'mean of two warm ten-update throughput confirmations',
                'samples_per_second':rate, 'train_frames':manifest['train_frames'],
                'training_only_epoch_days':manifest['train_frames']/rate/86400,
                'training_only_100_epochs_days':100*manifest['train_frames']/rate/86400,
                'excludes':['validation','checkpoint I/O','data preparation'],
                'note':'Not a completion prediction; full-epoch I/O, validation and early stopping can change duration.'}
    write_json(ROOT/'training_time_preliminary.json', estimate)
    status('training', selected_runtime=selected, global_effective_batch=352, wandb_entity=entity,
           preliminary_samples_per_second=rate, preliminary_training_only_epoch_days=estimate['training_only_epoch_days'])
    command = [PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
               Path(__file__).with_name('train_krv100.py'),'--source',SOURCE,'--family-root',FAMILY,
               '--variant','krv100','--command','train',*runtime_arguments(selected),
               '--performance-profile',ROOT/'profiling/selected_runtime.json']
    code = run(command, 'training', environment)
    if code:
        raise RuntimeError(f'Training exited with {code}; inspect training.log')
    if not (FAMILY/'runs/krv100/training_complete.json').exists():
        raise RuntimeError('Trainer exited without a completion record')
    status('complete', wandb_entity=entity)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        status('failed', error_type=type(error).__name__)
        raise
