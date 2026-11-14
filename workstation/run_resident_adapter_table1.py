#!/usr/bin/env python3
"""Resume selected low-clutter VLA checkpoints, ten repeats per scene and model."""
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import traceback

import run_low_clutter_table1 as low
from table1_display_state import write_if_changed

ROOT = Path(os.environ.get('DOM_RESIDENT_ADAPTER_ROOT', low.ROOT/'resident_adapters-20261108'))
MODELS = ['smolvla_kr', 'vlash_kr', 'pi05_baseline', 'pi05_kr']
STOP = threading.Event()


def stamp(): return datetime.now(timezone.utc).isoformat()
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def training_checkpoint_allowed(model, policy):
    if model['entry']['progress']['epoch'] >= policy['minimum_completed_epochs']: return True
    exception = policy.get('authorized_exceptions', {}).get(model['key'], {})
    return bool(exception.get('user_authorized') and exception.get('checkpoint_sha256') == model['entry']['checkpoint_sha256'])


def validate_rows(records, prompts, metrics, test, offset=0):
    cfg = low.read(test['path']); by_prompt = {r['trial']: r for r in prompts}; by_metric = {r['trial']: r for r in metrics}
    assert len(by_prompt) == len(prompts) and len(by_metric) == len(metrics)
    trials = [r['trial'] for r in records]
    assert trials == list(range(1, len(trials)+1))
    for row in records:
        trial = row['trial']; global_trial = trial + offset
        assert row['env_name'] == test['name'] and row['status'] in (0, 1)
        assert row['success'] == (row['status'] == 0)
        assert 1 < row['n_steps'] <= round(test['horizon_seconds']/row['step_dt_s']) + 2
        assert abs(row['task_time_s'] - row['n_steps']*row['step_dt_s']) < 1e-5
        assert 1 <= global_trial <= 20
        assert by_prompt[trial]['text'] == cfg['evaluation_prompts'][global_trial-1]
        assert by_metric[trial]['status'] == row['status']
        assert Path(by_metric[trial]['trajectory_path']).is_file()


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'plan.json').exists(), 'Existing schedule must be resumed without rebuilding identities'
    old = low.read(low.ROOT/'catalog.json')
    if not (ROOT/'catalog.before.json').exists(): low.write(ROOT/'catalog.before.json', old)
    receipt = low.read(low.WORK/'training/palmetto-updated-20261102/pi0.5kr/transfer_receipt.json')
    legacy = low.read(low.WORK/'evaluations/multivla20k-20261029/catalog.json')['models']['pi05_kr']
    entry = {**legacy, 'checkpoint': receipt['checkpoint'], 'weight_path': receipt['checkpoint'],
             'sha256': receipt['sha256'], 'checkpoint_sha256': receipt['sha256'],
             'progress': receipt['progress'], 'label': 'pi0.5 + KR (resumed, epoch 6 + partial 7)',
             'selection': receipt['selection'], 'training_wandb_run': receipt['training_wandb_run'],
             'user_authorized_latest_partial': True, 'transfer_receipt': str(low.WORK/'training/palmetto-updated-20261102/pi0.5kr/transfer_receipt.json'),
             'training_scope': '20k_subset_with_prior_full_data_exposure', 'client_kind': 'resident_adapter'}
    stat = Path(entry['checkpoint']).stat()
    entry.update(checkpoint_bytes=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns,
                 weight_bytes=stat.st_size, weight_mtime_ns=stat.st_mtime_ns)
    assert entry['identity'] == receipt['identity'] and entry['progress']['updates'] == 545349
    catalog = {**old, 'models': {**old['models'], 'pi05_kr': entry}, 'enabled_models': MODELS,
               'held_models': ['vlash_baseline'], 'queue_order': MODELS}
    if 'pi05_kr' not in catalog['order']: catalog['order'].append('pi05_kr')
    protocol = low.read(low.ROOT/'protocol.json'); tests = {t['name']: t for t in protocol['tests']}
    assert len(tests) == 90
    for test in tests.values(): assert low.sha(test['path']) == test['config_sha256']
    from evaluation_training_policy import load_policy, require_eligible
    audit = load_policy()
    specs = []
    for name in MODELS:
        e = catalog['models'][name]; s = Path(e['checkpoint']).stat()
        assert (s.st_size, s.st_mtime_ns) == (e['checkpoint_bytes'], e['checkpoint_mtime_ns'])
        if name != 'pi05_kr': require_eligible(name, e)
        else:
            assert audit['models'][name]['identity'] == e['identity']
            assert audit['models'][name]['prior_full_data_epochs'] < 1
            assert e['user_authorized_latest_partial'] and receipt['copied_bytes_hash_verified']
        # Preserve all frozen policy/recipe/normalization pins. Controller changes get new pins.
        for path, digest in old['files_sha256'].items():
            if str(Path(path)).startswith(e['code_root']+'/'): assert low.sha(path) == digest, path
        root = low.ROOT/'results'/name; root.mkdir(exist_ok=True)
        if name == 'pi05_kr':
            low.write(root/'protocol.json', {**protocol, 'checkpoint': e})
            import shutil
            shutil.copytree(low.ROOT/'test_lists', root/'test_lists', dirs_exist_ok=True)
        specs.append({'key': name, 'entry': e, 'catalog': {k: catalog[k] for k in ['official_source', 'vlash_source']}})
    saved = {name: [] for name in MODELS}; sources = []
    # Retain acknowledged complete columns and the durable interrupted MP prefix.
    for name in MODELS:
        if name == 'pi05_kr': continue
        root = low.ROOT/'results'/name
        assert low.read(root/'protocol.json')['checkpoint']['checkpoint_sha256'] == catalog['models'][name]['checkpoint_sha256']
        for index, dim in enumerate(low.DIMS):
            folder = root/'full'/f'{index:02d}_{dim}'; path = folder/'trial_results.jsonl'
            if not path.exists(): continue
            raw = rows(path); trace = rows(folder/'prompt_trace.jsonl'); metrics = rows(folder/'trajectory_metrics.jsonl')
            groups = defaultdict(list)
            for r in raw: groups[r['env_name']].append(r)
            for scene, group in groups.items():
                validate_rows(group, [r for r in trace if r['env_name'] == scene], [r for r in metrics if r['env_name'] == scene], tests[scene])
            if (folder/'verification.json').exists(): assert low.read(folder/'verification.json')['verified']
            else: assert name == 'smolvla_kr' and dim == 'MP', 'Unrecognized interrupted source'
            saved[name] += [{**r, 'dimension': dim, 'source': str(folder)} for r in raw]
            sources.append({'model': name, 'folder': str(folder), 'records_sha256': low.sha(path), 'trials': len(raw),
                            'durable_prefix': not (folder/'verification.json').exists()})
    low.write(ROOT/'existing.json', {'records': saved, 'sources': sources})
    jobs = []
    lists = {dim: [s for s in (low.ROOT/'test_lists'/(dim+'.txt')).read_text().splitlines() if s] for dim in low.DIMS}
    pilot_scene = lists['CR'][0]; pilot_test = tests[Path(pilot_scene).stem]
    for i, name in enumerate([*MODELS, MODELS[0]]):
        key = f'pilot-{i}-{name}'; folder = ROOT/'blocks'/key
        jobs.append({'id': key, 'model': name, 'scene': pilot_scene, 'scene_sha256': pilot_test['config_sha256'],
                     'repeats': 1, 'repeat_offset': 0, 'folder': str(folder), 'dimension': 'CR', 'scored': False})
    for pass_index in (1, 2):
        begin, end = (1, 11) if pass_index == 1 else (11, 21)
        for dim in low.DIMS:
            assert len(lists[dim]) == 10
            for scene in lists[dim]:
                name_scene = Path(scene).stem
                for name in MODELS:
                    present = {r['trial'] for r in saved[name] if r['env_name'] == name_scene}
                    missing = [i for i in range(begin, end) if i not in present]
                    if not missing: continue
                    assert missing == list(range(missing[0], end))
                    key = f'pass{pass_index}-{dim}-{name_scene}-{name}'; folder = ROOT/'blocks'/key
                    jobs.append({'id': key, 'model': name, 'scene': scene, 'scene_sha256': tests[name_scene]['config_sha256'],
                                 'repeats': len(missing), 'repeat_offset': missing[0]-1, 'folder': str(folder),
                                 'dimension': dim, 'scored': True, 'pass_index': pass_index})
    for job in jobs:
        folder = Path(job['folder']); folder.mkdir(parents=True, exist_ok=True)
        assert not (folder/'trial_results.jsonl').exists()
        (folder/'case.txt').write_text(job['scene']+'\n')
    pins = {str(p): low.sha(p) for spec in specs for p in Path(spec['entry']['code_root']).rglob('*') if p.is_file() and p.suffix in ('.py', '.json')}
    pins.update({str(Path(__file__).parent/f): low.sha(Path(__file__).parent/f) for f in [
        'run_resident_adapter_table1.py', 'resident_adapter_table1.py', 'resident_adapter_table1_client.py',
        'resident_low_clutter_server.py', 'smolvla_table1_client.py']})
    plan = {'models': specs, 'catalog': {k: catalog[k] for k in ['official_source', 'vlash_source']}, 'jobs': jobs,
            'files_sha256': pins, 'success_history': {name: sorted({r['env_name'] for r in saved[name] if r['success']}) for name in MODELS},
            'first_pass_repeats': 10, 'remaining_repeats': 10, 'automatic_second_pass_authorized': True,
            'new_scored_trials': sum(j['repeats'] for j in jobs if j['scored']), 'minimum_latency_retention': .9,
            'protocol_sha256': low.sha(low.ROOT/'protocol.json')}
    low.write(ROOT/'plan.json', plan); low.write(low.ROOT/'catalog.json', catalog)
    low.write(ROOT/'request.json', {'enabled': True, 'user_authorized': True, 'models': MODELS, 'minimum_latency_retention': .9,
              'scene_first': True, 'concurrent_inference': False, 'block_repeats_maximum': 10,
              'automatic_second_pass_authorized': True, 'target_per_model': 1800, 'created_utc': stamp()})
    return plan


def collect(plan, validate=True):
    existing = low.read(ROOT/'existing.json'); records = {name: list(existing['records'][name]) for name in MODELS}
    if validate:
        for source in existing['sources']: assert low.sha(Path(source['folder'])/'trial_results.jsonl') == source['records_sha256']
    tests = {t['name']: t for t in low.read(low.ROOT/'protocol.json')['tests']}
    for job in plan['jobs']:
        folder = Path(job['folder'])
        if not job['scored'] or not (folder/'resident_client_completion.json').exists(): continue
        receipt = low.read(folder/'resident_client_completion.json')
        assert receipt['complete'] and receipt['trials'] == job['repeats']
        entry = next(s['entry'] for s in plan['models'] if s['key'] == job['model'])
        assert receipt['checkpoint_sha256'] == entry['checkpoint_sha256']
        raw = rows(folder/'trial_results.jsonl'); assert len(raw) == job['repeats']
        if validate:
            validate_rows(raw, rows(folder/'prompt_trace.jsonl'), rows(folder/'trajectory_metrics.jsonl'), tests[Path(job['scene']).stem], job['repeat_offset'])
            low.write(folder/'verification.json', {'verified': True, 'trials': len(raw), 'matched_prompts_verified': True,
                      'trajectory_metrics_verified': True, 'checkpoint_sha256': entry['checkpoint_sha256']})
        records[job['model']] += [{**r, 'trial': r['trial']+job['repeat_offset'], 'dimension': job['dimension'], 'source': str(folder)} for r in raw]
    for name, rr in records.items():
        keys = [(r['env_name'], r['trial']) for r in rr]; assert len(keys) == len(set(keys)), 'Duplicate trial reuse'
    return records


def publish(plan):
    records = collect(plan, validate=False)
    state = low.read(ROOT/'execution/status.json') if (ROOT/'execution/status.json').exists() else {}
    active = []
    if state.get('stage') == 'evaluating':
        job = state['job']; path = Path(job['folder'])/'trial_results.jsonl'
        count = len(rows(path)) if path.exists() else 0
        active = [{'model': job['model'], 'phase': 'full' if job['scored'] else 'pilot', 'dimension': job['dimension'],
                   'completed_records_in_phase': count, 'phase_trial_target': job['repeats'], 'pass_index': job.get('pass_index', 1)}]
    stage = state.get('stage', 'preparing')
    low.write(ROOT/'live_results.json', {'updated_utc': stamp(), 'stage': stage, 'active_evaluations': active,
              'records': records})
    low.status('evaluating' if stage != 'complete' else 'complete', active_evaluations=active,
               execution_mode='scene_first_sequential_resident_adapters', controller=str(ROOT), enabled_models=MODELS,
               automatic_second_pass=True)
    render_table()


def render_table():
    """Display all retained results, with explicit denominators for partial columns."""
    live_path = ROOT/'live_results.json'
    if not live_path.exists(): return
    live = low.read(live_path); catalog = low.read(low.ROOT/'catalog.json')
    held = low.read(ROOT/'held_records.json') if (ROOT/'held_records.json').exists() else {}
    policy = low.read(ROOT/'checkpoint_policy.json') if (ROOT/'checkpoint_policy.json').exists() else {}
    display_records = {**held, **live['records']}
    active = {r['model']: r for r in live['active_evaluations']}
    keys = [*low.DIMS, 'Success', 'Time']; lines = ['| Method | Status | '+' | '.join(keys)+' |', '|---|---|'+'---:|'*len(keys)]
    from table1_display_state import observed, status
    for name in catalog['order']:
        root = low.ROOT/'results'/name; values = observed(root); label = catalog['models'][name]['label']
        if name in display_records:
            rr = display_records[name]; totals = []
            for dim in low.DIMS:
                group = [r for r in rr if r['dimension'] == dim]; n = len(group); totals.append(n)
                if n:
                    score = 100*sum(r['success'] for r in group)/n
                    values[dim] = f'{score:.2f}' if n == 200 else f'{score:.2f} (n={n}, provisional)'
            if totals == [200]*9:
                values.update(Success=100*sum(r['success'] for r in rr)/1800, Time=sum(r['task_time_s'] for r in rr)/1800)
            elif totals == [100]*9:
                values.update(Success=f'{100*sum(r["success"] for r in rr)/900:.2f} (n=900, provisional)', Time=f'{sum(r["task_time_s"] for r in rr)/900:.2f} (provisional)')
            if name in policy.get('held_models', {}):
                state = 'On hold — awaiting 10 completed training epochs'
            elif name in active:
                a = active[name]
                state = f'Running {a["dimension"]}: {a["completed_records_in_phase"]}/{a["phase_trial_target"]} in current scene'
                if a['phase'] == 'pilot': state = 'Running excluded pilot / checkpoint switch check'
                elif a['dimension'] not in values: values[a['dimension']] = 'running'
            elif totals == [200]*9: state = 'Complete'
            elif live['stage'] == 'loading_resident_policies': state = 'Loading / qualifying resident models'
            elif live['stage'] == 'complete': state = 'Complete'
            else: state = 'Pending next scene block'
        else: state = ('On hold — awaiting 10 completed training epochs' if name in policy.get('held_models', {}) else
                       'Complete' if (root/'summary.json').exists() and low.read(root/'summary.json').get('complete') else 'Paused')
        cells = [f'{values[k]:.2f}' if isinstance(values.get(k), (int, float)) else values.get(k, 'pending') for k in keys]
        lines.append('| '+label+' | '+state+' | '+' | '.join(cells)+' |')
    path = low.ROOT/'table1_comparison.md'; old = path.read_text().splitlines()
    start = next(i for i, line in enumerate(old) if line.startswith('| Method |')); end = start
    while end < len(old) and old[end].startswith('|'): end += 1
    text = '\n'.join(old[:start]+lines+old[end:])+'\n'
    prefix = 'Resident adapter schedule:'
    text = '\n'.join(line for line in text.splitlines() if not line.startswith(prefix)).rstrip()+'\n\n'
    while '\n\n\n' in text: text = text.replace('\n\n\n', '\n\n')
    text += prefix+' one persistent Isaac Sim application; sequential checkpoints per scene, at most ten repeats per block. Complete repeats 1–10 across all nine columns, then 11–20 automatically (1,800 trials/model). Existing scored trials are retained. Startup pilots are excluded. GPU residency is conditional on per-model p50/p95 latency retaining at least 90% of isolated performance.'
    if policy:
        if policy.get('authorized_exceptions'):
            partial = [catalog['models'][key]['label'] for key in policy['authorized_exceptions'] if key in catalog['models']]
            text += ' Explicitly authorized partial-training checkpoints: '+', '.join(partial)+'. Their training budgets differ from completed epoch-10 baselines.'
        else:
            text += ' Active checkpoints have at least 10 completed training epochs. Previously completed partial-training rows retain their actual checkpoint labels.'
        if policy.get('held_models'):
            text += ' Unfinished checkpoints remain on hold.'
    text += '\n'
    write_if_changed(path, text)


def launch(plan):
    execution = ROOT/'execution'; execution.mkdir(exist_ok=True)
    listing = execution/'initial_case.txt'; listing.write_text(plan['jobs'][0]['scene']+'\n')
    kit = low.WORK/'kit/table1-resident-adapters'
    kit_args = os.environ['KRVLA_KIT_ARGS'].replace('--portable-root='+str(low.WORK/'kit'), '--portable-root='+str(kit))
    for token, sub in [('omni_global_cache', 'cache'), ('omni_global_logs', 'logs'), ('omni_global_data', 'data')]:
        kit_args = kit_args.replace(f'--/app/tokens/{token}='+str(low.WORK/sub/'omniverse'), f'--/app/tokens/{token}='+str(kit/sub))
    kit_args += ' --/app/extensions/fsWatcherEnabled=false --/renderer/activeGpu=0 --/renderer/multiGpu/enabled=false'
    server = [str(low.REPO/'workstation/isaaclab-paper45.sh'), '-p', str(low.REPO/'workstation/resident_low_clutter_server.py'),
              '--headless', '--enable_cameras', '--device', 'cuda:0', '--num_envs', '1', '--tolerance', '0.07',
              '--scene_dir', str(low.WORK/'assets/scenes'), '--object_dir', str(low.WORK/'assets/objects'),
              '--env_cfg', str(listing), '--output_dir', str(execution/'videos'), '-n', '1', '--save',
              '--img_port', '27600', '--act_port', '27601', '--kit_args', kit_args]
    client = [low.read(low.ROOT/'catalog.json')['python'], '-u', str(low.REPO/'workstation/resident_adapter_table1_client.py'),
              '--plan', str(ROOT/'plan.json'), '--output', str(execution)]
    env = dict(os.environ, DOM_OFFICIAL_SOURCE=plan['catalog']['official_source'], DOM_RESIDENT_PLAN=str(ROOT/'plan.json'),
               DOM_TRIAL_RESULTS=str(execution/'unused.jsonl'), DOM_EXPECTED_SIM='4.5.0', DOM_EXPECTED_LAB='2.2.1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    low.write(execution/'commands.json', {'server': server, 'client': client})
    processes = []; last = time.monotonic(); size = -1
    try:
        with (execution/'server_shared.log').open('w') as sl, (execution/'client_shared.log').open('w') as cl:
            processes.append(subprocess.Popen(server, env=dict(env, CUDA_VISIBLE_DEVICES='0'), stdout=sl, stderr=subprocess.STDOUT, start_new_session=True))
            processes.append(subprocess.Popen(client, env=dict(env, CUDA_VISIBLE_DEVICES='1'), stdout=cl, stderr=subprocess.STDOUT, start_new_session=True))
            while processes[1].poll() is None:
                if STOP.wait(5): raise InterruptedError('Evaluation stopped')
                state = low.read(execution/'status.json') if (execution/'status.json').exists() else {}
                if processes[0].poll() is not None and state.get('stage') != 'complete': raise RuntimeError('Resident simulator exited')
                publish(plan)
                current = sum(p.stat().st_size for p in (ROOT/'blocks').glob('*/trial_results.jsonl'))
                if current != size: size = current; last = time.monotonic()
                # Loading and four-backbone paired qualification can exceed 20 minutes.
                if state.get('stage') == 'evaluating' and time.monotonic()-last > 1200: raise TimeoutError('No completed trial for 20 minutes')
            if processes[1].returncode: raise RuntimeError('Resident adapter client failed; inspect execution/client_shared.log')
        records = collect(plan)
        tests = low.read(low.ROOT/'protocol.json')['tests']
        for name, rr in records.items():
            for test in tests: assert {r['trial'] for r in rr if r['env_name'] == test['name']} == set(range(1, 21))
            assert len(rr) == 1800
            observed = {d: 100*sum(r['success'] for r in rr if r['dimension'] == d)/200 for d in low.DIMS}
            observed.update(Success=100*sum(r['success'] for r in rr)/1800, Time=sum(r['task_time_s'] for r in rr)/1800)
            aggregate = ROOT/'aggregates'/name; aggregate.mkdir(parents=True, exist_ok=True)
            (aggregate/'trial_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
            low.write(aggregate/'verification.json', {'verified': True, 'trials': 1800, 'sources': sorted({r['source'] for r in rr}),
                      'checkpoint_sha256': next(m['entry']['checkpoint_sha256'] for m in plan['models'] if m['key'] == name)})
            low.write(low.ROOT/'results'/name/'summary.json', {'complete': True, 'verified_trials': 1800, 'observed': observed,
                      'checkpoint': next(m['entry'] for m in plan['models'] if m['key'] == name), 'aggregate': str(aggregate),
                      'protocol_sha256': plan['protocol_sha256']})
        low.write(ROOT/'completion.json', {'complete': True, 'all_twenty_repeats_all_models_verified': True,
                  'trials_saved_by_model': {k: len(v) for k, v in records.items()}, 'updated_utc': stamp()})
        publish(plan)
        request = low.read(ROOT/'request.json')
        if request.get('notify_on_completion'):
            import shutil
            command = shutil.which('notify-send')
            outcome = {'requested': True, 'updated_utc': stamp(), 'complete_models': list(records),
                       'results_table': str(low.ROOT/'table1_comparison.md')}
            if command:
                try:
                    sent = subprocess.run([command, '--app-name=Dynamic VLA', '--urgency=normal',
                        'Table 1 evaluation complete',
                        ', '.join(records)+' finished 1,800 trials each. Results: '+str(low.ROOT/'table1_comparison.md')],
                        timeout=15, capture_output=True, text=True)
                    outcome.update(delivered_to_desktop=sent.returncode == 0, returncode=sent.returncode)
                except (OSError, subprocess.TimeoutExpired) as error:
                    outcome.update(delivered_to_desktop=False, error=str(error))
            else: outcome.update(delivered_to_desktop=False, error='notify-send unavailable')
            low.write(ROOT/'completion_notification.json', outcome)
    finally:
        for process in reversed(processes): low.stop_group(process)


def main():
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    ROOT.mkdir(parents=True, exist_ok=True)
    with (low.ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = low.read(ROOT/'plan.json') if (ROOT/'plan.json').exists() else prepare()
        MODELS[:] = [model['key'] for model in plan['models']]
        policy = low.read(ROOT/'checkpoint_policy.json') if (ROOT/'checkpoint_policy.json').exists() else {}
        if policy:
            for model in plan['models']:
                assert training_checkpoint_allowed(model, policy), 'Unfinished training checkpoint is on hold'
        for path, digest in plan['files_sha256'].items(): assert low.sha(path) == digest, 'Pinned source changed: '+path
        try: launch(plan)
        except BaseException as error:
            low.write(ROOT/'failure.json', {'error': str(error), 'traceback': traceback.format_exc(), 'updated_utc': stamp()})
            low.status('paused_infrastructure_error', active_evaluations=[], reason=str(error)); raise


if __name__ == '__main__': main()
