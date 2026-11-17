"""Queue a 200-CR-trial/model/condition sweep after the active 1800/model run."""
import argparse
import collections
import copy
import fcntl
import inspect
import json
import math
from pathlib import Path
import signal
import subprocess

import extend_dom_krv_epoch4_1800 as history
import run_closed_loop_pilot as runner

low = runner.low
ROOT = low.WORK/'evaluations/velocity-uncertainty-cr200-20261116'
PREDECESSOR = history.ROOT
CONDITIONS = [('clean', 0.), ('noise001', .01), ('noise005', .05), ('noise010', .10), ('efficienttam', None)]
DEFAULT_TRIALS = 100


def trial_target(plan):
    return int(plan['protocol']['trials_per_model_per_condition'])


def grouped_jobs(jobs, tests):
    scenes = {test['path']: i for i, test in enumerate(tests)}
    models = {'domv_best': 0, 'krv_latest_constant': 1}
    conditions = {condition: i for i, (condition, _) in enumerate(CONDITIONS)}
    return sorted(jobs, key=lambda j: (scenes[j['scene']], models[j['model']],
                                      conditions[j['condition']], j.get('repeat_offset', 0)))


def records(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def check_simulator_error():
    path = ROOT/'execution/server_shared.log'
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size-65536))
            if b'Traceback (most recent call last):' in stream.read():
                raise RuntimeError('Simulator raised an exception; inspect execution/server_shared.log')


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'plan.json').exists():
        return low.read(ROOT/'plan.json')
    previous = low.read(PREDECESSOR/'scene-first/plan.json')
    krv = copy.deepcopy(next(m for m in previous['models'] if m['key'] == 'krv_latest_constant'))
    domv = copy.deepcopy(low.read(low.WORK/'evaluations/closed-loop-domv-innovation-cdf-20261112/plan.json')['models'][0])
    domv.update(vpd=False, vi=False, deterministic_observation_noise=True, require_velocity_sensitivity=True)
    models = [krv, domv]
    tests = [t for t in previous['tests'] if t['dimension'] == 'CR']
    assert len(tests) == 10 and all(not m.get('vpd') and not m.get('vi') for m in models)
    prompts = copy.deepcopy(previous['saved_prompts'])
    # Collect actual second-pass prompts as they become available before launch.
    for phase in ['krv_epoch4', 'dom']:
        for path in (PREDECESSOR/phase/'blocks').glob('*/prompt_trace.jsonl'):
            scene = next((t['path'] for t in tests if t['name'] == path.parent.name), None)
            if scene:
                for row in records(path):
                    prompts.setdefault(scene, {})[str(row['global_trial'])] = row['text']
    jobs = []
    for test in tests:
        for offset in range(0, DEFAULT_TRIALS//len(tests), 10):
            # Exercise the real tracker immediately, before a long Gaussian sweep.
            for condition, sigma in [CONDITIONS[-1], *CONDITIONS[:-1]]:
                for model in models:
                    uid = f"{test['name']}-{offset:02d}-{condition}-{model['key']}"
                    folder = ROOT/'blocks'/uid
                    folder.mkdir(exist_ok=True, parents=True)
                    (folder/'case.txt').write_text(test['path']+'\n')
                    jobs.append(dict(id=uid, model=model['key'], scene=test['path'],
                        scene_sha256=test['config_sha256'], dimension='CR', repeats=10,
                        repeat_offset=offset, folder=str(folder), scored=True, seed=test['seed'],
                        vi=False, vpd=False, condition=condition,
                        velocity_sensor='efficienttam' if sigma is None else 'gaussian', sigma_v_mps=sigma))
    files = [low.REPO/'workstation'/name for name in [
        'run_velocity_uncertainty_cr.py', 'velocity_uncertainty_server.py', 'velocity_uncertainty_sensor.py',
        'velocity_sensor_math.py', 'efficienttam_stream_worker.py', 'dom_krv_scene_client.py',
        'dom_krv_scene_worker.py', 'krv_profile_ablation_worker.py', 'krv_profile_ablation_client.py',
        'resident_table1_worker.py', 'resident_table1_client.py', 'dom_repeat_extension_server.py',
        'krv_full_dom_server.py', 'krv_profile_ablation_server.py', 'resident_low_clutter_server.py',
        'low_clutter_table1_server.py', 'resident_table1_server.py', 'collection_matched_server.py',
        'sampled_video_table1_server.py', 'causal_gate_diagnostic_server.py', 'table1_video_sampling.py',
        'krv_profile_math.py', 'velocity_innovation_worker.py']]
    for source in {m['source'] for m in models}:
        files.extend(Path(source).rglob('*.py'))
    files.extend((low.WORK/'perception/EfficientTAM/efficient_track_anything').rglob('*.py'))
    files.extend((low.WORK/'perception/EfficientTAM/efficient_track_anything').rglob('*.yaml'))
    files.append(low.WORK/'perception/checkpoints/efficienttam_ti_512x512.pt')
    tracker_revision = subprocess.check_output(['git', '-C', str(low.WORK/'perception/EfficientTAM'), 'rev-parse', 'HEAD'], text=True).strip()
    protocol = dict(trials_per_model_per_condition=DEFAULT_TRIALS, conditions=5, models=2, total_trials=DEFAULT_TRIALS*10,
        scene_count=10, repeats_per_scene=DEFAULT_TRIALS//10, category='CR', tolerance_m=.03,
        physical_velocity_changes=False, measurement_noise='iid Gaussian per simulation observation, per axis',
        noise_seed=16311, matched_noise_key='scene path, global repeat, observation index; independent of model/condition',
        VPD=False, VI=False, KRV_gate='constant',
        tracker='EfficientTAM-Ti 512', tracker_source='https://github.com/yformer/EfficientTAM',
        tracker_revision=tracker_revision, tracker_device='simulation GPU 0, isolated worker',
        tracker_initialization='privileged target instance mask on first frame only',
        tracker_input='opst_cam RGB, metric depth, calibrated intrinsics/extrinsics, robot base transform',
        tracker_memory='initial conditioning and last 64 non-conditioning outputs; only current RGB frame',
        velocity_estimator='causal least-squares position slope, trailing 0.16s, minimum three samples',
        tracker_loss_policy='zero velocity; clear slope history; never fall back to privileged state',
        ground_truth_use='first-frame mask; all later object position/velocity for diagnostics only',
        position_reference='visible surface centroid versus privileged object COM; surface/centre offset is not removed',
        reported_noise='per-axis bias/std/RMSE and sigma_eq=sqrt(mean_xyz,t(error^2)); not assumed Gaussian',
        comparison_limit='existing DOMV and KRV checkpoints have different training histories/dataset sizes and saved normalizers; native KRV and deterministic DOMV policy RNG are not paired',
        extra_camera_modalities='depth and non-colorized instance masks enabled in every arm',
        reset_camera_guard='Two render-only updates in every condition after each reset, explicitly invalidate sensor caches; physics timestamp unchanged',
        prerequisite=str(PREDECESSOR/'completion.json'))
    plan = dict(models=models, jobs=grouped_jobs(jobs, tests), tests=tests, saved_prompts=prompts, success_history={},
                protocol=protocol, files_sha256={str(p): low.sha(p) for p in files})
    assert len(jobs) == DEFAULT_TRIALS and len({j['id'] for j in jobs}) == DEFAULT_TRIALS
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'protocol.json', protocol)
    render(plan)
    return plan


def render(plan):
    counts = collections.Counter()
    successes = collections.Counter()
    diagnostics = collections.defaultdict(list)
    for job in plan['jobs'] + plan.get('retained_jobs', []):
        key = (job['condition'], job['model'])
        rows = records(Path(job['folder'])/'trial_results.jsonl')
        counts[key] += len(rows)
        successes[key] += sum(bool(x['success']) for x in rows)
        for path in (Path(job['folder'])/'velocity_diagnostics').glob('repeat*.json'):
            diagnostics['/'.join(key)].append(low.read(path))
    target = trial_target(plan)
    lines = [f'# Velocity uncertainty: {target} CR trials per model per condition', '',
             'Velocity-conditioned DOMV and historical constant-gate KRV; VPD and VI disabled. Physical motion is unchanged. Percentages use completed trials.', '',
             '| Velocity measurement | DOMV | KRV |', '|---|---:|---:|']
    for condition, sigma in CONDITIONS:
        cells = []
        for model in ['domv_best', 'krv_latest_constant']:
            k = (condition, model);n = counts[k];s = successes[k]
            cells.append(f'{100*s/n:.2f}% ({s}/{n})' if n else f'pending (0/{target})')
        label = f'Gaussian sigma={sigma:g} m/s/axis' if sigma is not None else 'EfficientTAM-Ti RGB-D'
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', 'EfficientTAM noise is measured against privileged position/velocity, never guessed. See error_metrics.json for per-model bias/std/RMSE, tracking availability and sensor latency. Visible-surface position and COM differ geometrically.']
    (ROOT/'live_results.md').write_text('\n'.join(lines)+'\n')
    low.write(ROOT/'results.json', {'counts': {'/'.join(k): v for k, v in counts.items()},
        'successes': {'/'.join(k): v for k, v in successes.items()}})
    # Aggregate sufficient statistics, preserving each model's trajectory distribution.
    metrics = {}
    for key, entries in diagnostics.items():
        model_metrics = {}
        for field in ['velocity_error_mps', 'velocity_error_ready_mps', 'visible_surface_vs_COM_error_m']:
            valid = [e[field] for e in entries if e[field]['samples']]
            n = sum(e['samples'] for e in valid)
            if not n:
                model_metrics[field] = {'samples': 0};continue
            bias = [sum(e['samples']*e['bias_xyz'][i] for e in valid)/n for i in range(3)]
            mse = [sum(e['samples']*e['rmse_xyz'][i]**2 for e in valid)/n for i in range(3)]
            model_metrics[field] = dict(samples=n, bias_xyz=bias, rmse_xyz=[math.sqrt(v) for v in mse],
                std_xyz=[math.sqrt(max(0, mse[i]-bias[i]**2)) for i in range(3)],
                sigma_equivalent=math.sqrt(sum(mse)/3))
        total = sum(e['observations'] for e in entries)
        model_metrics['unavailable_fraction'] = 1-sum(e['ready_observations'] for e in entries)/total if total else None
        model_metrics['initialization_failures'] = sum(bool(e.get('initialization_failure')) for e in entries)
        metrics[key] = model_metrics
    low.write(ROOT/'error_metrics.json', metrics)


def status(stage, **kwargs):
    low.write(ROOT/'status.json', dict(stage=stage, updated_utc=runner.stamp(), **kwargs))
    if (ROOT/'plan.json').exists():
        render(low.read(ROOT/'plan.json'))


def verify(plan):
    counts = collections.Counter()
    seen = set()
    for job in plan['jobs'] + plan.get('retained_jobs', []):
        rows = records(Path(job['folder'])/'trial_results.jsonl')
        assert len(rows) == job['repeats']
        if job.get('retained_interrupted_block'):
            assert low.sha(Path(job['folder'])/'trial_results.jsonl') == job['retained_results_sha256']
        else:
            receipt = low.read(Path(job['folder'])/'resident_client_completion.json')
            assert receipt['complete'] and receipt['trials'] == len(rows)
            assert receipt['checkpoint_sha256'] == next(m['weight_sha256'] for m in plan['models'] if m['key'] == job['model'])
        assert len(list((Path(job['folder'])/'velocity_diagnostics').glob('repeat*.json'))) == job['repeats']
        for row in rows:
            assert row['status'] in (0, 1)
            key = (job['condition'], job['model'], job['scene'], row['trial']+job.get('repeat_offset', 0))
            assert key not in seen
            seen.add(key)
        counts[(job['condition'], job['model'])] += len(rows)
    assert len(counts) == 10 and set(counts.values()) == {trial_target(plan)}
    low.write(ROOT/'completion.json', dict(verified=True, trials=sum(counts.values()), completed_utc=runner.stamp()))
    render(plan)
    update_paper()


def update_paper():
    """Fill only the five reserved data rows after all matched trials pass checks."""
    import math
    result = low.read(ROOT/'results.json')
    target = trial_target(low.read(ROOT/'plan.json'))
    metrics = low.read(ROOT/'error_metrics.json')
    estimates = [metrics['efficienttam/'+key]['velocity_error_mps'] for key in ['domv_best', 'krv_latest_constant']]
    n = sum(e['samples'] for e in estimates)
    sigma = math.sqrt(sum(e['samples']*e['sigma_equivalent']**2 for e in estimates)/n)
    path = low.REPO/'CVPR27_Kinematic_Residual_Flow_Matching_for_Vision_Language_Action_Policies/tables/velocity_uncertainty.tex'
    text = path.read_text()
    original = text
    labels = ['Privileged & $0$', 'Gaussian & $0.01$', 'Gaussian & $0.05$', 'Gaussian & $0.10$', 'EfficientTAM &']
    for (condition, _), prefix in zip(CONDITIONS, labels):
        matching = [line for line in text.splitlines() if line.startswith(prefix)]
        assert len(matching) == 1
        values = []
        for key in ['domv_best', 'krv_latest_constant']:
            k = condition+'/'+key
            assert result['counts'][k] == target
            values.append(f"{100*result['successes'][k]/target:.2f}")
        label = f'EfficientTAM & ${sigma:.4f}$' if condition == 'efficienttam' else prefix
        text = text.replace(matching[0], label+' & '+' & '.join(values)+r' \\')
    (ROOT/'velocity_uncertainty_before_results.tex').write_text(original)
    (ROOT/'velocity_uncertainty_results.tex').write_text(text)
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--after-vu200', action='store_true',
                        help='Start after the explicitly requested 200-VU-trial pause for both models')
    parser.add_argument('--resume-interrupted', action='store_true', help='Retain scored trials and continue disjoint repeat tails')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        print(json.dumps({'jobs': len(prepare()['jobs']), 'plan': str(ROOT/'plan.json')}));return
    signal.signal(signal.SIGTERM, lambda *_: runner.STOP.set())
    signal.signal(signal.SIGINT, lambda *_: runner.STOP.set())
    with (ROOT/'runner.lock').open('a') as own:
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not runner.STOP.is_set():
            prerequisite = PREDECESSOR/('vu200-pause.json' if args.after_vu200 else 'completion.json')
            if prerequisite.exists():
                dependency = low.read(prerequisite)
                if dependency.get('verified'):
                    if args.after_vu200:
                        assert dependency['paused'] and dependency['vu_trials_each'] == 200
                    break
            status('waiting_for_vu200_pause' if args.after_vu200 else 'waiting_for_1800_each_completion', predecessor=str(prerequisite))
            runner.STOP.wait(30)
        if runner.STOP.is_set():return
        with (low.ROOT/'queue.lock').open('a') as gpu:
            while not runner.STOP.is_set():
                try:
                    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB);break
                except BlockingIOError:
                    runner.STOP.wait(5)
            if runner.STOP.is_set():return
            # Refresh prompts from the now-completed predecessor before freezing.
            plan = prepare()
            if args.resume_interrupted:
                original = copy.deepcopy(plan)
                retained = plan.setdefault('retained_jobs', [])
                pending = []
                for job in plan['jobs']:
                    folder = Path(job['folder'])
                    rows = records(folder/'trial_results.jsonl')
                    if not rows:
                        pending.append(job);continue
                    assert len(rows) <= job['repeats']
                    assert [r['trial'] for r in rows] == list(range(1, len(rows)+1))
                    kept = copy.deepcopy(job)
                    kept.update(repeats=len(rows), retained_interrupted_block=True,
                                retained_results_sha256=low.sha(folder/'trial_results.jsonl'))
                    assert len(list((folder/'velocity_diagnostics').glob('repeat*.json'))) == len(rows)
                    retained.append(kept)
                    if len(rows) < job['repeats']:
                        tail = copy.deepcopy(job)
                        tail['repeat_offset'] += len(rows)
                        tail['repeats'] -= len(rows)
                        tail['id'] += f"-resume-from-{tail['repeat_offset']+1:02d}"
                        tail['folder'] = str(ROOT/'blocks'/tail['id'])
                        Path(tail['folder']).mkdir(exist_ok=True, parents=True)
                        (Path(tail['folder'])/'case.txt').write_text(tail['scene']+'\n')
                        pending.append(tail)
                assert sum(j['repeats'] for j in pending+retained) == trial_target(plan)*len(plan['models'])*len(CONDITIONS)
                plan['jobs'] = pending
                archive = ROOT/f"recovery-before-resume-{runner.stamp().replace(':', '-')}.json"
                low.write(archive, original)
                plan['protocol']['reporting_recovery'] = 'Retain completed trial records; resume at next unexecuted global repeat in a fresh runtime/policy RNG session'
                for name in ['run_velocity_uncertainty_cr.py', 'velocity_uncertainty_server.py', 'velocity_uncertainty_sensor.py', 'dom_krv_scene_worker.py']:
                    path = low.REPO/'workstation'/name
                    plan['files_sha256'][str(path)] = low.sha(path)
            plan['protocol']['prerequisite'] = str(prerequisite)
            plan['jobs'] = grouped_jobs(plan['jobs'], plan['tests'])
            plan['protocol']['execution_order'] = 'scene -> DOMV then KRV -> clean,0.01,0.05,0.10,EfficientTAM -> repeat blocks; completed trials retained'
            plan['protocol']['start_boundary'] = 'both models paused at 200 VU trials' if args.after_vu200 else '1800 trials each complete'
            for job in plan['jobs']:
                scene = job['scene']
                for phase in ['krv_epoch4', 'dom']:
                    for row in records(PREDECESSOR/phase/'blocks'/Path(scene).stem/'prompt_trace.jsonl'):
                        plan['saved_prompts'].setdefault(scene, {})[str(row['global_trial'])] = row['text']
            assert all(all(str(i) in plan['saved_prompts'][t['path']] for i in range(1, 21)) for t in plan['tests'])
            low.write(ROOT/'plan.json', plan)
            low.write(ROOT/'protocol.json', plan['protocol'])
            assert low.read(low.WORK/'perception/efficienttam-cpu-readiness.json')['passed']
            runner.ROOT = ROOT;runner.status = status;runner.verify = verify
            code = inspect.getsource(runner.launch)
            code = code.replace('closed_loop_pilot_server.py', 'velocity_uncertainty_server.py')
            code = code.replace('closed_loop_pilot_client.py', 'dom_krv_scene_client.py')
            code = code.replace('total=75', "total=len(plan['jobs'])").replace('completed=75', "completed=len(plan['jobs'])")
            runner.check_simulator_error = check_simulator_error
            code = code.replace("                if STOP.wait(5): raise InterruptedError('Pilot stopped')", "                if STOP.wait(5): raise InterruptedError('Pilot stopped')\n                check_simulator_error()")
            exec(compile(code, __file__, 'exec'), runner.__dict__)
            try:
                runner.launch(plan)
            except BaseException as exc:
                status('failed', error=repr(exc));raise


if __name__ == '__main__':
    main()
