#!/usr/bin/env python3
"""Select a safe two-A6000 runtime using warm, real optimizer throughput.

Invoked by prepare_krv100_training.py after evaluation, full data and W&B gates.
Each trial starts from the released weights and discards its optimizer updates.
"""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys


def runtime_arguments(runtime):
    return ['--microbatch',str(runtime['microbatch']),'--workers',str(runtime['workers']),
            '--prefetch',str(runtime['prefetch']),
            '--pin-memory' if runtime['pin_memory'] else '--no-pin-memory',
            '--fused' if runtime['fused'] else '--no-fused']


def execute(command, source, log, timeout=1800):
    with log.open('a') as stream:
        process = subprocess.Popen([str(value) for value in command], cwd=source,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124
        finally:
            # torchrun's children must also be gone before the next trial.
            try:
                os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def search_runtimes(search, trial):
    base = {'microbatch':2,'workers':0,'prefetch':2,'pin_memory':True,'fused':True,'precision':'bf16'}
    compute = []
    for microbatch in search['microbatches']:
        runtime = {**base,'microbatch':microbatch}
        result = trial(runtime,'gpu_only','compute_capacity')
        if result:
            compute.append(result)
        else:
            # Ascending memory sweep: avoid repeated larger OOM attempts.
            break
    if not compute:
        raise RuntimeError('No compute configuration passed the memory headroom requirement')
    compute.sort(key=lambda row:row['samples_per_second'],reverse=True)
    candidates = []
    for measured in compute[:2]:
        for workers in search['workers']:
            runtime = {**measured['runtime'],'workers':workers}
            result = trial(runtime,'end_to_end','data_loading')
            if result:
                candidates.append(result)
    if not candidates:
        raise RuntimeError('No end-to-end runtime passed profiling')
    best = max(candidates,key=lambda row:row['samples_per_second'])['runtime']
    refinements = [{**best,'pin_memory':False},{**best,'fused':False}]
    if best['workers']:
        refinements.append({**best,'prefetch':4})
    for runtime in refinements:
        result = trial(runtime,'end_to_end','runtime_refinement')
        if result:
            candidates.append(result)
    candidates.sort(key=lambda row:row['samples_per_second'],reverse=True)
    confirmed = []
    for candidate in candidates[:3]:
        repeats = [trial(candidate['runtime'],'end_to_end','confirmation',updates=10,repeat=index)
                   for index in range(2)]
        if all(repeats):
            confirmed.append({'runtime':candidate['runtime'],'repeats':repeats,
                              'samples_per_second':statistics.mean(row['samples_per_second'] for row in repeats)})
    if not confirmed:
        raise RuntimeError('No runtime passed repeated steady-state confirmation')
    return confirmed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family-root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-selection',type=Path,
                        help='Reconfirm the measured runtime after an approved family revision')
    args = parser.parse_args()
    source, root, output = args.source.resolve(),args.family_root.resolve(),args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source))
    from utils.dom_experiment import atomic_json, digest, file_digest
    from core.dom_training import load_family
    manifest = load_family(root)
    recipe = json.loads((root/'recipe.json').read_text())
    if recipe['precision'] != 'bf16' or recipe['effective_batch'] != 352:
        raise ValueError('Profiler requires the authorized BF16, batch-352 recipe')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1':
        raise ValueError('Profiler requires both physical workstation GPUs (0,1)')
    identity = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version',
                                       '--format=csv,noheader,nounits'],text=True).strip()
    search = {'version':1,'manifest':manifest['fingerprint'],'gpu_identity':identity,
              'host_memory_limit_mib':int(os.environ['SLURM_MEM_PER_NODE']),
              'worker_sha256':file_digest(source/'core/dom_distributed_profiling.py'),
              'supervisor_sha256':file_digest(Path(__file__)),
              'microbatches':[2,4,8,16,32,44,88,176],'workers':[0,1,2,4,8],
              'warmup_updates':2,'search_updates':5,'confirmation_updates':10,
              'confirmation_repeats':2,'headroom_fraction':.15}
    search['metadata_adapter_sha256'] = file_digest(Path(__file__).with_name('training_metadata.py'))
    search['worker_launcher_sha256'] = file_digest(Path(__file__).with_name('profile_krv100_worker.py'))
    prior_selected = None
    if args.prior_selection:
        prior_selected = json.loads(args.prior_selection.read_text())
        checked = dict(prior_selected); expected = checked.pop('fingerprint')
        old_search = json.loads(args.prior_selection.with_name('search.json').read_text())
        checked_search = dict(old_search); old_id = checked_search.pop('fingerprint')
        prior_root = args.prior_selection.parent.parent/'family'
        prior_manifest = load_family(prior_root)
        derivation = json.loads((root/'revision_provenance.json').read_text())
        if (digest(checked)!=expected or digest(checked_search)!=old_id
                or not prior_selected.get('complete') or not prior_selected.get('safe')
                or not prior_selected.get('correctness_checks_passed')
                or prior_selected['search_fingerprint']!=old_id
                or prior_selected['manifest']!=prior_manifest['fingerprint']
                or derivation['prior_manifest']!=prior_manifest['fingerprint']
                or prior_manifest['source_sha256']!=manifest['source_sha256']
                or prior_manifest['weights_sha256']!=manifest['weights_sha256']
                or prior_selected['world_size']!=2 or prior_selected['variant']!='krv100'
                or any(search[key]!=old_search[key] for key in
                       ('gpu_identity','host_memory_limit_mib','worker_sha256','warmup_updates',
                        'confirmation_updates','confirmation_repeats','headroom_fraction'))):
            raise ValueError('Prior runtime cannot be reused for this source, hardware or revised family')
        prior_recipe = json.loads((prior_root/'recipe.json').read_text())
        if any(recipe[key]!=prior_recipe[key] for key in
               ('effective_batch','precision','lr','sampling','seed','epochs','early_stopping')):
            raise ValueError('Reconfirmation requires the same scientific recipe')
        search.update(version=2, prior_selection_sha256=file_digest(args.prior_selection),
                      prior_search_fingerprint=old_id, revalidation_only=True,
                      selected_runtime=prior_selected['runtime'])
    search_id = digest(search)
    search_path = output/'search.json'
    # Measurements retain the identity of their actual producer. An explicitly
    # recorded controller-only repair may resume them when every data, worker,
    # hardware and measurement-protocol input remains identical.
    if search_path.exists():
        previous = json.loads(search_path.read_text())
        previous_id = previous.pop('fingerprint')
        if digest(previous) != previous_id:
            raise ValueError('Saved profiling search provenance was modified')
        if previous != search:
            repair = json.loads((output/'controller_resume.json').read_text())
            current_inputs = {k:v for k,v in search.items() if k!='supervisor_sha256'}
            previous_inputs = {k:v for k,v in previous.items() if k!='supervisor_sha256'}
            if (current_inputs != previous_inputs or not repair.get('controller_only')
                    or repair['search_fingerprint'] != previous_id
                    or repair['previous_supervisor_sha256'] != previous['supervisor_sha256']
                    or repair['current_supervisor_sha256'] != search['supervisor_sha256']
                    or repair['readiness_adapter_sha256'] != file_digest(Path(__file__).with_name('readiness_krv100.py'))):
                raise ValueError('Controller repair cannot reuse these profiling measurements')
            search, search_id = previous, previous_id
    controller = {'supervisor_sha256':file_digest(Path(__file__)),
                  'readiness_adapter_sha256':file_digest(Path(__file__).with_name('readiness_krv100.py')),
                  'measurement_search_fingerprint':search_id,
                  'measurement_producer_sha256':search['supervisor_sha256']}
    controller.update(metadata_adapter_sha256=search['metadata_adapter_sha256'],
                      worker_launcher_sha256=search['worker_launcher_sha256'])
    atomic_json(output/'execution_controller.json',controller)
    selected_path = output/'selected_runtime.json'
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())
        check = dict(selected);fingerprint = check.pop('fingerprint')
        if (digest(check)!=fingerprint or selected.get('search_fingerprint')!=search_id
                or not selected.get('complete') or not selected.get('safe')):
            raise ValueError('Existing profiling selection has different provenance; use a fresh output folder')
        print(json.dumps({'phase':'profiling_already_complete','runtime':selected['runtime']}),flush=True)
        return
    atomic_json(output/'search.json',{**search,'fingerprint':search_id})
    rows = []
    reused_trials = 0

    def report(phase, **details):
        state = {'phase':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'search_fingerprint':search_id,'trials_recorded':len(rows),
                 'cached_trials_reused':reused_trials,**details}
        atomic_json(output/'profiling_status.json',state)
        print(json.dumps(state),flush=True)
        fields = ['trial','stage','complete','safe','microbatch','workers','prefetch','pin_memory','fused',
                  'samples_per_second','median_update_latency_s','data_wait_fraction',
                  'gpu0_util_percent','gpu1_util_percent','peak_gpu_gib','peak_ram_gib','failure']
        with (output/'comparison.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for row in rows:
                metrics = row.get('result',{})
                ranks = metrics.get('ranks',[])
                flat = {'trial':row['trial'],'stage':row['stage'],'complete':metrics.get('complete',False),
                        'safe':metrics.get('safe',False),**row['runtime'],'failure':row.get('failure','')}
                flat.pop('precision',None)
                for key in ('samples_per_second','median_update_latency_s','data_wait_fraction'):
                    flat[key] = metrics.get(key,'')
                if ranks:
                    for rank in ranks:
                        flat[f"gpu{rank['rank']}_util_percent"] = rank['telemetry']['mean_gpu_utilization_percent']
                    flat['peak_gpu_gib'] = max(rank['telemetry']['peak_gpu_bytes'] for rank in ranks)/1024**3
                    flat['peak_ram_gib'] = sum(rank['telemetry']['peak_ram_bytes'] for rank in ranks)/1024**3
                writer.writerow(flat)
        return state

    def trial(runtime, mode, stage, updates=5, repeat=0, trace=False):
        nonlocal reused_trials
        definition = {'search':search_id,'runtime':runtime,'mode':mode,'stage':stage,
                      'updates':updates,'repeat':repeat,'trace':trace}
        trial_id = digest(definition)[:20]
        # Vary frame order instead of repeatedly measuring videos warmed in
        # the OS file cache. Use the production uniform sampler/augmentation.
        epoch = int(trial_id[:8],16) if mode=='end_to_end' else 0
        directory = output/'trials'/trial_id
        directory.mkdir(parents=True,exist_ok=True)
        result_path, log = directory/'measurement.json',directory/'trial.log'
        row_path = directory/'trial.json'
        report(stage,trial=trial_id,runtime=runtime)
        if row_path.exists():
            row = json.loads(row_path.read_text())
            if row['definition'] != definition:
                raise ValueError('Cached trial provenance mismatch')
            if row.get('result'):
                check=dict(row['result']);expected=check.pop('fingerprint')
                if digest(check)!=expected:
                    raise ValueError('Cached performance measurement was modified')
            if row.get('failure')=='unexpected_error':
                raise RuntimeError(f'Prior correctness/model/data failure; inspect {log}')
            reused_trials += 1
        else:
            command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                       Path(__file__).with_name('profile_krv100_worker.py'),'--source',source,'--family-root',root,
                       '--runtime',json.dumps(runtime),'--mode',mode,'--output',result_path,
                       '--warmup','2','--updates',str(updates),'--epoch',str(epoch)]
            if trace:
                command.append('--trace')
            code = execute(command,source,log)
            row = {'trial':trial_id,'definition':definition,'stage':stage,'runtime':runtime,
                   'exit_code':code,'log':str(log),
                   'producer_supervisor_sha256':file_digest(Path(__file__))}
            if code == 0:
                result = json.loads(result_path.read_text())
                check = dict(result);expected = check.pop('fingerprint')
                if (digest(check)!=expected or not result['complete'] or result['runtime']!=runtime
                        or result['manifest']!=manifest['fingerprint'] or result['mode']!=mode
                        or result['sampling_epoch']!=epoch):
                    raise ValueError('Invalid profiling measurement')
                row['result'] = result
                if not result['safe']:
                    row['failure'] = 'memory_headroom'
            else:
                tail = log.read_text(errors='replace')[-200000:].lower()
                capacity = ('out of memory','cannot allocate memory','insufficient shared memory',
                            'bus error','dataloader timed out','no space left on device')
                row['failure'] = ('timeout' if code==124 else
                                  'capacity_or_loader' if any(term in tail for term in capacity)
                                  else 'unexpected_error')
            atomic_json(row_path,row)
        rows.append(row)
        report(stage,trial=trial_id,last_failure=row.get('failure'))
        if row.get('failure') == 'unexpected_error':
            raise RuntimeError(f'Profiling found a model/data/correctness error; inspect {log}')
        return row.get('result') if not row.get('failure') else None

    if prior_selected:
        repeats = [trial(prior_selected['runtime'],'end_to_end','reconfirmation',updates=10,repeat=index)
                   for index in range(2)]
        if not all(repeats):
            raise RuntimeError('Previously selected runtime failed revised-family reconfirmation')
        confirmed = [{'runtime':prior_selected['runtime'],'repeats':repeats,
                      'samples_per_second':statistics.mean(row['samples_per_second'] for row in repeats)}]
    else:
        confirmed = search_runtimes(search, trial)
    winner = None
    for candidate in sorted(confirmed,key=lambda row:row['samples_per_second'],reverse=True):
        report('correctness_checks',runtime=candidate['runtime'])
        command = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                   Path(__file__).with_name('readiness_krv100.py'),'--source',source,
                   '--family-root',root,'--runtime',json.dumps(candidate['runtime'])]
        log = output/f"readiness-{digest(candidate['runtime'])[:16]}.log"
        hardware = candidate['repeats'][0]['hardware_by_rank'][0]
        if prior_selected and candidate['repeats'][0]['hardware_by_rank']!=prior_selected['hardware_by_rank']:
            raise ValueError('Both GPU identities must match the prior runtime selection')
        readiness_key = digest({'hardware':hardware,'world_size':2,
                                'runtime':candidate['runtime'],'variant':'krv100'})[:16]
        readiness_path = root/'distributed-profiles'/f'{readiness_key}.json'
        if readiness_path.exists():
            readiness = json.loads(readiness_path.read_text())
            checked = dict(readiness);expected = checked.pop('fingerprint')
            if (digest(checked)==expected and readiness.get('complete') and readiness.get('safe')
                    and readiness.get('manifest')==manifest['fingerprint']
                    and readiness.get('runtime')==candidate['runtime'] and readiness.get('hardware')==hardware
                    and readiness.get('hardware_by_rank')==candidate['repeats'][0]['hardware_by_rank']
                    and readiness.get('world_size')==2 and readiness.get('variant')=='krv100'
                    and readiness.get('samples')==353 and readiness.get('optimizer_updates')==2
                    and all(readiness.get(k) for k in ('strict_reload','rank_rng_reload','future_inputs_ignored'))
                    and readiness.get('readiness_adapter_sha256')==controller['readiness_adapter_sha256']):
                report('correctness_checks',runtime=candidate['runtime'],reused_completed_readiness=True)
                winner = candidate
                break
        code = execute(command,source,log)
        if code==0:
            readiness = json.loads(readiness_path.read_text())
            checked = dict(readiness);expected = checked.pop('fingerprint')
            if (digest(checked)!=expected or not readiness.get('complete') or not readiness.get('safe')
                    or readiness.get('manifest')!=manifest['fingerprint']
                    or readiness.get('runtime')!=candidate['runtime']
                    or readiness.get('hardware_by_rank')!=candidate['repeats'][0]['hardware_by_rank']
                    or readiness.get('samples')!=353 or readiness.get('optimizer_updates')!=2
                    or not all(readiness.get(k) for k in ('strict_reload','rank_rng_reload','future_inputs_ignored'))
                    or file_digest(Path(__file__).with_name('readiness_krv100.py'))!=controller['readiness_adapter_sha256']):
                raise ValueError('Readiness result differs from the required complete checks')
            readiness['readiness_adapter_sha256'] = controller['readiness_adapter_sha256']
            readiness['readiness_log_sha256'] = file_digest(log)
            readiness.pop('fingerprint')
            readiness['fingerprint'] = digest(readiness)
            atomic_json(readiness_path,readiness)
            winner = candidate
            break
        tail = log.read_text(errors='replace')[-200000:].lower()
        if code!=124 and not any(term in tail for term in ('out of memory','memory headroom','cannot allocate memory')):
            raise RuntimeError(f'Selected runtime failed a correctness check; inspect {log}')
    if winner is None:
        raise RuntimeError('No confirmed runtime passed the complete readiness checks')
    # Existing timelines describe the unchanged source/runtime. Preserve their
    # actual old data identity; fresh timing and correctness use the new family.
    trace_provenance = None
    if prior_selected:
        traces = []
        for original in prior_selected['traces']:
            relative = Path(original).relative_to(root.parent/'profiling')
            archived = args.prior_selection.parent/relative
            if not archived.is_file():
                raise ValueError('Missing original profiling timeline')
            traces.append(str(archived.resolve()))
        trace_provenance = {'manifest':prior_selected['manifest'],
                            'prior_selection_sha256':file_digest(args.prior_selection),
                            'trace_sha256':{path:file_digest(path) for path in traces},
                            'reason':'Unchanged numerical source, model, runtime and hardware; fresh revised-family confirmations and readiness checks passed'}
    else:
        traced = trial(winner['runtime'],'end_to_end','timeline_trace',trace=True)
        if not traced:
            raise RuntimeError('Selected runtime did not pass the timeline trace run')
        traces = sorted(str(path) for path in (output/'trials'/rows[-1]['trial']/'traces').glob('rank-*.trace.json'))
    if len(traces)!=2:
        raise RuntimeError('Both rank timeline traces are required')
    selected = {'complete':True,'safe':True,'manifest':manifest['fingerprint'],'variant':'krv100',
                'world_size':2,'runtime':winner['runtime'],'search_fingerprint':search_id,
                'hardware_by_rank':winner['repeats'][0]['hardware_by_rank'],
                'samples_per_second':winner['samples_per_second'],'confirmation':winner['repeats'],
                'execution_controller':controller,
                'correctness_checks_passed':True,'traces':traces,
                'selection':'Highest mean warm end-to-end throughput over two ten-update confirmations',
                'scope':'Training updates; excludes validation/checkpoint I/O. Full epoch throughput can differ.'}
    if prior_selected:
        selected['selection'] = 'Previously selected runtime reconfirmed over two fresh ten-update measurements after approved dataset exclusions'
        selected['trace_provenance'] = trace_provenance
        selected['prior_selection_sha256'] = file_digest(args.prior_selection)
    # Save the human-readable report before publishing the training approval.
    lines = ['# KRVLA + Velocity profiling','',selected['selection']+'.','',
             'Effective batch 352, BF16, learning rate 3e-5 and uniform frame sampling remain fixed.',
             'Each trial starts from released weights; trial updates are discarded.',
             'GPU and host memory retain at least 15% measured headroom.','',
             '| Microbatch/GPU | Workers/GPU | Prefetch | Pin | Fused AdamW | Confirmed samples/s |',
             '|---:|---:|---:|:---:|:---:|---:|']
    for candidate in sorted(confirmed,key=lambda row:row['samples_per_second'],reverse=True):
        runtime = candidate['runtime']
        lines.append(f"| {runtime['microbatch']} | {runtime['workers']} | {runtime['prefetch']} | "
                     f"{runtime['pin_memory']} | {runtime['fused']} | {candidate['samples_per_second']:.3f} |")
    lines.extend(['',f"Selected runtime: `{json.dumps(winner['runtime'],sort_keys=True)}`.",'',
                  'See comparison.csv for both GPUs’ utilization, loader wait, throughput and peak memory.',
                  'Each trials/<id>/ directory contains the measurement and its process log.',
                  'The selected readiness check also passed optimizer, uneven tail, inference, checkpoint and RNG checks.',
                  'Timeline traces are separate from timing comparisons; open them in Perfetto.',
                  '',selected['scope'],''])
    (output/'profiling_report.md').write_text('\n'.join(lines))
    selected['fingerprint'] = digest(selected)
    atomic_json(selected_path,selected)
    report('complete',selected_runtime=winner['runtime'],samples_per_second=winner['samples_per_second'])


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if '--output' in sys.argv:
            status_path = Path(sys.argv[sys.argv.index('--output')+1])/'profiling_status.json'
            if status_path.exists():
                state = json.loads(status_path.read_text())
                state.update(failed_stage=state['phase'],phase='failed',error_type=type(error).__name__,
                             error=str(error),updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                temporary=status_path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(status_path)
        raise
