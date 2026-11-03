"""Read-only DeltaAI scheduler and authorized submission workflow evidence."""
import csv,io,json,subprocess
from pathlib import Path
def refresh_audit(run):
    """Read audit evidence; scheduler completion alone never establishes clean data."""
    if not run.get('audit_results'): return
    root=Path(run['audit_results'])
    reports=[json.loads(p.read_text()) for p in sorted(root.glob('status-*.json'))]
    summary={'shards_seen':len(reports),'shards_complete':sum(r['state']=='complete' for r in reports),
             'completed_episodes':sum(r['completed_episodes'] for r in reports),
             'failed_episodes':sum(r['failed_episodes'] for r in reports),
             'frames_decoded':sum(r['frames_decoded'] for r in reports),
             'metadata_errors':sorted({x for r in reports for x in r['metadata_errors']}),
             'warnings':sorted({x for r in reports for x in r['warnings']})}
    summary['all_episodes_checked']=summary['shards_complete']==run['audit_expected_shards'] and summary['completed_episodes']==run['audit_expected_episodes']
    summary['clean_for_checked_criteria']=summary['all_episodes_checked'] and not summary['failed_episodes'] and not summary['metadata_errors']
    run['audit_summary']=summary
    raw=subprocess.check_output(['sacct','-X','-n','-j',str(run['job_id']),'-P','--format=JobID,State'],universal_newlines=True)
    states=[line.split('|')[1] for line in raw.splitlines() if '|' in line]
    run['audit_scheduler_states']=states
    if states:
        if any(s.startswith(('FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')) for s in states):run['scheduler_state']='AUDIT ALLOCATION ERROR'
        elif any(s=='RUNNING' for s in states):run['scheduler_state']='RUNNING'
        elif any(s=='PENDING' for s in states):run['scheduler_state']='PENDING'
        elif any(s.startswith('CANCELLED') for s in states):run['scheduler_state']='CANCELLED'
        elif all(s=='COMPLETED' for s in states):run['scheduler_state']='COMPLETED'
    run.setdefault('audit_note',run.get('note',''))
    run['note']=run['audit_note']+' Checked '+str(summary['completed_episodes'])+'/'+str(run['audit_expected_episodes'])+' episodes; '+str(summary['failed_episodes'])+' failed checks. '+('Full sweep complete.' if summary['all_episodes_checked'] else 'Full sweep not complete.')

workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-ordered-domv-20261014/status.json')
scale=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-scale-domv-20261014/status.json')
submission=scale.parent/'production-submission.json'
if scale.exists() and submission.exists():
    scale_status=json.loads(scale.read_text());scale_receipt=json.loads(submission.read_text())
    if scale_receipt.get('job_id') and scale_status.get('training_job')==scale_receipt['job_id']:
        workflow=scale
if not workflow.exists():workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-isolated-domv-20261014/status.json')
if not workflow.exists():workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-existing-data-domv-20261012/status.json')
if not workflow.exists():workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-video-clean-domv-20261012/status.json')
if not workflow.exists():workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-bnfix-domv-20261011/status.json')
if not workflow.exists():workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-control-domv-20261011/status.json')
if not workflow.exists():workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-lean-domv-20261010/status.json')
if not workflow.exists():workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-profiled-domv-20261009/status.json')
for run in REQUEST:
    run.setdefault('_refresh_original_job',run.get('job_id'))
    if run.get('name')=='domv100' and run.get('group')=='production' and not run.get('superseded') and workflow.exists():
        status=json.loads(workflow.read_text())
        if 'setup-scale' in str(workflow) and run.get('job_id')!=status.get('training_job'):
            previous=dict(run);previous.update(group='historical',superseded=True,
                _refresh_original_job='prior:'+str(run.get('job_id')),
                note='Superseded before training by approved two-node production '+str(status.get('training_job')))
            REQUEST.append(previous)
        run['workflow_status']=status
        run['workflow_status_file']=str(workflow)
        run['job_id']=status.get('training_job')
        run['dependency']=None
        run['scheduler_state']='Not submitted'
        root=Path(status.get('family','/work/nvme/bivv/aparameshwaran/dynamicvla/domv-profiled-ddp-20261009'))
        run['family']=str(root);run['run_directory']=str(root/'runs/domv100')
        if (root/'manifest.json').exists():
            run['manifest_fingerprint']=json.loads((root/'manifest.json').read_text())['fingerprint']
            run['source_sha256']=json.loads((root/'manifest.json').read_text())['source_sha256']
        if (root/'recipe.json').exists():
            run['recipe_fingerprint']=json.loads((root/'recipe.json').read_text())['fingerprint']
        if (root/'selected-runtime.json').exists():run['selected_runtime']=json.loads((root/'selected-runtime.json').read_text())
        if run['job_id']:
            run['stdout']=str(root/('logs/train-'+run['job_id']+'.out'))
            run['stderr']=str(root/('logs/train-'+run['job_id']+'.err'))
        if 'setup-scale' in str(workflow):
            run['world_size']=status['selected_runtime']['world_size']
            run['gpu']='GH200, '+str(run['world_size']//4)+' nodes'
            run['note']='Two-node approval: actual200 updates/70400 samples,8 checkpoint saves,post-training validation,exact checkpoint/RNG reload,15percentGPU and per-nodeRAM headroom. Global352 retained. Completed804900-file audit and measured runtime reused; no dependency or repeated profiling. Model/math source and scientific settings retained. Stage: '+status['stage']
            run['training_health']='Awaiting allocation after successful two-node check' if status['stage']=='production_submitted' else 'Production startup or continuation'
        elif 'setup-ordered' in str(workflow):
            run['note']=('Production held for user-requested worst-case interactive stress/speed profiling '+str(status.get('stress_profile_job'))+'. Conservative approval3381007 remains preserved; no production updates.' if status.get('production_held') else 'Production after successful interactive3381007: four GH200s, microbatch22/workers1/accumulation4/global352. Actual18-update full-data test and checkpoint/RNG reload passed. Verified804900-file audit and measured runtime reused; no repeated readiness or data sweep. Source and scientific settings retained; native library cause unconfirmed.')
        elif 'setup-isolated' in str(workflow):
            run['note']='Isolated recovery stage: '+status['stage']+'. Verified804900 file scan reused. Microbatch22/workers1 candidate; checkpoint smoke and full-data benchmark run in separate processes within15minutes total, then unchanged training in the same48hour four-GH200 allocation. Native crash cause remains unconfirmed. Check failure blocks training.'
        elif status.get('combined_allocation'):
            run['name']='domv100';run['note']='Combined allocation stage: '+status['stage']+'. Four GH200s/one node/48h including startup; byte verification, one readiness attempt capped at15minutes, then training in the same allocation. No dependency or separate startup queue waits. Check failure blocks optimizer training.'
        else:
            run['note']='Preparation stage: '+status['stage']+'. Existing DeltaAI dataset is being verified against the completed Palmetto audit; all nine invalid training episodes are excluded in the new family. Fresh production submission has no Slurm dependency; checksum and bounded readiness must pass first.' if 'setup-existing-data' in str(workflow) else 'Profiling adaptation stage: '+status['stage']+'. Old job 3348823 cancelled. Fresh production submission has no Slurm dependency.'
        if status.get('stage')=='combined_preparation_failed' and run.get('native_failure_signal'):
            run['note']+=' Confirmed failure: '+run['native_failure_signal']+' on rank0 during '+run.get('failure_phase','readiness')+'; all804900 file checks and four-rank checkpoint/RNG smoke passed. Production never began; underlying native library not identified.'
        verification_job=status.get('verification_job')
        if verification_job and not any(str(r.get('job_id'))==verification_job for r in REQUEST):
            REQUEST.append({'cluster':'DeltaAI','job_id':verification_job,'name':'dom-existing-verify','group':'support','world_size':1,'gpu':'GH200 reserved; CPU checksum work','purpose':'Freeze final nine exclusions and verify existing bytes against the completed Palmetto decode audit','note':'No dataset archive transfer or repeated video decode sweep; checksum failure blocks training.'})
        if status.get('readiness_total_limit_minutes') and not status.get('readiness_reused'):
            run['note']+=' Bounded readiness: '+str(status['readiness_candidate_limit_minutes'])+' minutes per attempt, '+str(status['readiness_total_limit_minutes'])+' minutes total; stop after first safe configuration.'
        profile_job=status.get('profile_job')
        if profile_job and not any(str(r.get('job_id'))==profile_job for r in REQUEST):
            REQUEST.append({'cluster':'DeltaAI','job_id':profile_job,'name':'domv-runtime-4gh200','group':'support',
                'world_size':status.get('selected_runtime',{}).get('world_size',4),'gpu':'GH200','purpose':'Full-data distributed runtime comparison and checkpoint readiness',
                'note':'Select measured runtime before submitting dependency-free DOM+velocity production.'})
if workflow.exists():
    status=json.loads(workflow.read_text())
    for run in REQUEST:
        if str(run.get('job_id'))==status.get('profile_job') and status.get('readiness_complete') and 'setup-ordered' in str(workflow):
            run['purpose']='Completed four-GPU interactive DOM+velocity readiness'
            run['note']='Interactive3381007 passed18 full-data optimizer updates/6336 samples, checkpoint/RNG reload and memory checks.75.46 frames/s; microbatch22/workers1. Allocation released after9m55s. Production reuses approval.'
        elif str(run.get('job_id'))==status.get('profile_job') and status.get('readiness_total_limit_minutes'):
            run['purpose']='Bounded four-GPU DOM+velocity readiness'
            run['note']=str(status['readiness_candidate_limit_minutes'])+' minutes per candidate; '+str(status['readiness_total_limit_minutes'])+' minutes total; full-data, checkpoint and RNG verification required. Single-GPU measurements preserved.'
# The four-node phase-gate job is a separate experiment, not a DOM replacement.
for run in REQUEST:
    if run.get('name')!='krvp100' or run.get('group')!='production' or run.get('superseded'):continue
    phase_setup=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-krvp-4node-20261014')
    phase_status=phase_setup/'status.json'
    if not phase_status.exists():continue
    phase=json.loads(phase_status.read_text())
    receipt=json.loads((phase_setup/'production-submission.json').read_text())
    assert str(receipt['job_id'])==str(run['job_id'])==str(phase['training_job'])
    phase_root=Path(phase['family']);manifest=json.loads((phase_root/'manifest.json').read_text())
    recipe=json.loads((phase_root/'recipe.json').read_text())
    run.update(workflow_status=phase,workflow_status_file=str(phase_status),
        family=str(phase_root),run_directory=str(phase_root/'runs/krvp100'),
        manifest_fingerprint=manifest['fingerprint'],source_sha256=manifest['source_sha256'],
        recipe_fingerprint=recipe['fingerprint'],world_size=16,gpu='GH200, 4 nodes',dependency=None,
        stdout=str(phase_root/('logs/train-'+str(run['job_id'])+'.out')),
        stderr=str(phase_root/('logs/train-'+str(run['job_id'])+'.err')),
        readiness_complete=phase.get('readiness_complete',False),
        note='Separate KRVLA coordinate+velocity+causal phase gate;16GH200s/4nodes;microbatch22/workers4/accumulation1/global352.100epoch cap,LR3e-5,seed42,horizon20,original pretrained initialization,matched split/normalization and validation stopping. One15-minute bounded16-rank smoke/checkpoint/RNG plus200-update full-data check, then training in same48h allocation; no dependency or repeated data scan. Stage: '+phase['stage'])
    if (phase_root/'selected-runtime.json').exists():
        run['selected_runtime']=json.loads((phase_root/'selected-runtime.json').read_text())
    checks=list(phase_setup.glob('scale-check-*-16.json'))
    if checks:
        newest=max(checks,key=lambda p:p.stat().st_mtime)
        check=json.loads(newest.read_text())
        run['readiness_receipt']=str(newest)
        run['readiness_summary']={k:check[k] for k in ['job_id','complete','passed','error','smoke_exit','benchmark_exit','elapsed_seconds'] if k in check}
    run['training_health']='Queued; actual16-GPU readiness not yet run' if phase['stage']=='production_submitted_pending_readiness' else phase['stage']

# Operational stopping is separate from the frozen100-epoch scheduler identity.
for run in REQUEST:
    if run.get('group')!='production' or run.get('superseded') or run.get('name') not in ('domv100','krvp100'):continue
    phase_status=run.get('workflow_status',{})
    cap=phase_status.get('operational_epoch_cap')
    if cap is not None:
        run.update(target_epochs=cap,operational_epoch_cap=cap,
            scheduler_horizon_epochs=phase_status.get('scheduler_horizon_epochs'),
            operational_epoch_control_file=phase_status.get('operational_epoch_control_file'),
            operational_epoch_control_fingerprint=phase_status.get('operational_epoch_control_fingerprint'))
        run['note']=run.get('note','').replace('100epoch cap','10total-epoch stopping cap')
        run['note']+=' Effective stopping cap '+str(cap)+' total completed epochs, including restored progress; original100-epoch LR schedule and checkpoint identity retained for exact continuation. Validation stopping remains enabled.'

ids=','.join(str(r['job_id']) for r in REQUEST if r.get('job_id'))
raw=subprocess.check_output(['sacct','-X','-j',ids,'-P','--format=JobIDRaw,JobName%50,State,Elapsed,ReqTRES%100'],universal_newlines=True)
accounting={r['JobIDRaw']:r for r in csv.DictReader(io.StringIO(raw),delimiter='|')}
for run in REQUEST:
    row=accounting.get(str(run.get('job_id')))
    if row:run.update(scheduler_state=row['State'],elapsed=row['Elapsed'],accounting=row)
    directory=run.get('run_directory')
    if directory:
        root=Path(directory)
        for name in ('latest.json','best.json','final.json','training_complete.json'):
            path=root/name
            if path.is_file():run[name]=json.loads(path.read_text())
        provenance=root/'provenance.json'
        if provenance.is_file():
            tracker_id=json.loads(provenance.read_text()).get('wandb_id')
            if run.get('wandb_mode')=='offline':
                run['wandb_offline_id']=tracker_id
                run.pop('wandb_id',None);run.pop('wandb_url',None)
            else:
                run['wandb_id']=tracker_id
                if run['wandb_id']:run['wandb_url']='https://wandb.ai/aparame-clemson-university/Kinematic-Residual-VLA/runs/'+run['wandb_id']
    if run.get('name')=='krvp100' and run.get('group')=='production':
        if run.get('training_complete.json'):run['training_health']='Training completion receipt recorded'
        elif run.get('wandb_id') or run.get('latest.json'):run['training_health']='Production initialized; inspect scheduler and checkpoint progress'
    if run.get('name')=='domv100' and run.get('group')=='production' and not run.get('superseded'):
        if run.get('training_complete.json'):run['training_health']='Training completion receipt recorded'
        elif run.get('latest.json') and run.get('scheduler_state')=='PENDING':run['training_health']='Checkpointed continuation awaiting allocation'
        elif run.get('latest.json') and run.get('scheduler_state')=='RUNNING':run['training_health']='Production initialized; inspect checkpoint and logged progress'
    refresh_audit(run)
print(json.dumps(REQUEST))
