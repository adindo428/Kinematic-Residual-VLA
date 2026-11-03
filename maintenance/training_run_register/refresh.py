"""Refresh the local register from authenticated SSH; emit native Page blocks."""
from pathlib import Path
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import json,subprocess,os
ROOT=Path(__file__).resolve().parent


def main():
    path=ROOT/'registry.json';register=json.loads(path.read_text())
    socket=os.environ.get('KRVLA_PALMETTO_SOCKET','/private/tmp/vlcm-palmetto-k1vtw3i7/socket')
    script='REQUEST='+repr(register['runs'])+'\n'+(ROOT/'probe.py').read_text()+'\nprint(json.dumps(probe(REQUEST)))\n'
    cmd=['ssh','-F','/dev/null','-S',socket,'-o','BatchMode=yes','-o','ConnectTimeout=15',
         'aparame@slogin.palmetto.clemson.edu','python3 -']
    try:
        result=subprocess.run(cmd,input=script,text=True,capture_output=True,check=True,timeout=60)
        register['runs']=json.loads(result.stdout)
        register['last_successful_cluster_check']=datetime.now(timezone.utc).isoformat()
        register.pop('refresh_error',None)
    except (subprocess.SubprocessError,ValueError) as exc:
        register['refresh_error']='Palmetto refresh unavailable; last verified status retained. '+type(exc).__name__
    delta_runs=[r for r in register['runs'] if r.get('cluster')=='DeltaAI']
    ids=','.join(str(r['job_id']) for r in delta_runs if r.get('job_id'))
    delta_script='REQUEST='+repr(delta_runs)+'\n'+(ROOT/'delta_probe.py').read_text()
    try:
        delta=subprocess.run(['ssh','-F','/dev/null','-S',os.environ.get('KRVLA_DELTAAI_SOCKET','/tmp/deltaai-kinematic.sock'),
            '-o','BatchMode=yes','-o','ConnectTimeout=15','aparameshwaran@gh-login01.delta.ncsa.illinois.edu','python3 -'],
            input=delta_script,text=True,capture_output=True,check=True,timeout=45)
        refreshed={}
        for r in json.loads(delta.stdout):
            original_job=r.pop('_refresh_original_job',r.get('job_id'))
            refreshed[(r['cluster'],str(original_job))]=r
        matched=set()
        for r in delta_runs:
            key=(r['cluster'],str(r.get('job_id')))
            if key in refreshed:r.update(refreshed[key]);matched.add(key)
        register['runs'].extend(r for key,r in refreshed.items() if key not in matched)
        register['last_successful_deltaai_check']=datetime.now(timezone.utc).isoformat()
        register.pop('deltaai_refresh_error',None)
    except (subprocess.SubprocessError,ValueError) as exc:
        register['deltaai_refresh_error']='DeltaAI refresh unavailable; last verified status retained. '+type(exc).__name__
    ncsa_runs=[r for r in register['runs'] if r.get('cluster')=='Delta']
    if ncsa_runs:
        try:
            script='REQUEST='+repr(ncsa_runs)+'\n'+(ROOT/'ncsa_delta_probe.py').read_text()
            response=subprocess.run(['ssh','-F','/dev/null','-S',os.environ.get('KRVLA_DELTA_SOCKET','/tmp/delta-krvla.sock'),
                '-o','BatchMode=yes','-o','ConnectTimeout=15','aparameshwaran@login.delta.ncsa.illinois.edu','python3 -'],
                input=script,text=True,capture_output=True,check=True,timeout=45)
            refreshed={str(r['job_id']):r for r in json.loads(response.stdout)}
            for r in ncsa_runs:
                if str(r['job_id']) in refreshed:r.update(refreshed[str(r['job_id'])])
            register['last_successful_delta_check']=datetime.now(timezone.utc).isoformat()
            register.pop('delta_refresh_error',None)
        except (subprocess.SubprocessError,ValueError) as exc:
            register['delta_refresh_error']='Delta refresh unavailable; last verified evidence retained. '+type(exc).__name__
    # W&B is a separate evidence source; never replace scheduler/checkpoint state
    # with tracker state or assume logged updates have been checkpointed.
    tracked=[r for r in register['runs'] if r.get('wandb_id')]
    if tracked:
        identities=[dict(entity=r.get('wandb_entity','aparame-clemson-university'),
                         project=r.get('wandb_project','Kinematic-Residual-VLA'),run=r['wandb_id'])
                    for r in tracked]
        script='REQUEST='+repr(identities)+'\n'+(ROOT/'wandb_probe.py').read_text()
        try:
            response=subprocess.run(cmd,input=script,text=True,capture_output=True,check=True,timeout=90)
            evidence=json.loads(response.stdout)
            errors=[]
            for item in evidence:
                for r in tracked:
                    if (r.get('wandb_entity','aparame-clemson-university'),r.get('wandb_project','Kinematic-Residual-VLA'),r['wandb_id']) != (item['entity'],item['project'],item['run']):continue
                    if item.get('error'):
                        r['wandb_refresh_error']=item['error'];errors.append(item['run']);continue
                    r.pop('wandb_refresh_error',None)
                    r.update(wandb_state=item['state'],wandb_summary=item['summary'],
                             wandb_checked_at=item['checked_at'],wandb_config=item['config'],
                             wandb_url='https://wandb.ai/'+item['entity']+'/'+item['project']+'/runs/'+item['run'])
            if errors:register['wandb_refresh_error']='Some W&B reads unavailable; previous evidence retained: '+', '.join(errors)
            else:
                register.pop('wandb_refresh_error',None)
                register['last_successful_wandb_check']=datetime.now(timezone.utc).isoformat()
        except (subprocess.SubprocessError,ValueError) as exc:
            register['wandb_refresh_error']='W&B refresh unavailable; last verified metrics retained. '+type(exc).__name__
    register['rendered_at']=datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(register,indent=2)+'\n')
    blocks=render(register)
    (ROOT/'page-blocks.json').write_text(json.dumps(blocks,indent=2)+'\n')
    print(json.dumps({'registry':str(path),'last_successful_cluster_check':register.get('last_successful_cluster_check'),
        'refresh_error':register.get('refresh_error'),'run_count':len(register['runs']),'page_blocks':blocks}))


def cell(value):return str(value if value is not None else '—').replace('|','\\|').replace('\n',' ')
def table(columns,rows):
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(cell(x) for x in row)+' |' for row in rows)

def render(reg):
    stamp=reg.get('last_successful_cluster_check')
    stamp=datetime.fromisoformat(stamp).astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %I:%M %p %Z') if stamp else 'not yet checked'
    intro='DynamicVLA and KRVLA training submissions, including the current six-model DOM manipulation study, earlier tracking runs, and the separate DeltaAI DOM launch. Scheduler completion and checkpoint epochs are reported separately; neither implies task success.\n\nLast successful Palmetto check: **'+stamp+'**. Automatic refresh: hourly from this local task. Other agents can use this Page and the registry paths below; retain historical jobs when adding new submissions.'
    if reg.get('refresh_error'):intro+='\n\n'+reg['refresh_error']
    if reg.get('last_successful_deltaai_check'):
        delta_stamp=datetime.fromisoformat(reg['last_successful_deltaai_check']).astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %I:%M %p %Z')
        intro+='\n\nLast successful DeltaAI check: **'+delta_stamp+'**.'
    if reg.get('deltaai_refresh_error'):intro+='\n\n'+reg['deltaai_refresh_error']
    blocks=[intro,'## Current DOM manipulation study']
    rows=[]
    for r in reg['runs']:
        if str(r.get('scheduler_state','')).upper().startswith(('CANCELLED','CANCELED')):continue
        if r.get('group')!='production' or r.get('superseded'):continue
        progress=r.get('latest.json',{}).get('progress',{})
        local=r.get('cluster')=='Local workstation'
        epoch=progress.get('epoch','Not checked' if local else 0)
        rows.append([r['name'],r.get('method',r['name']),r.get('job_id') or (r.get('wandb_id') if local else 'Not submitted'),
            ('W&B '+r.get('wandb_state','unverified')) if local else r.get('scheduler_state','Not submitted'),
            r.get('world_size',8),r.get('gpu','H100 or H200')+(' (DeltaAI)' if r['cluster']=='DeltaAI' else ''),epoch,
            (str(r.get('wandb_summary',{}).get('Train/optimizer_updates','—'))+' logged updates; samples '+str(r.get('wandb_summary',{}).get('Train/samples','—'))) if r.get('wandb_id') else r.get('note','—')])
    blocks.append(table(['Run','Method','Job or W&B ID','Scheduler or W&B status','GPUs / ranks','GPU class','Checkpoint epochs','Logged progress'],rows))
    for r in reg['runs']:
        if str(r.get('scheduler_state','')).upper().startswith(('CANCELLED','CANCELED')):continue
        if r.get('cluster')!='Local workstation':continue
        summary=r.get('wandb_summary',{})
        checked=r.get('wandb_checked_at')
        checked=datetime.fromisoformat(checked).astimezone(ZoneInfo('America/New_York')).strftime('%Y-%m-%d %I:%M %p %Z') if checked else 'unverified'
        blocks.append('Local workstation '+r['name']+': [W&B run]('+r['wandb_url']+'). W&B state: '+r.get('wandb_state','unverified')+
            '; last checked '+checked+'. Logged updates: '+str(summary.get('Train/optimizer_updates','unavailable'))+
            '; samples: '+str(summary.get('Train/samples','unavailable'))+'; fractional epoch: '+str(summary.get('Train/epoch','unavailable'))+
            '; training loss: '+str(summary.get('Train/loss','unavailable'))+'. Checkpoint progress is not verified by W&B logs. '+r.get('note',''))
    blocks.append('DynamicVLA study settings: learning rate3e-5; the two DeltaAI runs now stop at10 total completed epochs, including restored progress; the original100-epoch LR schedule is retained for exact checkpoint continuation; other runs retain their separately recorded budgets; validation MSE stopping after at least 5 epochs and 3 successive validations without a 0.5% relative improvement; global batch 352; the planned Palmetto four-H100 alternative uses microbatch44/accumulation2 and four-A100 alternative microbatch22/accumulation4; DeltaAI uses the approved two-node/eight-GH200 runtime: microbatch44/workers4 per GPU, accumulation1/global352 for DOM+velocity; separate four-node/sixteen-GH200 KRVLA+velocity+causal-phase-gate uses microbatch22/workers4/accumulation1/global352, pending its own actual16-rank readiness; BF16; 20 predicted and executed actions; seed 42; full parameter fine tuning. Fixed validation anchors decide stopping. Best and final checkpoints retain optimizer state. Simulator task evaluation follows training on Palmetto; DeltaAI checkpoints are transferred for evaluation.\n\nThe final paired cluster split contains201224 episodes/22827975frames; training197198episodes/22370201frames. The original6073 unavailable episodes, six empty videos and three short videos are excluded. The user approved all nine invalid training-episode exclusions on October12; final split SHA bbeca518e7f6ee85c39e5863a4d6745c36de53f13b8831595ca36c304ff9950c matches across the final DeltaAI and Palmetto families. Validation2015/test2011 and initialization/normalization remain unchanged. DeltaAI now uses its existing dataset; archive transfer was cancelled. DeltaAI production3381947 reuses the completed804900-file audit and successful two-node/eight-GH200200-update readiness3381859; no repeated profiling or full scan. The48hour allocation retains checkpoint continuation. The archive-transfer automation remains paused. Other unsent variants must use the final matched split when launched.')
    delta=next((r for r in reg['runs'] if r.get('cluster')=='DeltaAI' and r.get('name')=='domv100' and not r.get('superseded')),None)
    if delta:
        workflow=delta.get('workflow_status',{})
        blocks.append('DeltaAI DOM + velocity: '+delta.get('note','Awaiting runtime selection')+' Current stage: `'+workflow.get('stage','unverified')+'`. Workflow status: `'+delta.get('workflow_status_file','not recorded')+'`. Runtime selection changes execution settings while retaining the paired scientific recipe.')
    for phase in reg['runs']:
        if phase.get('cluster')=='DeltaAI' and phase.get('name')=='krvp100' and phase.get('group')=='production' and not phase.get('superseded'):
            blocks.append('DeltaAI KRVLA + velocity + causal phase gate (`krvp100`): job '+str(phase['job_id'])+'; scheduler '+phase.get('scheduler_state','unverified')+'. '+phase.get('note','')+' Workflow: `'+phase.get('workflow_status_file','not recorded')+'`. Run directory: `'+phase.get('run_directory','not recorded')+'`. Source SHA256: `'+phase.get('source_sha256','unverified')+'`; manifest `'+phase.get('manifest_fingerprint','unverified')+'`; recipe `'+phase.get('recipe_fingerprint','unverified')+'`. Readiness state: '+str(phase.get('readiness_complete',False))+'. Scheduler state does not establish readiness or successful training.')
    blocks+=['## Other training submissions']
    rows=[]
    for r in reg['runs']:
        if str(r.get('scheduler_state','')).upper().startswith(('CANCELLED','CANCELED')):continue
        if r.get('group') not in ('historical','production') or (r.get('group')=='production' and not r.get('superseded')):continue
        rows.append([r['cluster'],r.get('job_id'),r['name'],r.get('scheduler_state','Unverified'),r.get('gpu','See accounting'),
            r.get('target_epochs','Unknown'),r.get('checkpoint_epoch','Not checked'),r.get('note','')])
    blocks.append(table(['Cluster','Job ID','Training run','Scheduler status','GPU request','Epoch target','Checkpoint epoch','Notes'],rows))
    blocks.append('Historical tracking recipes use their own 80/10/10 episode split and normalization. Keep them separate from the new DOM manipulation comparison. A FAILED scheduler status with a 75-epoch config is preserved as both facts; the config alone does not verify the matching optimizer or successful job exit.')
    blocks+=['## Preparation and profiling submissions']
    rows=[]
    for r in reg['runs']:
        if str(r.get('scheduler_state','')).upper().startswith(('CANCELLED','CANCELED')):continue
        if r.get('group')!='support':continue
        rows.append([r['cluster'],r['job_id'],r['name'],r.get('scheduler_state','Unverified'),r.get('purpose'),r.get('note','')])
    blocks.append(table(['Cluster','Job ID','Job','Scheduler status','Purpose','Notes'],rows))
    blocks+=['## Logs checkpoints and provenance']
    active=next(r for r in reg['runs'] if r.get('cluster')=='Palmetto' and r.get('group')=='production'
                and r.get('name')=='krv100' and not r.get('superseded'))
    blocks.append('Current Palmetto job: `'+str(active.get('job_id','Not submitted'))+'` (`krv100`).\n\n'+ '\n\n'.join(label+': `'+str(active.get(key,'Not yet created'))+'`' for label,key in [('Standard output','stdout'),('Errors','stderr'),('Run and checkpoint directory','run_directory'),('Frozen family','family')])+ '\n\nSource SHA256: `'+reg['active_source_sha256']+'`. Recipe fingerprint: `'+reg['active_recipe_fingerprint']+'`. The persistent W&B ID is added once training initializes; display name remains `krv100`. Detailed paths for historical checkpoints and scheduler records are in the local JSON registry.')
    blocks+=['## How agents maintain this register']
    blocks.append('Online register: [KRVLA training run register]('+reg.get('page_url','')+').' +'\n\nLocal registry: `'+str(ROOT/'registry.json')+'`. Refresh command: `python3 "'+str(ROOT/'refresh.py')+'"`. The command refreshes cluster evidence and writes `page-blocks.json`; it does not submit, cancel, or alter training.\n\nRead the Page and its current block hashes before updating online tables. Add each new submission to `registry.json` using cluster plus job ID as its identity, preserving previous records and notes. Record variant, GPU count and class, learning rate, epoch cap, stopping rule, source/family fingerprints, output/log paths, and W&B URL when available. Keep an explicit last-successful-check timestamp if SSH expires. Do not turn an unknown or stale status into a claimed success.\n\nThe local hourly heartbeat refreshes this Page. It stays quiet while states are unchanged and alerts on starts, completion, failure, or required action. This monitor never launches or cancels jobs.')
    return blocks

if __name__=='__main__':main()
