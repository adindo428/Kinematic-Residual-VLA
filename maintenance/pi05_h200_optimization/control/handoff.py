"""Separate scheduler-owned, allocated-H200-first exact-run handoff controller."""
import argparse,fcntl,gc,hashlib,json,os,shutil,statistics,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from handoff_guards import (NAMES,PARENTS,WANDB,CORE,PARENT_SOURCE,require,
    fingerprint,validate_pair,validate_allocation,parent_decision,
    validate_saved_stop,validate_checkpoint,migration_manifest)

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,path)

def read(path):return json.loads(Path(path).read_text())
def now():return datetime.now(timezone.utc).isoformat()

def command(args,check=True):
    p=subprocess.run(args,text=True,capture_output=True,timeout=60)
    if check and p.returncode:raise RuntimeError(str(args[:2])+': '+p.stderr[-1500:])
    return p

def accounting(jid):
    rows=command(['sacct','-X','-j',str(jid),'-n','-P','--format=JobIDRaw,State,ExitCode']).stdout.splitlines()
    rows=[s.split('|') for s in rows if s.split('|')[0]==str(jid)]
    require(len(rows)==1,'Unknown accounting identity')
    return {'state':rows[0][1],'exit_code':rows[0][2]}

def live_job(jid):
    p=command(['scontrol','show','job',str(jid),'-o'],check=False)
    if p.returncode:
        require('Invalid job id' in p.stderr,'Unknown Slurm access failure')
        return None
    d=dict(x.split('=',1) for x in p.stdout.split() if '=' in x)
    require(d.get('JobId')==str(jid),'Slurm identity differs');return d

def visible(jid):
    p=command(['squeue','-j',str(jid),'-h','-o','%i'],check=False)
    if p.returncode:
        require('Invalid job id' in p.stderr,'Unknown queue access failure');return False
    return bool(p.stdout.strip())

def allocation_key(j):return (j.get('StartTime'),j.get('Restarts'),j.get('BatchHost'))

def terminal_accounting(jid,deadline):
    while time.monotonic()<deadline:
        a=accounting(jid)
        if a['state'].startswith(('CANCELLED','COMPLETED')):return a
        require(a['state'] in ('RUNNING','COMPLETING','PENDING'),'Unexpected parent terminal state; reconcile')
        time.sleep(2)
    raise RuntimeError('Parent accounting did not become terminal; checkpoint preserved')

def verify_frozen(c):
    root=Path(c['root']);parent=Path(c['parent_root'])
    require(sha(root/'source-manifest.json')==c['source_manifest_sha256'],'Frozen source manifest changed')
    manifest=read(root/'source-manifest.json');require(manifest['source_sha256']==c['source_sha256'],'Wrong family source')
    require({p.name for p in root.glob('*.py')}==set(manifest['files']),'Submitted source file inventory changed')
    for n,h in manifest['files'].items():require(sha(root/n)==h,'Source changed: '+n)
    require(sha(root/'preparation-manifest.json')==c['preparation_manifest_sha256'],'Preparation inventory changed')
    for n,h in read(root/'preparation-manifest.json')['files'].items():require(sha(root/n)==h,'Preparation changed: '+n)
    for n,h in read(root/'implementation-lock.json')['files'].items():require(sha(n)==h,'Native model implementation changed')
    require((root/'env/bin/python').resolve()==(parent/'env/bin/python').resolve(),'Parent/target Python environments differ')
    for name in NAMES:
        spec=c['runs'][name]
        require(sha(root/'recipes'/f'{name}.json')==spec['recipe_sha256'],'Recipe changed')
        require(sha(spec['parent_script'])==spec['parent_script_sha256'],'Parent launch script bytes changed')
        require((root/'recipes'/f'{name}.json').read_bytes()==(parent/'recipes'/f'{name}.json').read_bytes(),'Parent recipe differs')
        require(sha(root/'profiles'/f'{name}.json')==spec['gpu_profile_sha256'],'GPU receipt changed')
    return manifest

def parent_identity(c,name):
    p=Path(c['parent_root']);out=p/'runs'/name;spec=c['runs'][name]
    prov=read(out/'provenance.json');wandb=read(out/'wandb.json')
    require(str(prov['job_id'])==PARENTS[name] and prov['identity']==spec['parent_identity'],'Wrong canonical parent identity')
    require(prov['hardware']=='NVIDIA A100 80GB PCIe' and prov['world_size']==2,'Wrong canonical parent hardware')
    require(fingerprint(prov['source'])==PARENT_SOURCE,'Parent source fingerprint differs')
    for n,h in prov['source'].items():require(sha(p/n)==h,'Parent source changed')
    require(wandb['id']==prov['wandb_id']==WANDB[name],'Wrong parent W&B identity')
    target=read(Path(c['root'])/'source-manifest.json')['files']
    require(all(prov['source'][n]==target[n] for n in CORE),'Core model/data/objective differs')
    require(prov['recipe']==read(Path(c['root'])/'recipes'/f'{name}.json'),'Provenance recipe differs')
    require(prov['preparation']==read(Path(c['root'])/'prepared.json'),'Provenance preparation differs')
    j=live_job(PARENTS[name])
    if j:require(j.get('Command')==spec['parent_script'],'Wrong live parent submission script')
    return prov,wandb

def compare(c,approvals):
    parent=Path(c['parent_root']);comparisons={}
    for name in NAMES:
        path=parent/'logs'/f'{name}-train-{PARENTS[name]}.out';rows=[]
        for line in path.read_text(errors='replace').splitlines():
            try:m=json.loads(line)
            except ValueError:continue
            if 'Train/samples_per_second' in m:rows.append(m)
        rows=rows[-20:];require(len(rows)==20,'A100 comparison incomplete')
        median=statistics.median(m['Train/samples_per_second'] for m in rows);a=approvals[name]
        comparisons[name]={'identity':a['identity'],'gpu_profile_sha256':c['runs'][name]['gpu_profile_sha256'],
            'h200_sustained_samples_per_second':a['samples_per_second'],
            'a100_production_median_samples_per_second':median,'ratio_to_a100':a['samples_per_second']/median,
            'a100_observations':rows,'a100_log_modified_at_ns':path.stat().st_mtime_ns}
    return comparisons

def verify_pair_receipts(c):
    root=Path(c['root']);manifest=verify_frozen(c)
    approvals={n:read(root/'profiles'/f'{n}.json') for n in NAMES}
    for n in NAMES:require(approvals[n]['identity']==c['runs'][n]['approved_identity'],'Wrong approved runtime identity')
    comparisons=compare(c,approvals)
    validate_pair(approvals,comparisons,accounting(c['gpu_profile_job_id']),read(root/'profile-complete.json'),manifest['files'])
    for n in NAMES:parent_identity(c,n)
    return approvals,comparisons

def verify_release(c):
    approvals,comparisons=verify_pair_receipts(c)
    import torch
    require(str(torch.__version__)==c['torch_version'],'Runtime Torch version differs')
    releases={}
    for name in NAMES:
        a=approvals[name];release={'approved':True,'paired_migration_authorized':True,
            'gpu_profile_sha256':c['runs'][name]['gpu_profile_sha256'],'identity':a['identity'],
            'throughput_ratio_to_a100':comparisons[name]['ratio_to_a100'],
            'comparison':comparisons[name],'paired_gpu_profile_job_id':c['gpu_profile_job_id'],
            'verified_at':now(),'verification_job_id':os.environ['SLURM_JOB_ID']}
        path=Path(c['root'])/'production-release'/f'{name}.json';atomic(path,release);releases[name]=sha(path)
    atomic(HERE/'verified.json',{'verified':True,'at':now(),'job_id':os.environ['SLURM_JOB_ID'],
        'config_sha256':sha(HERE/'config.json'),'controller_sha256':sha(__file__),
        'gpu_profile_job_id':c['gpu_profile_job_id'],'gpu_profile_accounting':accounting(c['gpu_profile_job_id']),
        'release_sha256':releases,'comparisons':comparisons,'torch_version':str(torch.__version__),
        'cpu_only_revalidation':True,'not_a_new_gpu_test':True})

def inspect_checkpoint(path,metadata,recipe):
    import torch
    require(Path(path).is_file(),'Expected production checkpoint absent')
    checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    validate_checkpoint(checkpoint,metadata,recipe)
    require(checkpoint['model'],'Model weights absent')
    for state in checkpoint['optimizer']['state'].values():
        require(all(k in state for k in ('step','exp_avg','exp_avg_sq')),'Adam moments incomplete')
        require(float(state['step'])==metadata['progress']['updates'],'Adam update counter differs')
        require(state['exp_avg'].dtype==torch.float32 and state['exp_avg_sq'].dtype==torch.float32,'Adam moment precision differs')
    require(len(checkpoint['scheduler']['_last_lr'])==len(checkpoint['optimizer']['param_groups']),'Scheduler groups differ')
    for lr,group in zip(checkpoint['scheduler']['_last_lr'],checkpoint['optimizer']['param_groups']):require(lr==group['lr'],'LR schedule differs')
    del checkpoint;gc.collect()


def stop_parent(c,name,folder):
    """Signal only this exact canonical job; cancel only a verified saved allocation."""
    out=Path(c['parent_root'])/'runs'/name;recipe=read(Path(c['root'])/'recipes'/f'{name}.json');jid=PARENTS[name]
    deadline=time.monotonic()+1200
    for attempt in range(3):
        j=live_job(jid)
        require(j is not None and j['Command']==c['runs'][name]['parent_script'],'Parent live script absent/different; reconcile')
        while j['JobState']=='COMPLETING' and time.monotonic()<deadline:
            time.sleep(2);j=live_job(jid);require(j is not None,'Parent terminal without our stop receipt; reconcile')
        if j['JobState']=='PENDING':
            require((out/'stopped.json').exists(),'Pending parent lacks saved stop receipt')
            latest=read(out/'latest.json');stopped=read(out/'stopped.json')
            require(stopped['progress']==latest['progress'] and stopped['reason']=='allocation_signal','Pending checkpoint/stop mismatch')
            inspect_checkpoint(out/'latest.pt',latest,recipe)
            current=live_job(jid)
            if current and current['JobState']=='RUNNING':continue
            require(current and current['JobState']=='PENDING','Parent pending state changed; reconcile')
            atomic(folder/'parent-saved-stop.json',{'at':now(),'parent_job_id':jid,'latest':latest,'stopped':stopped,'checkpoint_metadata_validated':True,'pending_automatic_requeue':True})
            command(['scancel',jid])
            while time.monotonic()<deadline and visible(jid):time.sleep(2)
            require(not visible(jid),'Pending parent still queued')
            a=terminal_accounting(jid,deadline)
            atomic(folder/'parent-terminal.json',{'at':now(),'parent_job_id':jid,'accounting':a,'latest':latest});return
        require(j['JobState']=='RUNNING','Parent not RUNNING; reconcile before mutation')
        original=allocation_key(j);sent=time.time_ns()
        atomic(folder/f'signal-{attempt}.json',{'at':now(),'signal_after_ns':sent,'parent_job_id':jid,'allocation':original,'latest_before':read(out/'latest.json')})
        command(['scancel','--signal=USR1','--batch',jid])
        retry=False
        while time.monotonic()<deadline:
            time.sleep(1)
            if not (out/'stopped.json').exists():continue
            stopped=read(out/'stopped.json');latest=read(out/'latest.json');lm=(out/'latest.json').stat().st_mtime_ns;sm=(out/'stopped.json').stat().st_mtime_ns
            if sm<=sent or lm<=sent:continue
            current=live_job(jid)
            if current and current['JobState']=='RUNNING' and allocation_key(current)!=original:
                retry=True;break
            # Header/state inspection precedes cancellation; no weights leave Palmetto.
            inspect_checkpoint(out/'latest.pt',latest,recipe)
            if read(out/'latest.json')!=latest:continue
            require(stopped['progress']==latest['progress'] and stopped['reason']=='allocation_signal','Stop/checkpoint mismatch')
            # Validate original allocation even if it is now pending an automatic requeue.
            current=live_job(jid)
            if current and current['JobState']=='RUNNING' and allocation_key(current)!=original:
                retry=True;break
            validate_saved_stop(latest,stopped,{'identity':latest['identity'],'progress':latest['progress']},sent,lm,sm,original,original)
            atomic(folder/'parent-saved-stop.json',{'at':now(),'parent_job_id':jid,'allocation':original,'signal_after_ns':sent,'stopped':stopped,'latest':latest,'checkpoint_metadata_validated':True,'current_scheduler':current})
            if current:command(['scancel',jid])
            while time.monotonic()<deadline and visible(jid):time.sleep(2)
            require(not visible(jid),'Parent remains in queue; refusing concurrent training')
            a=terminal_accounting(jid,deadline)
            atomic(folder/'parent-terminal.json',{'at':now(),'parent_job_id':jid,'accounting':a,'latest':read(out/'latest.json')})
            return
        if not retry:break
    raise RuntimeError('No verified latest saved parent stop; fail closed and reconcile')

def freeze_parent(c,name,folder):
    root=Path(c['root']);parent=Path(c['parent_root']);out=parent/'runs'/name
    require(not visible(PARENTS[name]),'Parent still active; refusing checkpoint freeze')
    prov,wandb=parent_identity(c,name);meta=read(out/'latest.json');recipe=read(root/'recipes'/f'{name}.json')
    require(meta['identity']==prov['identity'],'Parent metadata identity differs');inspect_checkpoint(out/'latest.pt',meta,recipe)
    dest=folder/'immutable-parent';dest.mkdir(exist_ok=True);copy=dest/'latest.pt'
    original_sha=sha(out/'latest.pt')
    if copy.exists():require(sha(copy)==original_sha,'Existing immutable parent differs')
    else:
        tmp=dest/'latest.pt.partial';shutil.copyfile(out/'latest.pt',tmp)
        require(sha(tmp)==original_sha,'Checkpoint copy SHA differs');os.replace(tmp,copy);copy.chmod(0o444)
    require(read(out/'latest.json')==meta,'Parent changed during freeze')
    for n,value in [('latest.json',meta),('provenance.json',prov),('wandb.json',wandb),('prepared.json',read(root/'prepared.json')),('recipe.json',recipe)]:
        path=dest/n
        if path.exists():require(read(path)==value,'Immutable metadata differs')
        else:atomic(path,value);path.chmod(0o444)
    m=migration_manifest(name,str(copy),original_sha,meta,recipe,read(root/'prepared.json'),prov['source'],read(root/'source-manifest.json')['files'],c['torch_version'],'NVIDIA H200',wandb,True)
    atomic(root/'resume'/f'{name}.json',m)
    target=root/'runs'/name;target.mkdir(exist_ok=True)
    if (target/'wandb.json').exists():require(read(target/'wandb.json')['id']==WANDB[name],'Target W&B identity differs')
    else:atomic(target/'wandb.json',wandb)
    atomic(folder/'migration-prepared.json',{'at':now(),'manifest':m,'wandb':wandb,'parent_artifacts_preserved':True})


def allocate(c,name):
    import torch
    root=Path(c['root']);folder=HERE/'handoff'/name;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        approvals,_=verify_pair_receipts(c);verified=read(HERE/'verified.json');submissions=read(HERE/'submissions.json')
        require(verified['verified'] and verified['config_sha256']==sha(HERE/'config.json') and verified['controller_sha256']==sha(__file__),'CPU release verification differs')
        require(accounting(verified['job_id'])=={'state':'COMPLETED','exit_code':'0:0'},'CPU revalidation not successful')
        require(submissions['production'][name]['job_id']==os.environ['SLURM_JOB_ID'],'Unregistered replacement allocation')
        require(str(torch.__version__)==c['torch_version'],'Runtime version differs')
        require(torch.cuda.device_count()==2,'Exactly two visible GPUs required')
        hw=[]
        for i in range(2):
            torch.cuda.set_device(i);h=torch.cuda.get_device_properties(i);a=torch.ones((8,8),device=i);usable=bool(torch.isfinite(a@a).all().item());del a
            hw.append({'name':h.name,'total_memory':h.total_memory,'cuda_usable':usable})
        require(sha(root/'production-release'/f'{name}.json')==verified['release_sha256'][name],'Measured release changed')
        validate_allocation(hw,os.environ['SLURM_JOB_ID'],submissions['production'][name]['job_id'],True,True)
        torch.cuda.empty_cache();atomic(folder/'allocated.json',{'at':now(),'job_id':os.environ['SLURM_JOB_ID'],'hardware':hw,'source_release_verified':True,'before_parent_signal':True})
        prov,wandb=parent_identity(c,name);out=Path(c['parent_root'])/'runs'/name;meta=read(out/'latest.json');completed=read(out/'completed.json') if (out/'completed.json').exists() else None
        # Already-completed A100 budgets release target without any repeat training.
        if completed:
            require(parent_decision(name,'COMPLETED',prov,meta,completed,True)=='release_target_without_training','Invalid completion')
            require((out/'latest.pt').is_file(),'Completed parent checkpoint absent')
            atomic(folder/'already-completed.json',{'at':now(),'parent_job_id':PARENTS[name],'completed':completed,'production_training_skipped':True});return 10
        own=root/'runs'/name/'latest.pt';ownmeta=own.with_suffix('.json')
        if own.exists():
            require(not visible(PARENTS[name]),'Parent active while target checkpoint exists')
            require(read(ownmeta)['identity']==approvals[name]['identity'],'Wrong own target checkpoint')
            inspect_checkpoint(own,read(ownmeta),read(root/'recipes'/f'{name}.json'));require((root/'runs'/name/'wandb.json').exists(),'Own run W&B missing');return 0
        require(not ownmeta.exists(),'Expected own checkpoint missing')
        prepared=folder/'migration-prepared.json'
        if prepared.exists():
            require(not visible(PARENTS[name]),'Parent restarted after migration prepare')
            m=read(prepared)['manifest'];require(read(root/'resume'/f'{name}.json')==m and sha(m['checkpoint'])==m['checkpoint_sha256'],'Immutable migration state differs');return 0
        parent_live=live_job(PARENTS[name]);parent_state=parent_live['JobState'] if parent_live else accounting(PARENTS[name])['state'].split()[0]
        parent_decision(name,parent_state,prov,meta,None,True)
        # A previous invocation may have saved/cancelled successfully before copying.
        if (folder/'parent-terminal.json').exists():require(not visible(PARENTS[name]),'Saved parent became active')
        elif (folder/'parent-saved-stop.json').exists() and not visible(PARENTS[name]):
            saved=read(folder/'parent-saved-stop.json');require(saved['latest']==read(out/'latest.json'),'Parent progressed after saved stop; reconcile')
            inspect_checkpoint(out/'latest.pt',saved['latest'],read(root/'recipes'/f'{name}.json'))
            a=terminal_accounting(PARENTS[name],time.monotonic()+120)
            atomic(folder/'parent-terminal.json',{'at':now(),'parent_job_id':PARENTS[name],'accounting':a,'latest':saved['latest'],'recovered_saved_stop':True})
        else:stop_parent(c,name,folder)
        freeze_parent(c,name,folder);return 0


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify','allocate']);p.add_argument('--recipe',choices=NAMES);p.add_argument('--config-sha',required=True);p.add_argument('--controller-sha',required=True);a=p.parse_args()
    require(sha(HERE/'config.json')==a.config_sha and sha(__file__)==a.controller_sha,'Submitted controller/config bytes changed')
    c=read(HERE/'config.json')
    for n,h in c['guard_files'].items():require(sha(HERE/n)==h,'Guard code changed')
    if a.mode=='verify':verify_release(c);return 0
    require(a.recipe is not None,'Missing exact recipe');return allocate(c,a.recipe)

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        atomic(HERE/'errors'/f"{os.environ.get('SLURM_JOB_ID','unsubmitted')}-{time.time_ns()}.json",{'at':now(),'error_type':type(e).__name__,'message':str(e),'no_fresh_initialization':True})
        raise
