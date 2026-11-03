"""Allocated-first exact-run budget extension controller; no duplicate training."""
import argparse,fcntl,gc,hashlib,json,os,shutil,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from budget import require,validate_extension,validate_progress,validate_release,CORE
from common import fingerprint
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


def allocation_key(j):return (j.get('StartTime'),j.get('Restarts'),j.get('BatchHost'))

def validate_saved_stop(latest,stopped,checkpoint,sent,lm,sm,original,current):
    require(lm>sent and sm>sent and original==current,'Stale checkpoint or changed allocation')
    require(stopped['reason']=='allocation_signal' and latest['progress']==stopped['progress']==checkpoint['progress'] and latest['identity']==checkpoint['identity'],'Checkpoint/stop state differs')

def verify_frozen(c,name):
    root=Path(c['root']);manifest=read(root/'source-manifest.json')
    require(sha(root/'source-manifest.json')==c['source_manifest_sha256'],'Source manifest changed')
    require({p.name for p in root.glob('*.py')}==set(manifest['files']),'Source inventory changed')
    for n,h in manifest['files'].items():require(sha(root/n)==h,'Frozen source changed: '+n)
    for n,h in read(root/'preparation-manifest.json')['files'].items():require(sha(root/n)==h,'Preparation changed: '+n)
    require(sha(root/'preparation-manifest.json')==c['preparation_manifest_sha256'],'Preparation inventory changed')
    for n,h in read(root/'implementation-lock.json')['files'].items():require(sha(n)==h,'Native model implementation changed')
    for d in ['recipes','parent-recipes','runtimes']:
        require(sha(root/d/(name+'.json'))==c['runs'][name][d+'_sha256'],'Recipe/runtime bytes changed')
    parent=read(root/'parent-recipes'/(name+'.json'));recipe=read(root/'recipes'/(name+'.json'))
    validate_extension(parent,recipe)
    require((root/'env/bin/python').resolve()==(Path(c['runs'][name]['parent_root'])/'env/bin/python').resolve(),'Parent/target environments differ')
    return manifest,recipe

def parent_identity(c,name):
    spec=c['runs'][name];p=Path(spec['parent_root']);out=p/'runs'/name
    prov=read(out/'provenance.json');wandb=read(out/'wandb.json')
    require(str(prov['job_id'])==spec['parent_job_id'] and prov['identity']==spec['parent_identity'],'Wrong canonical parent identity')
    require(prov['hardware']=='NVIDIA A100 80GB PCIe' and prov['world_size']==2,'Wrong canonical parent hardware/ranks')
    require(fingerprint(prov['source'])==spec['parent_source_sha256'],'Parent source fingerprint differs')
    for n,h in prov['source'].items():require(sha(p/n)==h,'Parent source changed: '+n)
    require(wandb['id']==prov['wandb_id']==spec['parent_wandb_id'],'Wrong parent W&B identity')
    target=read(Path(c['root'])/'source-manifest.json')['files']
    require(all(prov['source'][n]==target[n] for n in CORE),'Core model/data/objective changed')
    if 'instruction_generator.py' in prov['source']:require(prov['source']['instruction_generator.py']==target['instruction_generator.py'],'Instructions changed')
    require(prov['recipe']==read(Path(c['root'])/'parent-recipes'/(name+'.json')),'Original recipe differs')
    require(prov['preparation']==read(Path(c['root'])/'prepared.json'),'Audited preparation differs')
    require(sha(spec['parent_script'])==spec['parent_script_sha256'],'Parent launch script changed')
    j=live_job(spec['parent_job_id'])
    if j:require(j.get('Command')==spec['parent_script'],'Wrong live parent script')
    return prov,wandb

def inspect_checkpoint(path,metadata,recipe):
    import torch
    require(Path(path).is_file(),'Expected production checkpoint absent')
    c=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    require(set(('model','optimizer','scheduler','rng','progress','identity')).issubset(c),'Incomplete production checkpoint')
    require(c['model'] and c['identity']==metadata['identity'] and c['progress']==metadata['progress'],'Checkpoint metadata differs')
    validate_progress(c['progress'],recipe)
    require(c['scheduler']['last_epoch']==c['progress']['updates'],'Scheduler counter differs')
    require(len(c['rng'])==2 and all(set(('python','numpy','torch','cuda')).issubset(r) for r in c['rng']),'Per-rank RNG missing')
    require(c['optimizer']['state'] and c['optimizer']['param_groups'],'Optimizer state absent')
    for g in c['optimizer']['param_groups']:
        require(tuple(g['betas'])==(.9,.95) and g['eps']==1e-8 and g['weight_decay']==recipe['weight_decay'] and g['initial_lr']==recipe['lr'],'Optimizer recipe differs')
    for state in c['optimizer']['state'].values():
        require(all(k in state for k in ('step','exp_avg','exp_avg_sq')),'Adam moments missing')
        require(float(state['step'])==c['progress']['updates'],'Adam update counter differs')
        require(state['exp_avg'].dtype==torch.float32 and state['exp_avg_sq'].dtype==torch.float32,'Moment precision differs')
    require(len(c['scheduler']['_last_lr'])==len(c['optimizer']['param_groups']),'Scheduler groups differ')
    for lr,g in zip(c['scheduler']['_last_lr'],c['optimizer']['param_groups']):require(lr==g['lr'],'LR/scheduler state differs')
    del c;gc.collect()

def stop_parent(c,name,folder):
    """Signal only this exact canonical job; cancel only a verified saved allocation."""
    out=Path(c['runs'][name]['parent_root'])/'runs'/name;recipe=read(Path(c['root'])/'recipes'/f'{name}.json');jid=c['runs'][name]['parent_job_id']
    deadline=time.monotonic()+1200
    for attempt in range(3):
        completed=read(out/'completed.json') if (out/'completed.json').exists() else None
        if completed:
            latest=read(out/'latest.json')
            require(completed['progress']==latest['progress'] and completed['stop_reason']=='official_step_budget','Unexpected parent completion')
            require(not visible(jid) and accounting(jid)=={'state':'COMPLETED','exit_code':'0:0'},'Parent cleanup/accounting still active; reconcile')
            inspect_checkpoint(out/'latest.pt',latest,recipe)
            atomic(folder/'parent-terminal.json',{'at':now(),'parent_job_id':jid,'completed_prior_step_budget':completed,'latest':latest});return
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


def freeze_parent(c,name,folder,recipe):
    spec=c['runs'][name];root=Path(c['root']);parent=Path(spec['parent_root']);out=parent/'runs'/name
    require(not visible(spec['parent_job_id']),'Parent still active')
    prov,wandb=parent_identity(c,name);meta=read(out/'latest.json')
    require(meta['identity']==prov['identity'],'Parent metadata identity differs')
    inspect_checkpoint(out/'latest.pt',meta,recipe)
    dest=folder/'immutable-parent';dest.mkdir(exist_ok=True);copy=dest/'latest.pt';original_sha=sha(out/'latest.pt')
    if copy.exists():require(sha(copy)==original_sha,'Immutable parent differs')
    else:
        tmp=dest/'latest.pt.partial';shutil.copyfile(out/'latest.pt',tmp)
        require(sha(tmp)==original_sha,'Copy SHA differs');os.replace(tmp,copy);copy.chmod(0o444)
    require(read(out/'latest.json')==meta,'Parent changed during freeze')
    for n,value in [('latest.json',meta),('provenance.json',prov),('wandb.json',wandb),('parent-recipe.json',prov['recipe'])]:
        path=dest/n
        if path.exists():require(read(path)==value,'Immutable metadata differs')
        else:atomic(path,value);path.chmod(0o444)
    m={'latest_complete_production_checkpoint':True,'user_authorized_budget_extension':True,
       'user_authorized_cross_hardware':spec['hardware']!=prov['hardware'],
       'recipe_fingerprint':fingerprint(recipe),'parent_recipe':prov['recipe'],
       'preparation':prov['preparation'],'parent_source_sha256':spec['parent_source_sha256'],
       'parent_hardware':prov['hardware'],'target_hardware':spec['hardware'],'world_size':2,
       'torch_version':c['torch_version'],'parent_source':prov['source'],'progress':meta['progress'],
       'checkpoint':str(copy),'checkpoint_sha256':original_sha,'identity':meta['identity'],
       'parent_job_id':spec['parent_job_id'],'original_wandb_id':wandb['id']}
    atomic(root/'resume'/(name+'.json'),m)
    target=root/'runs'/name;target.mkdir(exist_ok=True)
    if (target/'wandb.json').exists():require(read(target/'wandb.json')['id']==spec['parent_wandb_id'],'Wrong target W&B')
    else:atomic(target/'wandb.json',wandb)
    atomic(folder/'migration-prepared.json',{'at':now(),'manifest':m,'parent_artifacts_preserved':True})

def allocate(c,name):
    import torch
    root=Path(c['root']);spec=c['runs'][name];folder=HERE/'handoff'/name;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,recipe=verify_frozen(c,name);state=read(HERE/'submissions.json')
        row=state['production'][name];require(row['job_id']==os.environ['SLURM_JOB_ID'],'Unregistered production allocation')
        require(sha(row['script'])==row['script_sha256'],'Submitted script bytes changed')
        profile=read(root/'profiles'/(name+'.json'));approval_sha=sha(root/'profiles'/(name+'.json'))
        require(accounting(state['profiles'][name]['job_id'])=={'state':'COMPLETED','exit_code':'0:0'},'Own GPU profile not successful')
        require(profile['source']==manifest['files'] and profile['job_id']==state['profiles'][name]['job_id'],'Wrong source/GPU receipt')
        require(profile['hardware']==spec['hardware'] and profile['world_size']==2,'Wrong actual profiled GPU')
        release={'approved':True,'five_epoch_extension_authorized':True,'gpu_profile_sha256':approval_sha,'identity':profile['identity'],'profile_job_id':state['profiles'][name]['job_id']}
        validate_release(release,profile,approval_sha)
        if recipe['method']=='pi05':require(profile['samples_per_second']>=1.25*9.7,'Sustained H200 benefit insufficient')
        require(str(torch.__version__)==c['torch_version'] and torch.cuda.device_count()==2,'Wrong runtime/allocation')
        hw=[]
        for i in range(2):
            torch.cuda.set_device(i);h=torch.cuda.get_device_properties(i);a=torch.ones((8,8),device=i)
            require(h.name==spec['hardware'] and bool(torch.isfinite(a@a).all()),'Allocated hardware/CUDA differs');del a
            hw.append({'name':h.name,'memory':h.total_memory})
        torch.cuda.empty_cache()
        atomic(root/'production-release'/(name+'.json'),release)
        atomic(folder/'allocated.json',{'at':now(),'job_id':os.environ['SLURM_JOB_ID'],'hardware':hw,'source_and_readiness_verified_before_parent_action':True})
        out=Path(spec['parent_root'])/'runs'/name
        own=root/'runs'/name/'latest.pt';ownmeta=own.with_suffix('.json')
        if own.exists():
            require(not visible(spec['parent_job_id']),'Parent active while own checkpoint exists')
            require(read(ownmeta)['identity']==profile['identity'],'Wrong own checkpoint identity')
            inspect_checkpoint(own,read(ownmeta),recipe);return 0
        require(not ownmeta.exists(),'Expected own checkpoint missing')
        if (folder/'migration-prepared.json').exists():
            require(not visible(spec['parent_job_id']),'Parent restarted after freeze')
            m=read(folder/'migration-prepared.json')['manifest']
            require(read(root/'resume'/(name+'.json'))==m and sha(m['checkpoint'])==m['checkpoint_sha256'],'Prepared migration changed');return 0
        prov,wandb=parent_identity(c,name);meta=read(out/'latest.json')
        require(meta['identity']==prov['identity'],'Wrong parent checkpoint identity')
        completed=read(out/'completed.json') if (out/'completed.json').exists() else None
        if completed:
            require(completed['progress']==meta['progress'] and completed['stop_reason']=='official_step_budget','Wrong prior completion')
            require(meta['progress']['updates']==prov['recipe']['steps'],'Prior completion counter differs')
            require(not visible(spec['parent_job_id']) and accounting(spec['parent_job_id'])=={'state':'COMPLETED','exit_code':'0:0'},'Completed parent still active/accounting incomplete')
            inspect_checkpoint(out/'latest.pt',meta,recipe)
            atomic(folder/'parent-terminal.json',{'at':now(),'parent_job_id':spec['parent_job_id'],'completed_prior_step_budget':completed,'latest':meta})
        elif (folder/'parent-terminal.json').exists():require(not visible(spec['parent_job_id']),'Stopped parent active')
        elif (folder/'parent-saved-stop.json').exists() and not visible(spec['parent_job_id']):
            saved=read(folder/'parent-saved-stop.json');require(saved['latest']==meta,'Parent changed after stop')
            inspect_checkpoint(out/'latest.pt',meta,recipe)
            a=terminal_accounting(spec['parent_job_id'],time.monotonic()+120)
            atomic(folder/'parent-terminal.json',{'at':now(),'parent_job_id':spec['parent_job_id'],'accounting':a,'latest':meta})
        else:stop_parent(c,name,folder)
        freeze_parent(c,name,folder,recipe);return 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',required=True);p.add_argument('--manifest-sha',required=True);a=p.parse_args()
    require(sha(HERE/'control-manifest.json')==a.manifest_sha,'Control inventory changed')
    for n,h in read(HERE/'control-manifest.json')['files'].items():require(sha(HERE/n)==h,'Control file changed: '+n)
    c=read(HERE/'config.json');require(a.recipe in c['runs'],'Unknown run');return allocate(c,a.recipe)
if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        atomic(HERE/'errors'/(str(os.environ.get('SLURM_JOB_ID','unsubmitted'))+'-'+str(time.time_ns())+'.json'),{'at':now(),'type':type(e).__name__,'error':str(e),'no_fresh_start':True});raise
