"""Install a scoped operational stopping control for two already-queued jobs."""
import hashlib,json,os,shutil,subprocess,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ACTIVATE=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-20261008/activate-training.sh')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def atomic(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(str(temp),str(path))

def main():
    control=json.loads((HERE/'control.json').read_text())
    assert digest({k:v for k,v in control.items() if k!='fingerprint'})==control['fingerprint']
    assert sha(HERE/'sitecustomize.py')==control['bootstrap_sha256']
    assert sha(Path(control['trainer_path']))==control['trainer_sha256']
    result={}
    for job,item in control['jobs'].items():
        raw=subprocess.check_output(['scontrol','show','job',job,'-o'],universal_newlines=True)
        fields=dict(x.split('=',1) for x in raw.split() if '=' in x)
        assert fields['JobState']=='PENDING',job
        root=Path(item['family']);manifest=json.loads((root/'manifest.json').read_text())
        assert manifest['fingerprint']==item['manifest']
        assert manifest['source_sha256']==control['source_sha256']
        for name,h in manifest['artifacts'].items():assert sha(root/name)==h,name
        run=root/'runs'/item['variant'];latest=run/'latest.json'
        result[job]={'variant':item['variant'],'state':fields['JobState'],
            'latest':json.loads(latest.read_text()) if latest.exists() else None,
            'time_limit':fields['TimeLimit'],'job_id_preserved':True,'epochs':10,
            'scheduler_horizon_epochs':100,'frozen_recipe_unchanged':True}
    # The existing batch scripts source this activation file at every allocation.
    # A job-scoped PYTHONPATH hook is inherited by torchrun on each allocated node.
    # This preserves queued IDs/age, source hashes and exact checkpoint identities.
    marker='# BEGIN authorized epoch10 control20261019'
    old=ACTIVATE.read_text()
    if marker not in old:
        backup=HERE/'activate-training.before.sh'
        assert not backup.exists(),'Inspect existing activation backup'
        shutil.copy2(str(ACTIVATE),str(backup))
        suffix='\n'+marker+'\ncase "${SLURM_JOB_ID:-}" in\n  3381947|3382533)\n'
        suffix+='    export DVLA_EPOCH_CONTROL_FILE='+str(HERE/'control.json')+'\n'
        suffix+='    export PYTHONPATH='+str(HERE)+':"${PYTHONPATH:-}"\n'
        suffix+='    ;;\nesac\n# END authorized epoch10 control20261019\n'
        temp=ACTIVATE.with_name(ACTIVATE.name+'.epoch10.tmp');temp.write_text(old+suffix)
        shutil.copymode(str(ACTIVATE),str(temp));os.replace(str(temp),str(ACTIVATE))
    else:
        assert 'export DVLA_EPOCH_CONTROL_FILE='+str(HERE/'control.json') in old
    for job,item in control['jobs'].items():
        root=Path(item['family']);setup=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')/('setup-scale-domv-20261014' if item['variant']=='domv100' else 'setup-krvp-4node-20261014')
        status_path=setup/'status.json';status=json.loads(status_path.read_text())
        status.update(operational_epoch_cap=10,scheduler_horizon_epochs=100,
            operational_epoch_control_file=str(HERE/'control.json'),
            operational_epoch_control_fingerprint=control['fingerprint'])
        atomic(status_path,status)
        atomic(root/'provenance/epoch-stop-control-20261019.json',dict(control,job=job))
    receipt={'control_fingerprint':control['fingerprint'],'bootstrap_sha256':control['bootstrap_sha256'],
        'activation_sha256':sha(ACTIVATE),'activation':str(ACTIVATE),'jobs':result,
        'installed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scheduler_requests_changed':False,'source_manifest_recipe_checkpoint_bytes_unchanged':True}
    atomic(HERE/'deployment.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
