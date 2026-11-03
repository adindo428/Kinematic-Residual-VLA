"""Record independently frozen Delta adapters; retain existing checkpoint identity."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/training-sources/delta-domv-f991015a')
ROOT=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014')
sys.path.insert(0,str(SOURCE))
from utils.dom_experiment import atomic_json,digest,file_digest,source_digest

def main():
    assert not (HERE/'deployment.json').exists(),'Inspect prior deployment instead of overwriting'
    manifest=json.loads((ROOT/'manifest.json').read_text());recipe=json.loads((ROOT/'recipe.json').read_text())
    assert digest({k:v for k,v in manifest.items() if k!='fingerprint'})==manifest['fingerprint']
    assert source_digest(SOURCE)==manifest['source_sha256']=='f991015a229476400e0fc772177af33759d1bc3f23901381e2f83c7faaeab921'
    for name,h in manifest['artifacts'].items():assert file_digest(ROOT/name)==h,name
    latest=json.loads((ROOT/'runs/domv100/latest.json').read_text())
    assert latest['fingerprint']==digest({'manifest':manifest['fingerprint'],'recipe':recipe,'variant':'domv100','world_size':8})
    selected=json.loads((ROOT/'selected-runtime.json').read_text())
    runtime={'manifest':manifest['fingerprint'],'runtime':selected['runtime'],'reason':'Same batch/runtime on eight H200s, requiring actual hardware and saved-checkpoint proof'}
    assert recipe['runtime_fingerprint']==digest(runtime['runtime'])
    runtime['fingerprint']=digest(runtime);atomic_json(HERE/'runtime.json',runtime)
    names=['activate.sh','entry.py','launch_ranks.sh','stop_ranks.py','production.sbatch','prepare.py','check.py','native_benchmark.py','sitecustomize.py','speed_probe.py','profile.py','profile.sbatch']
    deployment={'manifest':manifest['fingerprint'],'recipe':recipe['fingerprint'],'source':str(SOURCE),
        'source_sha256':manifest['source_sha256'],'family':str(ROOT),'variant':'domv100',
        'checkpoint_fingerprint':latest['fingerprint'],'initial_progress':latest['progress'],
        'files':{name:file_digest(HERE/name) for name in names},'runtime':runtime,
        'cluster':'Delta','account':'bivv-delta-gpu','world_size':8,'gpus_per_node':8,'node_count':1,
        'gpu':'H200141GB','allocation_hours':7,'maximum_charge_units':168,'operational_epoch_cap':10,
        'scheduler_horizon_epochs':100,'old_deltaai_job':'3381947',
        'checkpoint_preserved':str(ROOT.parent/'transfer-domv-delta-20261020/checkpoint-transfer.json')}
    deployment['fingerprint']=digest(deployment);atomic_json(HERE/'deployment.json',deployment)
    (HERE/'logs').mkdir(exist_ok=True)
    print(json.dumps(deployment),flush=True)

if __name__=='__main__':main()
