"""Approve only this allocation's complete sustained multi-node result."""
import json,os,sys
from pathlib import Path
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
from core.dom_training import load_family
from utils.dom_experiment import digest,atomic_json,file_digest
HERE=Path(__file__).resolve().parent
ROOT=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/domv-scale-ddp-20261014')
def verified(path):
    v=json.loads(path.read_text());assert digest({k:x for k,x in v.items() if k!='fingerprint'})==v['fingerprint'];return v
def main():
    job=os.environ['SLURM_JOB_ID'];world=int(os.environ['SLURM_JOB_NUM_NODES'])*4
    receipt=json.loads((HERE/('scale-check-'+job+'-'+str(world)+'.json')).read_text())
    assert receipt['complete'] and receipt['passed'] and not receipt['training_started']
    manifest=load_family(ROOT);profile=receipt['profile']
    assert digest({k:v for k,v in profile.items() if k!='fingerprint'})==profile['fingerprint']
    assert profile['manifest']==manifest['fingerprint'] and profile['world_size']==world and profile['benchmark_job_id']==job
    assert all(profile.get(k) for k in ['complete','safe','full_data_benchmark','strict_reload','rank_rng_reload','post_training_validation_passed','sustained_reload_passed'])
    assert profile['benchmark_samples']==70400 and profile['measured_updates']==180
    assert profile['benchmark_driver_sha256']==manifest['artifacts']['native_benchmark.py']
    assert profile['ram_headroom_basis']=='per-node sum of process peaks'
    assert not (ROOT/'runs/domv100/provenance.json').exists()
    recipe=verified(ROOT/'recipe.json');recipe.pop('fingerprint')
    assert recipe['epochs']==100 and recipe['lr']==3e-5 and recipe['effective_batch']==352 and recipe['precision']=='bf16' and recipe['sampling']=='frames'
    assert recipe['early_stopping']=={'enabled':True,'metric':'h20/action_mse','min_epochs':5,'patience':3,'relative_min_delta':.005}
    runtime=profile['runtime'];recipe['runtime_fingerprint']=digest(runtime)
    if world not in recipe['supported_world_sizes']:recipe['supported_world_sizes'].append(world)
    recipe['fingerprint']=digest(recipe);atomic_json(ROOT/'recipe.json',recipe)
    prior=verified(HERE/'bounded-single-gpu.json')
    selected={'manifest':manifest['fingerprint'],'runtime':runtime,'world_size':world,
        'node_count':world//4,'gpus_per_node':4,'global_effective_batch':352,
        'distributed_profile_fingerprint':profile['fingerprint'],'single_gpu_profile_fingerprint':prior['fingerprint'],
        'samples_per_second':profile['samples_per_second'],
        'training_epoch_hours_excluding_validation_and_io':manifest['train_frames']/profile['samples_per_second']/3600,
        'measurement_note':'200-update check;20warmup/180measured; timing includes checkpoints25..175, excludes startup/final validation/reload'}
    selected['fingerprint']=digest(selected);atomic_json(ROOT/'selected-runtime.json',selected)
    print(json.dumps({'selected_runtime':selected,'recipe_fingerprint':recipe['fingerprint']}),flush=True)
if __name__=='__main__':main()
