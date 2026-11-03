"""Multi-node entry adapter; reuse frozen model, sampler, training and checkpoint math."""
import argparse,json,logging,os,sys
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
import torch
import torch.distributed as dist
from core.dom_distributed import initialize_control_group,leader_call
from core.dom_distributed_training import distributed_smoke,train_distributed
from core.dom_training import load_family
from utils.dom_experiment import digest,atomic_json,file_digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--family-root',type=Path,required=True)
    p.add_argument('--runtime-file',type=Path,required=True);p.add_argument('--command',choices=['smoke','train'],required=True)
    args=p.parse_args();world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world==16 and int(os.environ['LOCAL_WORLD_SIZE'])==4 and torch.cuda.device_count()==4
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    torch.cuda.set_device(local);torch.set_num_threads(1);device=torch.device('cuda',local)
    props=torch.cuda.get_device_properties(device);assert 'GH200' in props.name
    root=args.family_root;manifest=load_family(root)
    assert manifest['artifacts']['multi_entry.py']==file_digest(Path(__file__))
    selected=json.loads(args.runtime_file.read_text());copy=dict(selected);expected=copy.pop('fingerprint');assert digest(copy)==expected
    runtime=selected['runtime'];assert selected['manifest']==manifest['fingerprint']
    assert 1<=runtime['microbatch']<=352//world and 352%(world*runtime['microbatch'])==0
    recipe=json.loads((root/'recipe.json').read_text());copy=dict(recipe);fp=copy.pop('fingerprint');assert digest(copy)==fp
    assert recipe['manifest']==manifest['fingerprint'] and recipe['effective_batch']==352 and recipe['epochs']==100
    assert runtime['precision']==recipe['precision']=='bf16' and runtime['sampling']==recipe['sampling']=='frames'
    if args.command=='train':assert recipe['runtime_fingerprint']==digest(runtime) and selected['world_size']==world
    hardware={'name':props.name,'total_memory':props.total_memory,'torch':torch.__version__,
              'cuda':torch.version.cuda,'capability':[props.major,props.minor]}
    key=digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':'krvp100'})[:16]
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    try:
        initialize_control_group()
        if args.command=='smoke':
            distributed_smoke(root,'krvp100',runtime,device);return 0
        profile=json.loads((root/'distributed-profiles'/(key+'.json')).read_text());copy=dict(profile);proof=copy.pop('fingerprint')
        assert digest(copy)==proof==selected['distributed_profile_fingerprint']
        assert profile['manifest']==manifest['fingerprint'] and profile['world_size']==world
        assert profile['variant']=='krvp100' and profile['future_inputs_ignored']
        assert profile['model_config']=={'residual':True,'velocity':True,'phase_gate':'causal_rules','formulation':'coordinate'}
        assert all(profile.get(k) for k in ['complete','safe','full_data_benchmark','strict_reload','rank_rng_reload','post_training_validation_passed','sustained_reload_passed'])
        assert profile['benchmark_samples']==70400 and profile['measured_updates']==180
        continuation=root/'runs/krvp100'/('continuation-'+os.environ['SLURM_JOB_ID']+'.json')
        leader_call(lambda:continuation.unlink(missing_ok=True))
        result=train_distributed(root,'krvp100',runtime,recipe,device)
        if result.get('interrupted'):
            leader_call(lambda:atomic_json(continuation,{'manifest':manifest['fingerprint'],'world_size':world,
                'updates':result['updates'],'samples':result['samples'],'checkpoint_saved':True}))
        return 75 if result.get('interrupted') else 0
    finally:dist.destroy_process_group()

if __name__=='__main__':sys.exit(main())
