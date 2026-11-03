"""Eight-H200 continuation of the existing eight-rank DOM experiment."""
import argparse,json,logging,math,os,sys
from pathlib import Path
from datetime import timedelta
sys.path.insert(0,os.environ['DVLA_PROJECT_DIR'])
import torch
import torch.distributed as dist
from core.dom_distributed import initialize_control_group,leader_call,rank_check
from core.dom_distributed_training import distributed_smoke,train_distributed,restore_distributed,distributed_loader
from core.dom_checkpoint import load_generation
from core.dom_training import load_family,build_policy,build_dataset,make_optimizer,make_scheduler,to_device,fixed_noise,amp_context
from utils.dom_experiment import atomic_json,digest,file_digest
HERE=Path(__file__).resolve().parent
ROOT=Path(os.environ['DOM100_ROOT'])

def real_resume_check(runtime,device,manifest):
    policy=build_policy(ROOT,'domv100',device)
    assert policy.config.condition_on_object_velocity and not policy.config.enable_residual_flow
    optimizer=make_optimizer(policy,3e-5,runtime['fused'])
    scheduler=make_scheduler(optimizer,100*math.ceil(manifest['train_frames']/352))
    run=ROOT/'runs/domv100';pointer=json.loads((run/'latest.json').read_text())
    recipe=json.loads((ROOT/'recipe.json').read_text())
    fingerprint=digest({'manifest':manifest['fingerprint'],'recipe':recipe,'variant':'domv100','world_size':8})
    assert pointer['fingerprint']==fingerprint
    state=load_generation(run,policy,optimizer,scheduler,fingerprint,restore_random=False)
    assert state['progress']==pointer['progress'] and state['distributed']['world_size']==8
    restore_distributed(state);expected=torch.rand(4,device=device)
    data=build_dataset(ROOT,json.loads((ROOT/'pilot_episodes.json').read_text())[:8])
    batch=next(iter(distributed_loader(data,{**runtime,'microbatch':2,'workers':0},lengths=[16])))
    batch=to_device(batch,device);noise=fixed_noise(batch,policy,device);policy.eval()
    with torch.no_grad(),amp_context(runtime['precision'],device):
        policy.reset();prediction=policy.predict_action_chunk(batch,noise=noise)
        assert torch.isfinite(prediction).all()
    def restore_check():
        restore_distributed(state)
        torch.testing.assert_close(expected,torch.rand(4,device=device),rtol=0,atol=0)
        with torch.no_grad(),amp_context(runtime['precision'],device):
            policy.reset();torch.testing.assert_close(prediction,policy.predict_action_chunk(batch,noise=noise))
        return True
    rank_check(restore_check)
    leader_call(lambda:atomic_json(HERE/('resume-check-'+os.environ['SLURM_JOB_ID']+'.json'),
        {'passed':True,'world_size':8,'variant':'domv100','checkpoint_fingerprint':fingerprint,
         'generation':pointer['generation'],'progress':state['progress'],'strict_weights':True,
         'optimizer_scheduler_loaded':True,'rank_rng_restored':True,'finite_prediction':True,
         'torch':torch.__version__,'source_sha256':manifest['source_sha256']}))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--command',choices=['smoke','train'],required=True)
    parser.add_argument('--diagnostic-world',type=int,default=8);parser.add_argument('--runtime-file',type=Path,default=HERE/'runtime.json')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    assert world==args.diagnostic_world and int(os.environ['LOCAL_WORLD_SIZE'])==world and torch.cuda.device_count()>=world
    assert world in (4,8) and (args.command=='smoke' or world==8)
    torch.cuda.set_device(local);torch.set_num_threads(1);device=torch.device('cuda',local)
    props=torch.cuda.get_device_properties(device);assert 'H200' in props.name and 'GH200' not in props.name
    deployment=json.loads((HERE/'deployment.json').read_text());assert deployment['files']['entry.py']==file_digest(Path(__file__))
    manifest=load_family(ROOT);assert manifest['fingerprint']==deployment['manifest']
    selected=json.loads(args.runtime_file.read_text());runtime=selected['runtime']
    assert digest({k:v for k,v in selected.items() if k!='fingerprint'})==selected['fingerprint']
    recipe=json.loads((ROOT/'recipe.json').read_text())
    assert recipe['epochs']==100
    if args.command=='train':assert recipe['runtime_fingerprint']==digest(runtime)
    dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    try:
        initialize_control_group()
        if args.command=='smoke':
            distributed_smoke(ROOT,'domv100',runtime,device)
            if world==8 and runtime['workers']==4:real_resume_check(runtime,device,manifest)
            return 0
        import sitecustomize
        assert sitecustomize.TRAINING_CONTROL_INSTALLED and train_distributed.__operational_epoch_cap__==10
        hardware={'name':props.name,'total_memory':props.total_memory,'torch':torch.__version__,
            'cuda':torch.version.cuda,'capability':[props.major,props.minor]}
        key=digest({'hardware':hardware,'world_size':8,'runtime':runtime,'variant':'domv100'})[:16]
        profile=json.loads((ROOT/'distributed-profiles'/(key+'.json')).read_text())
        approval=json.loads((HERE/'approval.json').read_text())
        assert digest({k:v for k,v in profile.items() if k!='fingerprint'})==profile['fingerprint']==approval['profile_fingerprint']
        assert profile['benchmark_driver_sha256']==deployment['files']['native_benchmark.py']
        assert all(profile.get(k) for k in ['complete','safe','strict_reload','rank_rng_reload','full_data_benchmark','sustained_reload_passed','post_training_validation_passed'])
        assert profile['world_size']==8 and profile['benchmark_samples']==70400 and profile['measured_updates']==180
        continuation=HERE/('continuation-'+os.environ['SLURM_JOB_ID']+'.json')
        leader_call(lambda:continuation.unlink(missing_ok=True))
        result=train_distributed(ROOT,'domv100',runtime,recipe,device)
        if result.get('interrupted'):
            leader_call(lambda:atomic_json(continuation,{'checkpoint_saved':True,'progress':result,'variant':'domv100','world_size':8}))
        return 75 if result.get('interrupted') else 0
    finally:dist.destroy_process_group()

if __name__=='__main__':sys.exit(main())
