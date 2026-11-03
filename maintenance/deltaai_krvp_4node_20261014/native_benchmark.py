"""Fresh-process full-data check; model, optimizer and sampler math are unchanged."""
import argparse,faulthandler,gc,json,logging,os,sys,threading,time,math,socket
from itertools import islice
from datetime import timedelta
from pathlib import Path

SOURCE=Path(os.environ['DVLA_PROJECT_DIR'])
sys.path.insert(0,str(SOURCE))
import torch
import torch.distributed as dist
from core.dom_distributed import initialize_control_group,leader_call,control_group,synchronize_checkpoint_buffers,save_distributed,rank_check
from core.dom_distributed_training import distributed_loader,wrap_policy,updates,restore_distributed
from core.dom_checkpoint import load_generation
from core.dom_training import build_dataset,build_policy,load_family,make_optimizer,make_scheduler,make_loader,seed_all,evaluate,to_device,fixed_noise,amp_context
from utils.dom_experiment import digest,atomic_json,file_digest


class HostMemory:
    """Sample host RAM without repeatedly launching GPU driver query processes."""
    def __init__(self):
        self.stop=threading.Event();self.peak=0
        self.thread=threading.Thread(target=self.collect,daemon=True)
    def collect(self):
        import psutil
        proc=psutil.Process()
        while not self.stop.is_set():
            try:self.peak=max(self.peak,proc.memory_info().rss+sum(p.memory_info().rss for p in proc.children(recursive=True)))
            except psutil.Error:pass
            self.stop.wait(.5)
    def __enter__(self):self.thread.start();return self
    def __exit__(self,*_):self.stop.set();self.thread.join(timeout=2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--family-root',type=Path,required=True);p.add_argument('--runtime-file',type=Path,required=True)
    args=p.parse_args();root=args.family_root
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==16 and int(os.environ['LOCAL_WORLD_SIZE'])==4
    fault=(root/'logs'/('native-fault-'+os.environ['SLURM_JOB_ID']+'-'+args.runtime_file.stem+'-rank'+str(rank)+'.log')).open('a',buffering=1)
    faulthandler.enable(file=fault,all_threads=True)
    # Periodic asynchronous stack dumping is omitted; retain on-crash traces.
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    torch.cuda.set_device(local);torch.set_num_threads(1);device=torch.device('cuda',local)
    props=torch.cuda.get_device_properties(device);assert 'GH200' in props.name
    dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    try:
        initialize_control_group();manifest=load_family(root);seed_all()
        assert manifest['artifacts']['native_benchmark.py']==file_digest(Path(__file__))
        selected=json.loads(args.runtime_file.read_text());expected=selected.pop('fingerprint');assert digest(selected)==expected
        runtime=selected['runtime'];assert selected['manifest']==manifest['fingerprint']
        assert runtime['sampling']=='frames' and runtime['precision']=='bf16'
        hardware={'name':props.name,'total_memory':props.total_memory,'torch':torch.__version__,
            'cuda':torch.version.cuda,'capability':[props.major,props.minor]}
        key=digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':'krvp100'})[:16]
        path=root/'distributed-profiles'/(key+'.json');approved=json.loads(path.read_text())
        proof=approved.pop('fingerprint');assert digest(approved)==proof
        assert approved['manifest']==manifest['fingerprint'] and approved['safe'] and approved['strict_reload'] and approved['rank_rng_reload']
        # Admission uses the completed byte audit for this exact read-only cache.
        reuse=json.loads((root/'provenance/verification-reuse.json').read_text())
        audit=reuse['original_report']
        assert audit['state']=='complete' and not audit['errors'] and audit['completed_files']==804900
        allowed=set(sum(json.loads((root/'split.json').read_text())['splits'].values(),[]))
        import utils.datasets as datasets
        def admit(repo_id,data_root,episodes,required_features,revision):
            assert repo_id=='hzxie/DOM' and Path(data_root).resolve()==Path(manifest['dataset_root']).resolve()
            assert revision==manifest['dataset_revision'] and episodes is not None
            assert set(episodes).issubset(allowed)
            assert set(required_features).issubset({'observation.images.opst_cam','observation.images.wrist_cam','observation.state','observation.environment_state','action'})
            logging.info('rank=%d reused verified cache admission for%d episodes',rank,len(episodes))
        datasets._ensure_lerobot_subset_cached=admit
        torch.cuda.reset_peak_memory_stats(device)
        with HostMemory() as memory:
            policy=build_policy(root,'krvp100',device)
            assert policy.config.enable_residual_flow and policy.config.condition_on_object_velocity
            assert policy.config.residual_phase_gate_mode=='causal_rules' and policy.config.residual_flow_formulation=='coordinate'
            model=wrap_policy(policy,device)
            logging.info('rank=%d model/DDP ready before validation prefetch',rank)
            split=json.loads((root/'split.json').read_text())['splits']
            logging.info('rank=%d fresh full-data check loading',rank)
            data=build_dataset(root,split['train'],training=True);assert len(data)==manifest['train_frames']
            val_loader=None
            if rank==0:
                val=build_dataset(root,split['validation'])
                val_loader=make_loader(val,runtime,indices=json.loads((root/'validation_anchors.json').read_text()))
                next(iter(val_loader))
            dist.barrier(group=control_group())
            torch.manual_seed(42+rank);torch.cuda.manual_seed(42+rank)
            optimizer=make_optimizer(policy,3e-5,runtime['fused']);scheduler=make_scheduler(optimizer,10000,warmup=0)
            progress={'epoch':0,'cursor':0,'updates':0,'samples':0}
            checkpoint=root/'sustained-checkpoints'/key
            checkpoint_fingerprint=digest({'manifest':manifest['fingerprint'],'runtime':runtime,'world':world,'confirmation_updates':200})
            loader=distributed_loader(data,runtime,sampling=runtime['sampling'])
            started=None;elapsed=None
            iterator=updates(model,optimizer,scheduler,loader,progress,runtime['precision'],device)
            for record in iterator:
                logging.info('rank=%d full-data update=%d loss=%g',rank,progress['updates'],record['loss'])
                if progress['updates']==20:torch.cuda.synchronize(device);started=time.perf_counter()
                if progress['updates'] in (25,50,75,100,125,150,175):
                    save_distributed(checkpoint,policy,optimizer,scheduler,progress,checkpoint_fingerprint,None,device=device)
                    logging.info('rank=%d sustained checkpoint saved update=%d',rank,progress['updates'])
                if progress['updates']==200:
                    torch.cuda.synchronize(device);elapsed=time.perf_counter()-started;break
            assert elapsed and progress['samples']==70400
            iterator.close();del iterator
            synchronize_checkpoint_buffers(policy)
            def validate():
                metrics=evaluate(policy,islice(val_loader,2),runtime['precision'],device)
                assert 'h20/action_mse' in metrics and all(math.isfinite(v) for v in metrics.values())
                logging.info('post-training validation passed: %s',metrics)
                return metrics
            metrics=leader_call(validate)
            probe=next(iter(distributed_loader(data,{**runtime,'microbatch':2,'workers':0},lengths=[world*2])))
            probe=to_device(probe,device);noise=fixed_noise(probe,policy,device)
            policy.eval()
            with torch.no_grad(),amp_context(runtime['precision'],device):
                policy.reset();prediction=policy.predict_action_chunk(probe,noise=noise)
            save_distributed(checkpoint,policy,optimizer,scheduler,progress,checkpoint_fingerprint,None,device=device)
            expected_random=torch.rand(4,device=device)
            with torch.no_grad():policy.model.state_proj.weight.add_(1.)
            def reload():
                saved=load_generation(checkpoint,policy,optimizer,scheduler,checkpoint_fingerprint,restore_random=False)
                assert saved['progress']==progress
                restore_distributed(saved)
                torch.testing.assert_close(expected_random,torch.rand(4,device=device),rtol=0,atol=0)
                with torch.no_grad(),amp_context(runtime['precision'],device):
                    policy.reset();torch.testing.assert_close(prediction,policy.predict_action_chunk(probe,noise=noise))
                logging.info('rank=%d sustained checkpoint/reload/RNG passed at200 updates',rank)
                return saved
            saved=rank_check(reload)
            torch.cuda.synchronize(device)
            peak=torch.cuda.max_memory_reserved(device);ram=memory.peak
            assert ram>0,'Host memory measurement unavailable'
            measured=torch.tensor([elapsed,peak,ram],dtype=torch.float64)
            dist.all_reduce(measured[:2],op=dist.ReduceOp.MAX,group=control_group())
            dist.all_reduce(measured[2:],op=dist.ReduceOp.SUM,group=control_group())
            limit=int(os.environ['SLURM_MEM_PER_NODE'])*1024**2
            ranks=[None]*world
            dist.all_gather_object(ranks,{'node':socket.gethostname(),'peak':ram},group=control_group())
            node_peaks={}
            for entry in ranks:node_peaks[entry['node']]=node_peaks.get(entry['node'],0)+entry['peak']
            assert len(node_peaks)==world//4
            logging.info('Memory headroom: GPU reserved=%d/%d; per-node summed process peaks=%s; node limit=%d',int(measured[1]),props.total_memory,node_peaks,limit)
            assert float(measured[1])<=.85*props.total_memory and max(node_peaks.values())<=.85*limit,'Insufficient memory headroom' 
            logging.info('rank=%d native full-data check complete',rank)
            approved.update(model_config={'residual':True,'velocity':True,'phase_gate':'causal_rules','formulation':'coordinate'},full_data_benchmark=True,measured_updates=180,warmup_updates=20,
                samples_per_second=63360/float(measured[0]),max_gpu_reserved_bytes=float(measured[1]),
                sum_peak_ram_bytes=float(measured[2]),node_peak_ram_bytes=node_peaks,
                max_node_peak_ram_bytes=max(node_peaks.values()),ram_headroom_basis="per-node sum of process peaks",benchmark_samples=progress['samples'],safe=True,
                benchmark_driver_sha256=file_digest(Path(__file__)),isolated_process=True,
                memory_monitor='host RAM only; CUDA peak from torch.cuda',benchmark_job_id=os.environ['SLURM_JOB_ID'],
                sustained_checkpoint_updates=[25,50,75,100,125,150,175,200],sustained_reload_passed=True,
                diagnostic_stack_dump_mode='on-crash only; no periodic watchdog',
                post_training_validation_passed=True,validation_batches=2,validation_metrics=metrics,
                policy_before_validation_prefetch=True,cache_admission='Reused exact completed804900-file audit')
            approved['fingerprint']=digest(approved)
            leader_call(lambda:atomic_json(path,approved))
        del loader,val_loader,model,policy,optimizer,scheduler,data
        gc.collect();torch.cuda.empty_cache()
        dist.barrier(group=control_group())
    finally:
        dist.destroy_process_group()
        faulthandler.disable();fault.close()


if __name__=='__main__':main()
