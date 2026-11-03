"""Fresh-process full-data check; model, optimizer and sampler math are unchanged."""
import argparse,faulthandler,gc,json,logging,os,sys,threading,time
from datetime import timedelta
from pathlib import Path

SOURCE=Path(os.environ['DVLA_PROJECT_DIR'])
sys.path.insert(0,str(SOURCE))
import torch
import torch.distributed as dist
from core.dom_distributed import initialize_control_group,leader_call,control_group
from core.dom_distributed_training import distributed_loader,wrap_policy,updates
from core.dom_training import build_dataset,build_policy,load_family,make_optimizer,make_scheduler,make_loader,seed_all
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
    assert world==4
    fault=(root/'logs'/('native-fault-'+os.environ['SLURM_JOB_ID']+'-rank'+str(rank)+'.log')).open('a',buffering=1)
    faulthandler.enable(file=fault,all_threads=True);faulthandler.dump_traceback_later(120,repeat=True,file=fault)
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
        key=digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':'domv100'})[:16]
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
        with HostMemory() as memory:
            policy=build_policy(root,'domv100',device);model=wrap_policy(policy,device)
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
            loader=distributed_loader(data,runtime,sampling=runtime['sampling'])
            torch.cuda.reset_peak_memory_stats(device);started=None;elapsed=None
            iterator=updates(model,optimizer,scheduler,loader,progress,runtime['precision'],device)
            for record in iterator:
                logging.info('rank=%d full-data update=%d loss=%g',rank,progress['updates'],record['loss'])
                if progress['updates']==2:torch.cuda.synchronize(device);started=time.perf_counter()
                if progress['updates']==18:
                    torch.cuda.synchronize(device);elapsed=time.perf_counter()-started;break
            assert elapsed and progress['samples']==6336
            iterator.close();del iterator
            peak=torch.cuda.max_memory_reserved(device);ram=memory.peak
            assert ram>0,'Host memory measurement unavailable'
            measured=torch.tensor([elapsed,peak,ram],dtype=torch.float64)
            dist.all_reduce(measured[:2],op=dist.ReduceOp.MAX,group=control_group())
            dist.all_reduce(measured[2:],op=dist.ReduceOp.SUM,group=control_group())
            limit=int(os.environ['SLURM_MEM_PER_NODE'])*1024**2
            assert float(measured[1])<=.85*props.total_memory and float(measured[2])<=.85*limit,'Insufficient memory headroom'
            logging.info('rank=%d native full-data check complete',rank)
            approved.update(full_data_benchmark=True,measured_updates=16,warmup_updates=2,
                samples_per_second=5632/float(measured[0]),max_gpu_reserved_bytes=float(measured[1]),
                sum_peak_ram_bytes=float(measured[2]),benchmark_samples=progress['samples'],safe=True,
                benchmark_driver_sha256=file_digest(Path(__file__)),isolated_process=True,
                memory_monitor='host RAM only; CUDA peak from torch.cuda',benchmark_job_id=os.environ['SLURM_JOB_ID'],
                policy_before_validation_prefetch=True,cache_admission='Reused exact completed804900-file audit')
            approved['fingerprint']=digest(approved)
            leader_call(lambda:atomic_json(path,approved))
        del loader,val_loader,model,policy,optimizer,scheduler,data
        gc.collect();torch.cuda.empty_cache()
        dist.barrier(group=control_group())
    finally:
        faulthandler.cancel_dump_traceback_later();dist.destroy_process_group()
        faulthandler.disable();fault.close()


if __name__=='__main__':main()
