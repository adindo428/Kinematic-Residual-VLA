"""Decoded full-data native KRV benchmark; fresh process for every runtime."""
import json,logging,os,time
from pathlib import Path
import torch,torch.distributed as dist
from core.dom_distributed_training import distributed_loader,wrap_policy,updates
from core.dom_distributed import control_group,leader_call
from core.dom_training import build_dataset,build_policy,load_family,make_optimizer,make_scheduler,seed_all,make_loader
from utils.dom_experiment import digest,atomic_json

def benchmark(family,runtime,device):
 family=Path(family);root=family.parent;manifest=load_family(family);seed_all();rank=dist.get_rank();world=dist.get_world_size()
 properties=torch.cuda.get_device_properties(device);assert properties.name=='NVIDIA H200' and world==2
 assert runtime['microbatch']<=176 and 3*runtime['workers']+2*runtime['main_threads']<=32
 hardware={'name':properties.name,'total_memory':properties.total_memory,'torch':torch.__version__,'cuda':torch.version.cuda,'capability':[properties.major,properties.minor]}
 key=digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':'krv100'})[:16]
 torch.cuda.reset_peak_memory_stats(device)
 data=build_dataset(family,json.load(open(family/'split.json'))['splits']['train'],training=True)
 assert len(data)==2406773
 val_loader=None
 if rank==0:
  validation=build_dataset(family,json.load(open(family/'split.json'))['splits']['validation'])
  val_loader=make_loader(validation,runtime,indices=json.load(open(family/'validation_anchors.json')));next(iter(val_loader))
 dist.barrier(group=control_group())
 policy=build_policy(family,'krv100',device);model=wrap_policy(policy,device)
 optimizer=make_optimizer(policy,3e-5,runtime['fused']);scheduler=make_scheduler(optimizer,68380)
 torch.manual_seed(42+rank);torch.cuda.manual_seed(42+rank)
 progress={'epoch':0,'cursor':0,'updates':0,'samples':0}
 loader=distributed_loader(data,runtime,sampling='frames');iterator=updates(model,optimizer,scheduler,loader,progress,runtime['precision'],device)
 records=[];started=None
 for record in iterator:
  logging.info('runtime=%s rank=%d update=%d rate=%g',key,rank,progress['updates'],record['samples_per_second'])
  if progress['updates']==3:
   torch.cuda.synchronize(device);started=time.monotonic()
  elif progress['updates']>3:records.append(record)
  if progress['updates']==11:
   torch.cuda.synchronize(device);ended=time.monotonic();elapsed=ended-started;break
 iterator.close();assert progress['updates']==11 and progress['samples']==3872
 peak=torch.cuda.max_memory_reserved(device);ram=int(os.environ.get('KRV_PEAK_RSS','0'));assert ram>0
 values=torch.tensor([elapsed,peak,ram],dtype=torch.float64)
 dist.all_reduce(values[:2],op=dist.ReduceOp.MAX,group=control_group());dist.all_reduce(values[2:],op=dist.ReduceOp.SUM,group=control_group())
 limit=int(os.environ['SLURM_MEM_PER_NODE'])*1024**2
 safe=float(values[1])<=.85*properties.total_memory and float(values[2])<=.85*limit
 report={'runtime':runtime,'hardware':hardware,'manifest':manifest['fingerprint'],'source':manifest['source_sha256'],'warmup_updates':3,'measured_updates':8,'measured_samples':2816,'decoded_data':True,'global_batch':352,'elapsed_seconds':float(values[0]),'training_window_monotonic_start':started,'training_window_monotonic_end':ended,'samples_per_second':2816/float(values[0]),'max_gpu_reserved_bytes':float(values[1]),'sum_peak_process_worker_rss_bytes':float(values[2]),'node_memory_limit':limit,'safe':safe,'rank':rank,'last_update':records[-1],'job_id':os.environ['SLURM_JOB_ID'],'benchmark_only_not_readiness':True}
 report['fingerprint']=digest(report)
 leader_call(lambda:atomic_json(root/'tuning/benchmarks'/f'{key}.json',report))
