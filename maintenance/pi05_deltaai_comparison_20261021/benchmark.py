"""Same-allocation, full-state reset π0.5 KR execution comparison on GH200."""
import os,sys,json,time,gc,contextlib,hashlib,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE/'source'
sys.path[:0]=[str(HERE/'pinned'),str(ROOT)]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common import atomic
from data import DOM,Batches
from train import optimizer_for,load,rng_state,restore_rng,worker_init
from runtime import Resources
from candidate import CandidatePolicy,PromptCollator,TimedIterator,CUDAPrefetch,state_prompt
from test_autocast import verify_autocast
from test_optimizer import verify_fused

VARIANTS={
 'control':dict(workers=8,tokens=False,prefetch=False,cache=False,checkpoint=True),
 'cpu_tokens':dict(workers=8,tokens=True,prefetch=False,cache=False,checkpoint=True),
 'cpu_tokens_prefetch':dict(workers=8,tokens=True,prefetch=True,cache=False,checkpoint=True),
 'workers12_prefetch':dict(workers=12,tokens=True,prefetch=True,cache=False,checkpoint=True),
 'autocast_cache':dict(workers=8,tokens=False,prefetch=False,cache=True,checkpoint=True),
 'no_recompute':dict(workers=8,tokens=False,prefetch=False,cache=False,checkpoint=False),
 'control_repeat':dict(workers=8,tokens=False,prefetch=False,cache=False,checkpoint=True),
}
IDENTITY='14ef74c0d9c21b29a25c077fb1e13794d94af2fe9d306729f70b93d9339fe025'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def checkpoint_mode(p,on):
 m=p.policy.model
 (m.gradient_checkpointing_enable if on else m.gradient_checkpointing_disable)()
def loader(data,progress,workers,collator=None):
 return torch.utils.data.DataLoader(data,batch_sampler=Batches(len(data),progress['epoch'],progress['cursor'],32,16,dist.get_rank(),2),num_workers=workers,generator=torch.Generator().manual_seed(42+progress['epoch']),pin_memory=True,worker_init_fn=worker_init,collate_fn=collator,prefetch_factor=2,persistent_workers=True,multiprocessing_context='spawn')
def close(it):
 if hasattr(it,'_shutdown_workers'):it._shutdown_workers()
def update(model,opt,sched,it,cache):
 opt.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=cache):loss=model(next(it))
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
 require(bool(torch.isfinite(norm)&torch.isfinite(loss)),'Nonfinite loss/gradient')
 opt.step();sched.step();return loss.detach()
def proof(policy,model,data,progress,v):
 batches=loader(data,progress,8);it=iter(batches);batch=next(it)
 collator=PromptCollator(ROOT,policy.policy.config.tokenizer_max_length)
 alternative=collator.prepare_batch(batch) if v['tokens'] else batch
 old,_=super(CandidatePolicy,policy).prepare(batch);new,_=policy.prepare(alternative)
 exact=all(torch.equal(old[k],new[k]) for k in old if torch.is_tensor(old[k]))
 require(exact,'Prepared tensors/tokens differ')
 if v['tokens']:
  off,scale=[getattr(policy,n) for n in policy._affine_buffers['observation.state']]
  edge=torch.linspace(-1,1,257)[:,None].expand(-1,6).contiguous()
  edge=torch.cat([edge,torch.nextafter(edge,torch.full_like(edge,float('inf'))),torch.nextafter(edge,torch.full_like(edge,float('-inf')))])
  raw=edge*scale.cpu()+off.cpu();tasks=['pick']*len(raw)
  require(state_prompt(tasks,policy.affine(raw.cuda(),'observation.state'))==state_prompt(tasks,(raw-collator.offset)/collator.scale),'CPU quantization differs near bin edges')
 state=rng_state();model.zero_grad(set_to_none=True);checkpoint_mode(policy,True)
 with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):a=model(batch)
 a.backward();saved=[p.grad.detach().cpu() if p.grad is not None else None for p in model.parameters() if p.requires_grad]
 model.zero_grad(set_to_none=True);restore_rng(state);checkpoint_mode(policy,v['checkpoint'])
 with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=v['cache']):b=model(alternative)
 b.backward();ok=bool(torch.allclose(a,b,rtol=1e-4,atol=1e-6));count=0;worst=0.
 for x,p in zip(saved,[p for p in model.parameters() if p.requires_grad]):
  ok=ok and ((x is None)==(p.grad is None))
  if x is not None and p.grad is not None:
   y=p.grad.detach().cpu();ok=ok and bool(torch.allclose(x,y,rtol=1e-4,atol=1e-6));worst=max(worst,float((x-y).abs().max()));count+=1
 require(ok,'Native loss/gradient equivalence failed')
 result={'rank':dist.get_rank(),'prepared_tensors_exact':exact,'loss_allclose':True,'gradient_tensors':count,'max_gradient_absolute_difference':worst,'rtol':1e-4,'atol':1e-6}
 del saved,old,new,alternative,batch,a,b;model.zero_grad(set_to_none=True);restore_rng(state);close(it);del it,batches;gc.collect();torch.cuda.empty_cache();dist.barrier();return result
def main(name):
 dist.init_process_group('nccl');rank=dist.get_rank();require(dist.get_world_size()==2,'Exactly two ranks')
 torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',int(os.environ['LOCAL_RANK']));props=torch.cuda.get_device_properties(device)
 require('GH200' in props.name,'Expected actual DeltaAI GH200');torch.set_num_threads(2)
 verify_autocast(device);verify_fused(device)
 for n,h in json.loads((HERE/'manifest.json').read_text())['files'].items():require(hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,'Frozen diagnostic code changed')
 recipe=json.loads((ROOT/'recipes/pi05_kr.json').read_text());data=DOM(ROOT,max_delay=0,training=True);require(len(data)==22370201,'Audited anchor count differs')
 p=CandidatePolicy(ROOT,recipe,device);model=DDP(p,device_ids=[device.index],find_unused_parameters=True,gradient_as_bucket_view=True);opt,sched=optimizer_for(p,recipe)
 def reset():
  progress=load(HERE/'checkpoint.pt',p,opt,sched,'diagnostic',rank,IDENTITY)
  require(progress==dict(updates=60004,samples=1920128,epoch=0,cursor=1920128) and sched.last_epoch==60004,'Wrong restored state')
  return progress
 progress=reset();v=VARIANTS[name];require(2*(v['workers']+2)<=32,'CPU oversubscription')
 startup_resources=Resources(gpu=False);startup_resources.__enter__()
 evidence=proof(p,model,data,progress,v);proofs=[None]*2;dist.all_gather_object(proofs,evidence)
 if rank==0:atomic(HERE/f'{name}-equivalence.json',proofs)
 progress=reset();checkpoint_mode(p,v['checkpoint']);batches=loader(data,progress,v['workers'],PromptCollator(ROOT,p.policy.config.tokenizer_max_length) if v['tokens'] else None);base=iter(batches);timed=TimedIterator(base);it=CUDAPrefetch(timed,device) if v['prefetch'] else timed
 for _ in range(10):update(model,opt,sched,it,v['cache'])
 torch.cuda.synchronize();dist.barrier();startup_resources.__exit__()
 safety=torch.tensor([torch.cuda.max_memory_reserved(device)/props.total_memory,startup_resources.peak_rss/(256*1024**3)],device=device,dtype=torch.float64)
 dist.all_reduce(safety[:1],op=dist.ReduceOp.MAX);dist.all_reduce(safety[1:],op=dist.ReduceOp.SUM)
 require(bool((safety<=.85).all()),'Startup/proof/warmup exceeds unchanged 15 percent GPU/node memory headroom')
 resources=Resources(gpu=rank==0);resources.__enter__();timed.wait_seconds=0;timed.count=0;start=time.perf_counter()
 for _ in range(200):update(model,opt,sched,it,v['cache'])
 torch.cuda.synchronize();seconds=time.perf_counter()-start;resources.__exit__()
 m=torch.tensor([seconds,timed.wait_seconds,torch.cuda.max_memory_reserved(device),max(startup_resources.peak_rss,resources.peak_rss)],device=device,dtype=torch.float64);all_m=[torch.empty_like(m) for _ in range(2)];dist.all_gather(all_m,m);rows=[x.tolist() for x in all_m]
 result={'variant':name,'runtime':v,'samples_per_second':6400/max(r[0] for r in rows),'timed_updates':200,'warmup_updates':10,'global_batch':32,'microbatch_per_rank':16,'world_size':2,'per_rank':rows,'gpu_name':props.name,'gpu_total_bytes':props.total_memory,'gpu_peak_reserved_bytes':max(r[2] for r in rows),'node_conservative_rss_bytes':sum(r[3] for r in rows),'headroom_pass':max(r[2] for r in rows)<=.85*props.total_memory and sum(r[3] for r in rows)<=.85*256*1024**3,'gpu_utilization_rank0_sampler':resources.mean_gpu_utilization,'cpu_busy_cores_rank0':resources.mean_cpu_busy_cores,'includes_checkpoint_io':False,'equivalence':proofs,'torch_version':str(torch.__version__),'slurm_job_id':os.environ['SLURM_JOB_ID'],'not_production_readiness':True,'production_changed':False}
 if rank==0:atomic(HERE/f'{name}-result.json',result);print(json.dumps(result),flush=True)
 del it;close(base);dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 name=sys.argv[1]
 try:main(name)
 except Exception as e:
  atomic(HERE/f'{name}-error-{os.environ.get("LOCAL_RANK","unknown")}.json',{'type':type(e).__name__,'message':str(e)});raise
