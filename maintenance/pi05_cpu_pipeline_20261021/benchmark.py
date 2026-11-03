"""Bounded, isolated pi05 CPU-pipeline experiment, never a production run."""
import os,sys,json,time,hashlib,gc,statistics
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'config.json').read_text());ROOT=Path(CONFIG['training_root'])
sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from common import sha,atomic,fingerprint
from data import DOM,Batches
from train import optimizer_for,load,update,rng_state,restore_rng,worker_init
from runtime import Resources
from candidate import CandidatePolicy,PromptCollator,TimedIterator,CUDAPrefetch,state_prompt

def require(condition,message):
 if not condition:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text())
def freeze():
 out=ROOT/'runs'/CONFIG['variant'];dest=HERE/'snapshot';dest.mkdir(exist_ok=True)
 if (dest/'manifest.json').exists():
  m=read(dest/'manifest.json');require(sha(m['checkpoint'])==m['checkpoint_sha256'],'Benchmark snapshot changed');return m
 for attempt in range(5):
  before=read(out/'latest.json');copy=dest/('checkpoint-'+str(time.time_ns())+'.pt')
  os.link(out/'latest.pt',copy)
  after=read(out/'latest.json')
  c=torch.load(copy,map_location='cpu',mmap=True,weights_only=False)
  okay=before==after and c['identity']==before['identity'] and c['progress']==before['progress']
  del c;gc.collect()
  if not okay:copy.unlink();continue
  require(before['identity']==CONFIG['production_identity'],'Wrong canonical pi05 KR snapshot')
  m={'checkpoint':str(copy),'checkpoint_sha256':sha(copy),'identity':before['identity'],'progress':before['progress'],'parent_job_id':'17782830','at':datetime.now(timezone.utc).isoformat(),'hard_link_to_atomically_replaced_latest_inode':True,'production_untouched':True}
  atomic(dest/'manifest.json',m);return m
 raise ValueError('Unable to reconcile latest production checkpoint snapshot')

def make_loader(data,runtime,progress,collator=None):
 return torch.utils.data.DataLoader(data,batch_sampler=Batches(len(data),progress['epoch'],progress['cursor'],32,16,dist.get_rank(),2),num_workers=runtime['workers'],generator=torch.Generator().manual_seed(42+progress['epoch']),pin_memory=True,worker_init_fn=worker_init,collate_fn=collator,prefetch_factor=runtime['prefetch'],persistent_workers=True,multiprocessing_context='spawn')

def close_iterator(obj):
 if hasattr(obj,'_shutdown_workers'):obj._shutdown_workers()

def equivalent(policy,model,data,progress,runtime):
 rank=dist.get_rank();collate=PromptCollator(ROOT,policy.policy.config.tokenizer_max_length)
 batches=make_loader(data,runtime,progress);it=iter(batches);batch=next(it)
 candidate=collate.prepare_batch(batch)
 old,_=super(CandidatePolicy,policy).prepare(batch);new,_=policy.prepare(candidate)
 require(old.keys()==new.keys(),'Prepared batch keys differ')
 tensor_mismatches=[]
 for k in old:
  if torch.is_tensor(old[k]):
   if not torch.equal(old[k],new[k]):tensor_mismatches.append(k)
  else:require(old[k]==new[k],'Task metadata changed')
 require(all('observation.language.' in k for k in tensor_mismatches),'Non-token prepared tensor changed')
 # Stress every exact bin edge and nearby values in physical state coordinates.
 offsets,scales=[getattr(policy,n) for n in policy._affine_buffers['observation.state']]
 normalized=torch.linspace(-1,1,257,device='cpu')[:,None].expand(-1,6).contiguous()
 normalized=torch.cat([normalized,torch.nextafter(normalized,torch.full_like(normalized,float('inf'))),torch.nextafter(normalized,torch.full_like(normalized,float('-inf')))],0)
 raw=normalized*scales.cpu()+offsets.cpu()
 tasks=['pick']*len(raw)
 expected=state_prompt(tasks,policy.affine(raw.to('cuda'),'observation.state'))
 actual=state_prompt(tasks,(raw-collate.offset)/collate.scale)
 boundaries_match=actual==expected
 eligible=not tensor_mismatches and boundaries_match
 # Same native model, inputs and RNG: compare loss and every trainable gradient.
 state=rng_state();model.zero_grad(set_to_none=True);compared=0;gradients_match=False
 if eligible:
  with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):a=model(batch)
  a.backward();grad=[p.grad.detach().clone() if p.grad is not None else None for p in model.parameters() if p.requires_grad]
  model.zero_grad(set_to_none=True);restore_rng(state)
  with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):b=model(candidate)
  b.backward();gradients_match=bool(torch.allclose(a,b,rtol=1e-4,atol=1e-6))
  for saved,p in zip(grad,[p for p in model.parameters() if p.requires_grad]):
   gradients_match=gradients_match and ((saved is None)==(p.grad is None))
   if saved is not None and p.grad is not None:
    gradients_match=gradients_match and bool(torch.allclose(saved,p.grad,rtol=1e-4,atol=1e-6));compared+=1
  del grad
 eligible=eligible and gradients_match
 model.zero_grad(set_to_none=True);restore_rng(state);del old,new,candidate,batch
 close_iterator(it);del it,batches;gc.collect();torch.cuda.empty_cache();dist.barrier()
 return {'rank':rank,'actual_batch_size':16,'synthetic_boundary_states':len(raw),'prepared_tensors_exact':not tensor_mismatches,'token_mismatches':tensor_mismatches,'bin_boundaries_exact':boundaries_match,'loss_gradient_consistency':gradients_match,'candidate_eligible':eligible,'trainable_gradient_tensors_compared':compared}

def run():
 dist.init_process_group('nccl');rank=dist.get_rank();require(dist.get_world_size()==2,'Two GPUs required')
 torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',int(os.environ['LOCAL_RANK']));props=torch.cuda.get_device_properties(device)
 require(props.name=='NVIDIA H200' and props.total_memory>=120*1024**3,'Actual H200 required')
 require(str(torch.__version__)==CONFIG['torch_version'],'Torch version differs')
 for n,h in read(HERE/'code-manifest.json')['files'].items():require(sha(HERE/n)==h,'Diagnostic source changed')
 parent_manifest=read(ROOT/'source-manifest.json');require(parent_manifest['source_sha256']==CONFIG['training_source_sha256'],'Wrong frozen training source')
 for n,h in parent_manifest['files'].items():require(sha(ROOT/n)==h,'Production training source changed')
 for n,h in read(ROOT/'preparation-manifest.json')['files'].items():require(sha(ROOT/n)==h,'Preparation changed')
 if rank==0:freeze()
 dist.barrier();snapshot=read(HERE/'snapshot/manifest.json');recipe=read(ROOT/'recipes'/(CONFIG['variant']+'.json'))
 torch.set_num_threads(2);data=DOM(ROOT,max_delay=recipe['max_delay'],training=True)
 require(len(data)==22370201 and recipe['global_batch']==32,'Audited data/batch differs')
 policy=CandidatePolicy(ROOT,recipe,device);model=DDP(policy,device_ids=[int(os.environ['LOCAL_RANK'])],find_unused_parameters=True,gradient_as_bucket_view=True)
 opt,sched=optimizer_for(policy,recipe)
 progress=load(Path(snapshot['checkpoint']),policy,opt,sched,'diagnostic-no-production-identity',rank,snapshot['identity'])
 require(progress==snapshot['progress'] and sched.last_epoch==progress['updates'],'Restored state differs')
 proof=equivalent(policy,model,data,progress,dict(workers=8,prefetch=2))
 proofs=[None,None];dist.all_gather_object(proofs,proof)
 cpu_eligible=all(p['candidate_eligible'] for p in proofs)
 if rank==0:atomic(HERE/'equivalence.json',{'passed':cpu_eligible,'per_rank':proofs,'snapshot':snapshot,'not_production_readiness':True})
 results=[]
 variants=[dict(name='control_workers8',cpu_prompts=False,prefetch_gpu=False,workers=8,prefetch=2),dict(name='worker_prompts8',cpu_prompts=True,prefetch_gpu=False,workers=8,prefetch=2),dict(name='worker_prompts8_cuda_prefetch',cpu_prompts=True,prefetch_gpu=True,workers=8,prefetch=2),dict(name='worker_prompts12_cuda_prefetch',cpu_prompts=True,prefetch_gpu=True,workers=12,prefetch=2),dict(name='control_workers8_repeat',cpu_prompts=False,prefetch_gpu=False,workers=8,prefetch=2)]
 if not cpu_eligible:
  variants=[variants[0],dict(name='workers12_control_prompt',cpu_prompts=False,prefetch_gpu=False,workers=12,prefetch=2),variants[-1]]
 for runtime in variants:
  require(2*(runtime['workers']+2)<=32,'CPU allocation oversubscribed')
  progress=load(Path(snapshot['checkpoint']),policy,opt,sched,'diagnostic',rank,snapshot['identity'])
  collator=PromptCollator(ROOT,policy.policy.config.tokenizer_max_length) if runtime['cpu_prompts'] else None
  batches=make_loader(data,runtime,progress,collator);base_it=iter(batches);timed=TimedIterator(base_it)
  iterator=CUDAPrefetch(timed,device) if runtime['prefetch_gpu'] else timed
  for _ in range(10):update(model,opt,sched,iterator,1,recipe['clip'])
  torch.cuda.synchronize();dist.barrier();timed.wait_seconds=0;timed.count=0
  startup=torch.cuda.max_memory_reserved(device);torch.cuda.reset_peak_memory_stats(device)
  resources=Resources(gpu=rank==0);resources.__enter__();torch.cuda.synchronize();started=time.perf_counter()
  for _ in range(200):update(model,opt,sched,iterator,1,recipe['clip'])
  torch.cuda.synchronize();seconds=time.perf_counter()-started;resources.__exit__()
  metrics=torch.tensor([seconds,timed.wait_seconds,max(startup,torch.cuda.max_memory_reserved(device)),resources.peak_rss],device=device,dtype=torch.float64)
  all_metrics=[torch.empty_like(metrics) for _ in range(2)];dist.all_gather(all_metrics,metrics)
  rows=[m.tolist() for m in all_metrics];wall=max(r[0] for r in rows);peak=max(r[2] for r in rows);rss=sum(r[3] for r in rows)
  result={'runtime':runtime,'timed_updates':200,'global_batch':32,'seconds_slowest_rank':wall,'samples_per_second':6400/wall,'per_rank_seconds':[r[0] for r in rows],'per_rank_dataloader_wait_seconds':[r[1] for r in rows],'gpu_peak_reserved_bytes':peak,'total_gpu_bytes':props.total_memory,'node_conservative_rss_bytes':rss,'node_ram_limit_bytes':256*1024**3,'fifteen_percent_headroom_pass':peak<=.85*props.total_memory and rss<=.85*256*1024**3,'mean_gpu_utilization_percent_rank0_sampler':resources.mean_gpu_utilization,'mean_cpu_busy_cores_rank_local':resources.mean_cpu_busy_cores,'same_production_snapshot':snapshot,'includes_checkpoints':False,'warmup_updates':10,'not_production_readiness':True}
  if rank==0:results.append(result);atomic(HERE/'benchmark-results.json',results);print(json.dumps({k:v for k,v in result.items() if k!='same_production_snapshot'}),flush=True)
  del iterator,timed;close_iterator(base_it);del base_it,batches;gc.collect();torch.cuda.empty_cache();dist.barrier()
 # Separate profiler pass: excluded from throughput measurements.
 progress=load(Path(snapshot['checkpoint']),policy,opt,sched,'diagnostic',rank,snapshot['identity'])
 batches=make_loader(data,dict(workers=8,prefetch=2),progress);it=iter(batches)
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
  for _ in range(3):
   with torch.profiler.record_function('decoded_update_with_loader_wait'):update(model,opt,sched,it,1,recipe['clip'])
   prof.step()
 torch.cuda.synchronize()
 if rank==0:
  rows=[{'name':e.key,'calls':e.count,'cpu_total_us':e.cpu_time_total,'self_cpu_us':e.self_cpu_time_total,'device_total_us':getattr(e,'device_time_total',0)} for e in prof.key_averages()]
  atomic(HERE/'profiler-summary.json',sorted(rows,key=lambda r:r['self_cpu_us'],reverse=True)[:80])
  prof.export_chrome_trace(str(HERE/'control-trace.json'))
  control=statistics.median([r['samples_per_second'] for r in results if r['runtime']['name'].startswith('control_')]);candidates=[r for r in results if not r['runtime']['name'].startswith('control_') and r['fifteen_percent_headroom_pass']];best=max(candidates,key=lambda r:r['samples_per_second']) if candidates else None
  atomic(HERE/'complete.json',{'complete':True,'job_id':os.environ['SLURM_JOB_ID'],'at':datetime.now(timezone.utc).isoformat(),'control_samples_per_second_median':control,'best_candidate':best,'candidate_speedup':best['samples_per_second']/control if best else None,'control_drift_ratio':results[-1]['samples_per_second']/results[0]['samples_per_second'],'production_untouched':True,'production_source_identity_unchanged':True,'not_production_readiness':True,'no_wandb_run_created':True})
 close_iterator(it);dist.barrier();dist.destroy_process_group()
if __name__=='__main__':
 try:run()
 except Exception as e:
  atomic(HERE/('error-'+os.environ.get('SLURM_PROCID',os.environ.get('LOCAL_RANK','unknown'))+'.json'),{'type':type(e).__name__,'message':str(e),'production_untouched':True});raise
