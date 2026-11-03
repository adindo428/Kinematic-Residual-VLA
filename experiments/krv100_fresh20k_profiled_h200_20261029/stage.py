"""Stage a new, independently frozen fresh20k KRV100 family on Palmetto."""
import hashlib,json,shutil,sys,compileall
from pathlib import Path
H=Path(__file__).resolve().parent
R=Path('/scratch/aparame/Dynamic_VLA/krv100-fresh20k-profiled-h200-20261029')
OLD=Path('/scratch/aparame/Dynamic_VLA/dom100-audited-20261012')
SRC=Path('/scratch/aparame/Dynamic_VLA/main8-20k-10epochs-20261021-r2/delta-domv-f991015a')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def replace(p,a,b):
 s=p.read_text();assert s.count(a)==1,(str(p),a[:60],s.count(a));p.write_text(s.replace(a,b))
def main():
 assert not (R/'source').exists(),'Already staged; reconcile rather than overwrite'
 pc=json.load(open(H/'pc-reference.json'))['config']; base=pc['manifest']
 code=R/'source';code.mkdir(parents=True)
 for name in ('core','utils','policies','proactive_vla','scripts','configs'):
  shutil.copytree(SRC/name,code/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'),symlinks=False)
 # Keep the established native math, full-anchor sampler and coordinated saves.
 # Increase readiness to 19 full updates plus an uneven tail; count startup peaks.
 p=code/'core/dom_distributed_training.py'
 replace(p,'"""Two real updates including an uneven tail, followed by exact RNG reload."""','"""Twenty decoded updates including an uneven tail, then strict reload."""')
 replace(p,"data = build_dataset(root,json.loads((root/'pilot_episodes.json').read_text())[:8],training=True)","torch.cuda.reset_peak_memory_stats(device)\n        data = build_dataset(root,json.loads((root/'split.json').read_text())['splits']['train'],training=True)")
 replace(p,"        torch.cuda.reset_peak_memory_stats(device)\n        loader = distributed_loader(data,runtime,lengths=[352+world-1])","        loader = distributed_loader(data,runtime,lengths=[352*19+world-1])")
 replace(p,"progress['updates'] != 2 or progress['samples'] != 352+world-1","progress['updates'] != 20 or progress['samples'] != 352*19+world-1")
 replace(p,"ram = process.memory_info().rss+sum(p.memory_info().rss for p in process.children(recursive=True))","ram = max(process.memory_info().rss+sum(p.memory_info().rss for p in process.children(recursive=True)), int(os.environ.get('KRV_PEAK_RSS','0')))" )
 replace(p,"name=variant,id=run_id,resume='allow',dir=str(destination)","name='fresh20k-palmetto-'+variant+'-s42',id=run_id,resume='allow',dir=str(destination)")
 replace(p,"if prediction.shape[1] != 20 or not torch.isfinite(prediction).all():","if prediction.shape[1:] != (20,7) or not torch.isfinite(prediction).all():")
 replace(p,"            restore_distributed(state)\n            del state","            restore_distributed(state)\n            leader_call(lambda:atomic_json(destination/'resume-restored.json',{'progress':progress,'optimizer':True,'scheduler':True,'rank_rng':True,'data_cursor':True,'world_size':dist.get_world_size(),'wandb_id':wandb_id,'job_id':os.environ.get('SLURM_JOB_ID')}))\n            del state")
 # Standing policy: native generation checkpoints retain independent latest/best pointers.
 checkpoint_source=(code/'core/dom_checkpoint.py').read_text()
 assert 'root/"latest.json"' in checkpoint_source and 'root/"best.json"' in checkpoint_source
 assert '"latest.json", "best.json", "final.json"' in checkpoint_source
 replace(p,"            value = metrics['h20/action_mse']\n", "            value = metrics['h20/action_mse']\n            if not math.isfinite(value):raise ValueError('Best-checkpoint validation metric is nonfinite')\n")
 # Avoid transient duplicated Adam moments while restoring an already populated optimizer.
 replace(code/'core/dom_checkpoint.py',"    optimizer.load_state_dict(state[\"optimizer\"])","    optimizer.state.clear()\n    optimizer.load_state_dict(state[\"optimizer\"])\n    if state['scheduler']['last_epoch'] != state['progress']['updates']:\n        raise ValueError('Checkpoint scheduler and update counters differ')")
 shutil.copy2(H/'entry.py',code/'scripts/fresh20k_krv.py')
 shutil.copy2(H/'benchmark.py',code/'scripts/krv_benchmark.py')
 meta=R/'family';meta.mkdir()
 original=json.load(open(OLD/'manifest.json'))
 for name in original['artifacts']:
  if name.startswith('initialization/'):continue
  shutil.copy2(OLD/name,meta/name)
 shutil.copy2('/scratch/aparame/Dynamic_VLA/main8-h200-only-20261022/split.json',meta/'split.json')
 assert sha(meta/'split.json')==base['split_sha256']=='98436815f85fbfc78d48e4332e8603684f6bae1620400b181767bab1c5736d88'
 split=json.load(open(meta/'split.json'))['splits'];selected=set(sum(split.values(),[]))
 episodes=[r for r in json.load(open(meta/'episodes.json')) if r['episode_index'] in selected]
 assert {k:len(v) for k,v in split.items()}==base['counts']
 assert len(selected)==24026 and sum(r['length'] for r in episodes if r['episode_index'] in set(split['train']))==2406773
 write(meta/'episodes.json',episodes);write(meta/'pilot_episodes.json',split['train'][:64])
 (meta/'phase_cache').symlink_to(OLD/'phase_cache',target_is_directory=True)
 (meta/'initialization').symlink_to(OLD/'initialization',target_is_directory=True)
 assert sha(meta/'initialization/model.safetensors')==base['weights_sha256']=='e3da8b1e0326d7a3ee6873f7fa17cdf7015201251901be5107377cb851ce3af8'
 sys.path.insert(0,str(code));from utils.dom_experiment import source_digest
 m={k:base[k] for k in ('dataset','version','fps','dataset_revision','model_revision','normalization','prediction_horizon','execution_horizon','velocity_mean','velocity_std','weights_sha256','counts','episodes','frames','train_frames','split_sha256')}
 m.update(dataset_root=original['dataset_root'],initialization=str(meta/'initialization'),phase_config=original['phase_config'],source_sha256=source_digest(code),dataset_selection_mode='fresh_finetuning_on_committed_20k_split')
 m['artifacts']={p.name:sha(p) for p in meta.glob('*.json')};m['artifacts']['initialization/config.json']=sha(meta/'initialization/config.json')
 m['fingerprint']=digest(m);write(meta/'manifest.json',m)
 recipe=dict(pc['recipe']);recipe.pop('fingerprint');recipe['manifest']=m['fingerprint'];recipe['early_stopping']={**recipe['early_stopping'],'enabled':False,'min_epochs':10};recipe.pop('evaluation_plan_sha256',None)
 recipe['selection']='Fresh released initialization; exact20k split; all anchors including partial final batch; ten full epochs; local-PC simulator evaluation only.'
 recipe['fingerprint']=digest(recipe);write(meta/'recipe.json',recipe)
 write(R/'runtime.json',{'microbatch':32,'workers':8,'prefetch':2,'precision':'bf16','pin_memory':True,'fused':True,'main_threads':2})
 write(R/'source-manifest.json',{'source_sha256':m['source_sha256'],'manifest':m['fingerprint'],'recipe':recipe['fingerprint'],'weights_sha256':m['weights_sha256'],'split_sha256':m['split_sha256'],'updates_per_epoch':6838,'total_updates':68380,'total_samples':24067730,'initialization':'released pretrained weights; not a production or readiness checkpoint','pc_reference':'664747c0','differences_from_pc':['H200 hardware/runtime microbatch32 versus PC40','early stopping disabled to guarantee ten full epochs','independent W&B ID; PC queue untouched'],'files':{str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')}})
 assert compileall.compile_dir(str(code),quiet=1)
 print(json.dumps({'staged':str(R),'manifest':m['fingerprint'],'source':m['source_sha256'],'total_updates':68380}))
if __name__=='__main__':main()
