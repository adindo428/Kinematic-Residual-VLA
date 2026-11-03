from pathlib import Path
import json,copy
here=Path(__file__).resolve().parent;registry=here.parent/'training_run_register/registry.json'
r=json.loads(registry.read_text());delta=json.loads((here/'delta-deployment.json').read_text());pal=json.loads((here/'deployment.json').read_text());submissions=json.loads((here/'replacement-submissions.json').read_text())
for x in r['runs']:
 if str(x.get('job_id')) in ['3360409','17017544','17091380','17091381']:
  x.update(superseded=True,group='historical')
  if x['cluster']=='DeltaAI':
   x.update(scheduler_state='FAILED',stop_reason='Published episode 195603 third-person MP4 has no frames (48 bytes)',note='FAILED after 45m29s: 80 updates/28160 samples logged, initialization-only checkpoint. Six empty upstream videos identified; approved exclusion is paired on both servers.')
  else:x.update(note='Cancelled before training to apply user-approved six-episode exclusion; replaced by clean-data allocation alternatives.',scheduler_state='CANCELLED')
rule={'enabled':True,'metric':'h20/action_mse','min_epochs':5,'patience':3,'relative_min_delta':.005}
newdelta={'cluster':'DeltaAI','job_id':None,'name':'domv100','method':'DOM + velocity','group':'production','purpose':'DOM manipulation training','world_size':4,'gpu':'GH200','lr':3e-5,'target_epochs':100,'early_stopping':rule,'family':delta['family'],'run_directory':delta['family']+'/runs/domv100','source_sha256':delta['source_sha256'],'manifest_fingerprint':delta['manifest'],'recipe_fingerprint':delta['recipe'],'split_sha256':delta['split_sha256'],'training_started':False,'production_checkpoint_present':False,'production_progress':{'epochs':0,'samples':0,'optimizer_updates':0},'note':'Approved six-episode exclusion; bounded corrected readiness before dependency-free production submission.'}
if not any(x.get('cluster')=='DeltaAI' and x.get('name')=='domv100' and not x.get('superseded') and x.get('group')=='production' for x in r['runs']):r['runs'].append(newdelta)
for entry in submissions['submissions']:
 old=next(x for x in r['runs'] if str(x.get('job_id'))==entry['replaces']);new={k:copy.deepcopy(v) for k,v in old.items() if k in ('cluster','name','variant','wandb_name','method','world_size','gpu','lr','target_epochs','early_stopping','runtime','allocation_candidate','progress_scope')}
 new.update(job_id=entry['job_id'],name=entry['name'],group='production',superseded=False,purpose='DOM manipulation training',family=pal['family'],run_directory=pal['family']+'/runs/krv100',source_directory=pal['source'],working_directory=pal['source'],source_sha256=pal['source_sha256'],recipe_fingerprint=pal['recipe'],manifest_fingerprint=pal['manifest'],split_sha256=pal['split_sha256'],scheduler_state='PENDING',replaces=entry['replaces'],training_started=False,production_checkpoint_present=False,production_progress={'epochs':0,'samples':0,'optimizer_updates':0},note='User-approved identical six-episode exclusion, coordinated loader failure propagation, checkpoints every 25 updates during first 1000 updates. Same scientific recipe and fixed validation/test. Allocation alternatives share one locked krv100 output.')
 prefix='dom100-ddp' if entry['name']=='krv100' else entry['name']
 new['stdout']=pal['source']+'/'+prefix+'-'+entry['job_id']+'.out';new['stderr']=pal['source']+'/'+prefix+'-'+entry['job_id']+'.err'
 if entry['name']!='krv100':new['alternate_of']=submissions['submissions'][0]['job_id']
 if not any(str(x.get('job_id'))==entry['job_id'] and x['cluster']=='Palmetto' for x in r['runs']):r['runs'].append(new)
r['active_source_sha256']=pal['source_sha256'];r['active_recipe_fingerprint']=pal['recipe'];r['data_correction']={'excluded_episode_ids':[194633,194760,195154,195405,195417,195603],'train_episodes':197201,'train_frames':22370523,'retained_episodes':201227,'retained_frames':22828297,'split_sha256':pal['split_sha256'],'user_approved':True,'applied_to_both_servers':True}
registry.write_text(json.dumps(r,indent=2)+'\n')
p=here.parent/'training_run_register/delta_probe.py';t=p.read_text();old="workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-bnfix-domv-20261011/status.json')";t=t.replace(old,"workflow=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-video-clean-domv-20261012/status.json')\nif not workflow.exists():"+old);p.write_text(t)
p=here.parent/'training_run_register/refresh.py';t=p.read_text();t=t.replace("active=next(r for r in reg['runs'] if r.get('job_id')=='16869869')","active=next(r for r in reg['runs'] if r.get('cluster')=='Palmetto' and r.get('name')=='krv100' and r.get('group')=='production' and not r.get('superseded'))")
t=t.replace("'Current Palmetto job: `16869869` (`krv100`).", "'Current Palmetto job: `'+str(active['job_id'])+'` (`krv100`).")
t=t.replace('201,233 episodes and 22,828,959 frames; training uses 197,207 episodes and 22,371,185 frames. The 6,073 excluded episodes are placement demonstrations.', '201,227 episodes and 22,828,297 frames; training uses 197,201 episodes and 22,370,523 frames. The original 6,073 unavailable placement episodes and six further episodes with empty published videos are excluded. The six-episode correction was approved on October 12 and applied on both servers.')
p.write_text(t)
print(json.dumps({'new_delta_family':delta['family'],'new_palmetto_jobs':[e['job_id'] for e in submissions['submissions']],'matching_split_sha256':pal['split_sha256']}))
