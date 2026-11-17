"""Retain two verified900-trial cohorts; add disjoint repeats11--20 serially."""
import collections,copy,fcntl,inspect,json,os,signal,sys
from pathlib import Path
import run_closed_loop_pilot as runner
low=runner.low
ROOT=low.WORK/'evaluations/dom-krv-epoch4-1800-20261116'
CURRENT=low.WORK/'evaluations/krv-full-dom-ablation-20261115'
OLD=low.WORK/'evaluations/krv-full-original-dom-20261106/variants/krv100_after_epoch004_partial005_constant'
DIMS=['CR','DA','LS','VU','SR','MP','VG','MG','DR']

def read(p):return json.loads(p.read_text())
def write(p,x):low.write(p,x)
def records(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def recount_history():
 ledger=read(OLD/'resident_schedule/ledger.json');result={}
 for dim,sources in ledger['sources']['krv_latest_constant'].items():
  selected={}
  for src in sources:
   count=collections.Counter()
   for row in records(Path(src['folder'])/'trial_results.jsonl'):
    name=row['env_name'];count[name]+=1
    limit=src.get('counts',{}).get(name,0) if 'counts' in src else src.get('repeats',10)
    if count[name]>limit:continue
    key=(name,count[name]+src.get('repeat_offset',0));assert key not in selected;selected[key]=row
  assert len(selected)==100
  result[dim]={'successes':sum(bool(v['success']) for v in selected.values()),'trials':100}
 return result

def summarize_phase(phase):
 result=copy.deepcopy(read(ROOT/'retained900.json')[phase])
 path=ROOT/phase/'plan.json'
 if path.exists():
  for job in read(path)['jobs']:
   data=records(Path(job['folder'])/'trial_results.jsonl');assert len(data)<=job['repeats']
   d=result[job['dimension']];d['successes']+=sum(bool(x['success']) for x in data);d['trials']+=len(data)
 return result

def render():
 if not (ROOT/'retained900.json').exists():return
 results={key:summarize_phase(key) for key in ['krv_epoch4','dom']}
 lines=['# Full DOM extension to1,800 trials per model','','Retain the completed first900 trials; add repeats11–20 on the same90 configurations. Target:200 trials per category for each model. The extension now uses scene-first scheduling with both checkpoints resident and one active inference worker; completed blocks are retained. KRV uses its constant gate, with VPD and VI off.','','| Method | CR | DA | LS | VU | SR | MP | VG | MG | DR | Overall |','|---|'+'---:|'*10]
 for key,label in [('krv_epoch4','KRV epoch4 + partial5, constant gate'),('dom','DOM update184997')]:
  values=results[key];s=sum(v['successes'] for v in values.values());n=sum(v['trials'] for v in values.values())
  cells=[f"{100*values[d]['successes']/values[d]['trials']:.2f}% ({values[d]['successes']}/{values[d]['trials']})" for d in DIMS]
  lines.append('| '+label+' | '+' | '.join(cells)+f' | {100*s/n:.2f}% ({s}/{n}) |')
 lines+=['','Percentages use completed trials. First900 records are retained, never rerun.','', 'KRV first900 used its historical native streaming/RNG wrapper; its extension retains native streaming and ongoing RNG sampling. DOM retains the newer deterministic per-observation-noise wrapper used in its first900. Runtime sessions differ; these totals are not a newly matched3,600-trial experiment.','']
 history=(CURRENT/'history-audit/live-history-section.md').read_text()
 text='\n'.join(lines)+'\n'+history
 (ROOT/'live_results.md').write_text(text);(CURRENT/'live_results.md').write_text(text)
 write(ROOT/'category_scores.json',{'models':results,'per_category_target':200,'per_model_target':1800,'first900_retained':True})

def status(phase,**kwargs):
 phase_root=runner.ROOT
 write(phase_root/'status.json',{'stage':phase,'updated_utc':runner.stamp(),**kwargs})
 write(ROOT/'status.json',{'phase':phase_root.name,'stage':phase,'updated_utc':runner.stamp(),**kwargs});render()

def prepare():
 ROOT.mkdir(exist_ok=True,parents=True)
 if (ROOT/'extension-plan.json').exists():return
 base=read(CURRENT/'plan.json');assert read(CURRENT/'scope-completion.json')['verified']
 dom_rows={d:{'successes':0,'trials':0} for d in DIMS}
 for j in base['jobs']:
  if j['model']!='dom':continue
  data=records(Path(j['folder'])/'trial_results.jsonl');assert len(data)==10
  assert read(Path(j['folder'])/'resident_client_completion.json')['complete']
  dom_rows[j['dimension']]['trials']+=len(data);dom_rows[j['dimension']]['successes']+=sum(bool(v['success']) for v in data)
 assert all(v['trials']==100 for v in dom_rows.values())
 kr_rows=recount_history();assert sum(v['successes'] for v in kr_rows.values())==256
 write(ROOT/'retained900.json',{'krv_epoch4':kr_rows,'dom':dom_rows})
 (ROOT/'previous-live-results.md').write_text((CURRENT/'live_results.md').read_text())
 kr=copy.deepcopy(read(OLD/'resident_schedule/model.json'));kr.update(vi=False,vpd=False)
 dom=copy.deepcopy(next(m for m in base['models'] if m['key']=='dom'))
 for key,model in [('krv_epoch4',kr),('dom',dom)]:
  dest=ROOT/key;dest.mkdir(exist_ok=True)
  assert low.sha(Path(model['checkpoint'])/'model.safetensors')==model['weight_sha256'];assert low.sha(Path(model['checkpoint'])/'config.json')==model['config_sha256']
  jobs=[]
  for test in base['tests']:
   assert low.sha(test['path'])==test['config_sha256']
   folder=dest/'blocks'/test['name'];folder.mkdir(exist_ok=True,parents=True);(folder/'case.txt').write_text(test['path']+'\n')
   jobs.append({'id':key+'-p2-'+test['name'],'model':model['key'],'scene':test['path'],'scene_sha256':test['config_sha256'],'repeats':10,'repeat_offset':10,'folder':str(folder),'dimension':test['dimension'],'scored':True,'seed':test['seed'],'vi':False})
  files=[Path(__file__),low.REPO/'workstation/dom_repeat_extension_server.py',low.REPO/'workstation/resident_table1_client.py',low.REPO/'workstation/resident_table1_worker.py',low.REPO/'workstation/resident_table1_server.py']+list(Path(model['source']).rglob('*.py'))
  plan={'models':[model],'jobs':jobs,'tests':base['tests'],'saved_prompts':{},'success_history':{model['key']:[]},'files_sha256':{str(p):low.sha(p) for p in files},'protocol':{'new_trials':900,'retained_trials':900,'combined_target':1800,'global_repeat_range':[11,20],'per_category_target':200},'phase':key}
  # Existing prompt history is retained; second-pass keys are generated once.
  for path in (OLD/'resident_schedule').rglob('prompt_trace.jsonl'):
   for row in records(path):
    idx=row.get('global_trial',row['trial']);scene=next((t['path'] for t in base['tests'] if t['name']==row['env_name']),None)
    if scene:plan['saved_prompts'].setdefault(scene,{})[str(idx)]=row['text']
  write(dest/'plan.json',plan)
 write(ROOT/'extension-plan.json',{'phases':['krv_epoch4','dom'],'target_each':1800,'additional_trials_each':900,'original_DOM_root':str(CURRENT),'historical_KRV_root':str(OLD),'created_utc':runner.stamp()});render()

def verify(plan):
 seen=set()
 for job in plan['jobs']:
  data=records(Path(job['folder'])/'trial_results.jsonl');assert len(data)==10
  done=read(Path(job['folder'])/'resident_client_completion.json');assert done['complete'] and done['trials']==10
  for row in data:
   key=(row['env_name'],int(row['trial'])+10);assert key not in seen and 11<=key[1]<=20;seen.add(key)
 assert len(seen)==900
 write(runner.ROOT/'completion.json',{'verified':True,'new_trials':900,'combined_trials':1800});render()

def launch(phase):
 directory=ROOT/phase;plan=read(directory/'plan.json')
 if (directory/'completion.json').exists():assert read(directory/'completion.json')['verified'];return
 if phase=='dom':
  for path in (ROOT/'krv_epoch4/blocks').glob('*/prompt_trace.jsonl'):
   for row in records(path):
    scene=next(t['path'] for t in plan['tests'] if t['name']==row['env_name'])
    idx=row['global_trial'];assert 11<=idx<=20
    plan['saved_prompts'].setdefault(scene,{})[str(idx)]=row['text']
  write(directory/'plan.json',plan)
 runner.ROOT=directory;runner.status=status;runner.verify=verify
 code=inspect.getsource(ORIGINAL_LAUNCH).replace("'closed-loop-pilot'","'dom-krv-1800'").replace('total=75',"total=len(plan['jobs'])").replace('completed=75',"completed=len(plan['jobs'])")
 if phase=='krv_epoch4':
  code=code.replace('closed_loop_pilot_server.py','resident_table1_server.py').replace('closed_loop_pilot_client.py','resident_table1_client.py')
  code=code.replace('    client = [',"    server += ['--wrapped-server', str(low.REPO/'workstation/collection_matched_server.py')]\n    client = [")
 else:
  code=code.replace('closed_loop_pilot_server.py','dom_repeat_extension_server.py').replace('closed_loop_pilot_client.py','krv_profile_ablation_client.py')
 exec(compile(code,__file__,'exec'),runner.__dict__);runner.launch(plan)

ORIGINAL_LAUNCH=runner.launch

def main():
 prepare();signal.signal(signal.SIGTERM,lambda *_:runner.STOP.set());signal.signal(signal.SIGINT,lambda *_:runner.STOP.set())
 with (ROOT/'runner.lock').open('a') as own,(low.ROOT/'queue.lock').open('a') as gpu:
  fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while not runner.STOP.is_set():
   try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:runner.STOP.wait(5)
  for phase in ['krv_epoch4','dom']:
   if runner.STOP.is_set():return
   launch(phase)
  write(ROOT/'completion.json',{'verified':True,'trials_per_model':1800,'new_trials':1800});render()

if __name__=='__main__':main()
