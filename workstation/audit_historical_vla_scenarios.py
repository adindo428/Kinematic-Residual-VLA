#!/usr/bin/env python3
"""Compare existing VLA trials by exact scenario configuration; launch nothing."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys

REPO=Path(__file__).resolve().parents[1];WORK=REPO/'scratch/dom45'
OUT=WORK/'results/cross-vla-scenario-audit-20261102'
ANCHOR=WORK/'evaluations/krv100-fresh20k-palmetto-table1-20261101/results/best'
SPECS=[
 ('krv_best','KRV100 best (epoch 5)','eligible',ANCHOR),
 ('krv_latest','KRV100 latest (epoch 10)','eligible',WORK/'evaluations/krv100-fresh20k-palmetto-table1-20261101/results/latest'),
 ('dom_best','DOM best (fresh 20k)','eligible',WORK/'evaluations/multivla20k-20261029/results/dom100'),
 ('dom_latest','DOM latest (fresh 20k)','eligible',WORK/'evaluations/dom-latest-table1-20261031/results/dom100'),
 ('smol','SmolVLA baseline','eligible',WORK/'evaluations/smolvla-runtime-repair-20261102/results/smolvla_baseline'),
 ('smol_kr','SmolVLA + KR','eligible',WORK/'evaluations/smolvla-runtime-repair-20261102/results/smolvla_kr'),
 ('vlash_kr','VLASH + KR final','eligible_partial',WORK/'evaluations/completed-vlas-after-current-20261102/results/vlash_kr'),
 ('pi05','pi0.5 baseline final','eligible_pilot_only',WORK/'evaluations/completed-vlas-after-current-20261102/results/pi05_baseline'),
 ('dom_velocity','DOM + velocity','historical_excluded',WORK/'evaluations/domv-after-smolvla-20261102/results/domv100'),
 ('old_krv_best','Older KRV100 epoch 3','historical_excluded',WORK/'results/krv100-best-epoch003-task-table1-20261025'),
 ('old_krv_latest','Older KRV100 epoch 8.09','historical_excluded',WORK/'results/krv100-latest-epoch008p09-task-table1-20261025'),
 ('released','Released DOM','historical_released',WORK/'results/dom-paper45-a6000-20261010-02'),
 ('krv_epoch1','Older KRV velocity epoch 1','historical_budget_unverified',WORK/'results/krvla-velocity-epoch001-paper45-a6000-20261013-01'),
]


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(cfg):return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def records(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def describe(test):
 from scipy.spatial.transform import Rotation
 cfg=read(test['path']);scene=cfg['scene'];obj=scene['object'];v=obj['init_state']['lin_vel']
 robot=scene['robot']['init_state'];q=robot['rot'];inverse=Rotation.from_quat(q[1:]+q[:1]).inv()
 relative_position=inverse.apply([a-b for a,b in zip(obj['init_state']['pos'],robot['pos'])]).tolist()
 relative_velocity=inverse.apply(v).tolist()
 bodies={k:x for k,x in scene.items() if isinstance(x,dict) and x.get('class_type')=='isaaclab.assets.rigid_object.rigid_object:RigidObject'}
 return {'name':test['name'],'dimension':test.get('dimension'),'task':test.get('task',cfg['instruction']['task']).replace('-','_'),
  'canonical_sha256':canonical(cfg),'config_path':test['path'],'house_usd':Path(scene['house']['spawn']['usd_path']).name,
  'seed':cfg['seed'],'target_asset':Path(obj['spawn']['usd_path']).stem,
  'target_position_robot_m':relative_position,'target_velocity_robot_m_s':relative_velocity,
  'house_asset_path':scene['house']['spawn']['usd_path'],
  'objects':[Path(x['spawn']['usd_path']).name for k,x in bodies.items() if k.startswith('object')],
  'containers':[Path(x['spawn']['usd_path']).name for k,x in bodies.items() if k.startswith('container')],
  'rigid_count':len(bodies),'initial_velocity_world_m_s':v,'initial_speed_m_s':math.sqrt(sum(x*x for x in v)),
  'angular_damping':obj['spawn']['rigid_props'].get('angular_damping'),'horizon_s':cfg['episode_length_s'],
  'instruction_metadata':cfg['instruction'],'external_perturbation':'perturbation' in cfg['events']}


def load_run(key,label,group,root):
 proto=read(root/'protocol.json');tests={t['name']:t for t in proto['tests']}
 configs={name:canonical(read(t['path'])) for name,t in tests.items()}
 for t in tests.values():
  assert sha(t['path'])==t['config_sha256'], 'Recorded frozen config changed: '+t['path']
 data=defaultdict(list);invalid=[];phases=[]
 for mode in ['full','pilot']:
  for path in sorted((root/mode).rglob('trial_results.jsonl')):
   rows=records(path);valid=[r for r in rows if r['status'] in [0,1]]
   seen=set()
   for row in valid:
    token=(row['env_name'],row['trial']);assert token not in seen;seen.add(token)
    assert row['success']==(row['status']==0) and row['n_steps']>1
    assert row['task_time_s']>0 and abs(row['task_time_s']-row['n_steps']*row['step_dt_s'])<1e-7
   verified=(path.parent/'verification.json').exists()
   if verified:
    v=read(path.parent/'verification.json');assert v['verified'] and v['trials']==len(rows) and len(valid)==len(rows)
    log=(path.parent/'client.log').read_text()
    assert f'Evaluation completed. Total tests run: {len(rows)}' in log
   phase={'path':str(path),'mode':mode,'phase_verified':verified,'valid_records':len(valid),'invalid_records':len(rows)-len(valid)}
   phases.append(phase)
   invalid.extend({**r,'source':str(path)} for r in rows if r['status'] not in [0,1])
   for row in valid:
    data[(mode,row['env_name'])].append({**row,'phase_verified':verified,'source':str(path)})
 cells=[]
 for (mode,name),rows in data.items():
  assert name in tests
  cells.append({'mode':mode,'scenario':name,'canonical_sha256':configs[name],'successes':sum(r['success'] for r in rows),
   'trials':len(rows),'percent':100*statistics.mean(r['success'] for r in rows),'phase_verified':all(r['phase_verified'] for r in rows),
   'success_trial_ids':[r['trial'] for r in rows if r['success']],
   'success_times_s':[r['task_time_s'] for r in rows if r['success']]})
 cp=proto.get('checkpoint',{})
 return {'key':key,'label':label,'group':group,'root':str(root),'checkpoint':cp,'phases':phases,'cells':cells,'invalid_record_count':len(invalid)}


def cell(run,scenario,mode='full'):
 items=[x for x in run['cells'] if x['mode']==mode and x['scenario']==scenario['name'] and x['canonical_sha256']==scenario['canonical_sha256']]
 assert len(items)<=1
 return items[0] if items else None


def display(value):
 if not value:return '—'
 return f"{value['successes']}/{value['trials']}"+('†' if not value['phase_verified'] else '')


def inventory():
 result=[]
 for base in [WORK/'evaluations',WORK/'results']:
  for path in sorted(base.rglob('trial_results.jsonl')):
   if any(x in path.parts for x in ['reports','videos']):continue
   rows=records(path);counts=Counter(r['status'] for r in rows)
   result.append({'path':str(path),'sha256':sha(path),'records':len(rows),'successes':counts[0],'manipulation_failures':counts[1],
                  'infrastructure_or_no_action':sum(v for k,v in counts.items() if k not in [0,1]),
                  'phase_verified':(path.parent/'verification.json').exists(),
                  'scored_or_pilot':'pilot' if 'pilot' in path.parts else 'full'})
 return result


def support(scenarios):
 family=WORK/'training/fresh20k-20261025/family';ids=set(read(family/'split.json')['splits']['train'])
 eps={e['episode_index']:e for e in read(family/'episodes.json')};hist=Counter();asset_counts=Counter();asset_rigid=Counter()
 for line in (WORK/'cache/huggingface/lerobot/hzxie/DOM/meta/camera.jsonl').open():
  row=json.loads(line);i=row['episode_index']
  if i not in ids:continue
  match=re.search(r'_franka_(.+)[ds]_O(\d+)_',row['filename']);assert match
  task=eps[i]['task'];asset=match[1];n=int(match[2]);hist[(task,n)]+=1;asset_counts[(task,asset)]+=1;asset_rigid[(task,asset,n)]+=1
 for s in scenarios:
  s['training_task_rigid_count_episodes']=hist[(s['task'],s['rigid_count'])]
  s['training_task_exact_asset_episodes']=asset_counts[(s['task'],s['target_asset'])]
  s['training_task_exact_asset_rigid_count_episodes']=asset_rigid[(s['task'],s['target_asset'],s['rigid_count'])]
 return [{'task':t,'rigid_count':n,'episodes':c} for (t,n),c in sorted(hist.items())]


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 anchor=read(ANCHOR/'protocol.json');scenarios=[describe(t) for t in anchor['tests']]
 runs=[load_run(*spec) for spec in SPECS];by_key={r['key']:r for r in runs}
 focus=[s for s in scenarios if s['task']=='pick' and (c:=cell(by_key['krv_best'],s)) and c['successes']>0]
 assert len(focus)==2 and sum(cell(by_key['krv_best'],s)['successes'] for s in focus)==15
 assert sum(cell(by_key['krv_best'],s)['trials'] for s in scenarios if s['task']=='pick')==160
 histogram=support(scenarios);inv=inventory()
 primary=runs[:6]
 for run in primary:
  checked=[cell(run,s) for s in scenarios]
  assert all(c and c['phase_verified'] and c['trials']==20 for c in checked)
  summary=read(Path(run['root'])/'summary.json')
  assert summary['complete'] and summary['verified_trials']==360
  assert abs(sum(c['successes'] for c in checked)/360*100-summary['observed']['Success'])<1e-8
 shared_nonzero=[s['name'] for s in scenarios if all((c:=cell(r,s)) and c['successes']>0 for r in primary)]
 audit={'created_utc':datetime.now(timezone.utc).isoformat(),'audit_script_sha256':sha(Path(__file__)),
  'selection_basis':'Post-hoc audit of existing results; not a newly validated benchmark. No evaluation started.',
  'focus_definition':'All pick configurations with at least one success for the original fresh KRV100 best checkpoint; retain all 160 pick trials as denominator.',
  'focus_scenarios':focus,'all_anchor_scenarios':scenarios,'runs':runs,'trial_file_inventory':inv,'training_task_rigid_histogram':histogram,
  'scenarios_with_nonzero_success_in_all_six_completed_comparison_runs':shared_nonzero,
  'limitations':['Pilots are separate from scored trials; infrastructure failures are never counted as manipulation failures.',
   'Partial phases are marked and do not have the same denominator/completeness as verified 20-trial cells.',
   'Scenario matching requires both the same name and identical canonical JSON. Released place counterparts are not custom picking trials.',
   'Actual prompt variants and model random seeds were not controlled across historical methods. Trial IDs are not paired samples.',
   'Historical released/full-data models are shown for context and are not eligible 20k-only comparison rows.',
   'Repeated fixed scenarios are not independent scene samples. Sparse successes do not establish statistical equivalence or rank models reliably.',
   'A final subset chosen using these scores is outcome-selected; treat this as development/diagnostic data and collect new preregistered evaluation evidence.']}
 write(OUT/'audit.json',audit)
 fields=['method','group','scenario','dimension','task','mode','successes','trials','percent','phase_verified']
 with (OUT/'all_scenario_results.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for r in runs:
   for s in scenarios:
    for mode in ['full','pilot']:
     c=cell(r,s,mode)
     if c:writer.writerow({'method':r['label'],'group':r['group'],'scenario':s['name'],'dimension':s['dimension'],'task':s['task'],**{k:c[k] for k in ['mode','successes','trials','percent','phase_verified']}})
 lines=['# Cross-VLA audit of the earlier successful picking scenarios','',
  '**Evaluation remains paused.** This report reads existing records only. All 15/160 fresh KRV100-best pick successes came from two configurations; the other six pick configurations contributed 0/120. These are 15 successful repeats, not 15 different tasks.','',
  '## What the two environments contain','',
  '| Scenario | Target | Container | Bodies | Initial speed | Horizon | House USD |',
  '|---|---|---|---:|---:|---:|---|']
 for s in focus:lines.append(f"| {s['dimension']} pick | {s['target_asset']} | {', '.join(s['containers'])} | {s['rigid_count']} | {s['initial_speed_m_s']:.3f} m/s | {s['horizon_s']} s | {s['house_usd']} |")
 for s in focus:
  c=cell(by_key['krv_best'],s)
  lines+=['',f"**{s['dimension']}**: `{s['name']}`; seed {s['seed']}; successes in repeats {', '.join(map(str,c['success_trial_ids']))}. Object descriptions: "+'; '.join(s['instruction_metadata']['objects'])+'.',
   f"Angular damping: {s['angular_damping']:.6f}. Initial robot-relative position (m): "+', '.join(f'{x:.3f}' for x in s['target_position_robot_m'])+'. Robot-relative velocity (m/s): '+', '.join(f'{x:.3f}' for x in s['target_velocity_robot_m_s'])+'.',
   f"Exact target asset occurs in {s['training_task_exact_asset_episodes']} training episodes for this task, but this task/asset/{s['rigid_count']}-body combination occurs in {s['training_task_exact_asset_rigid_count_episodes']}. The entire pick/{s['rigid_count']}-body stratum has {s['training_task_rigid_count_episodes']} training episodes."]
 lines+=['','CR uses can13 + plate12; MG uses fcan04 + box14. Both have one manipulable object and one container, with no distractor bodies. Both custom pick variants retain the upstream pick completion goal. House IDs here identify the evaluation houses; training-house membership is unverified.','',
  'The 20k pick demonstrations have 3–5 total rigid bodies; these successful 2-body pick setups are **not matched to that count distribution**. can13 occurs in 109 training pick episodes. fcan04 occurs in 113 training place episodes but zero training pick episodes, adding a task/asset support mismatch for the MG pick case. MG speed is inside the nominal collection speed range, and its angular damping (1.5774) exceeds the checked-in collection friction/angular-damping prior upper bound (1.5). Do not call MG a velocity-only OOD test or verified IID case. Actual training damping values were not retained.','',
  '## Same exact pick configurations across completed eligible runs','',
  '| Method | CR drink-can pick | MG food-can pick | Combined |','|---|---:|---:|---:|']
 for r in primary:
  cs=[cell(r,s) for s in focus];lines.append('| '+r['label']+' | '+' | '.join(display(c) for c in cs)+f" | {sum(c['successes'] for c in cs)}/{sum(c['trials'] for c in cs)} |")
 lines+=['','**The pattern is not shared by every VLA.** DOM/KRV variants succeed on both can-picking cases; both SmolVLA variants have 0/20 on each.','',
  '## Coverage gaps and historical context','',
  '| Method | Status | CR pick | MG pick | CR pilot | MG pilot |','|---|---|---:|---:|---:|---:|']
 for r in runs[6:]:lines.append('| '+r['label']+' | '+r['group']+' | '+' | '.join(display(cell(r,s,m)) for m in ['full','pilot'] for s in focus)+' |')
 lines+=['','† = interrupted/unverified full phase; counts shown descriptively. A dash means not evaluated in that mode, not zero success. Pilots have one repeat per selected scenario and are not pooled into full results. π0.5 has only a 0/1 CR-pick pilot; its two pilot successes were DA avocado placement and MP lemon placement, not these can picks. VLASH+KR has 0/7 partial CR-pick trials and a 0/1 pilot, with no MG-pick data. VLASH baseline has no usable records; π0.5+KR has only failed startup/no-action records.','',
  'Released DOM was tested on the original **place** counterparts, not the custom pick variants: 12/20 on CR can placement and 20/20 on MG food-can placement. This is historical full-data context, not a matched pick comparison. DOM+velocity and older KRV checkpoints exceed or lack the current training-history eligibility and remain historical only.','',
  '## Exact placement counterparts','',
  '| Method | CR can13 place | MG fcan04 place |','|---|---:|---:|']
 counterparts=[next(s for s in scenarios if s['dimension']==d and s['task']=='place') for d in ['CR','MG']]
 for r in [by_key['released'],*primary]:lines.append('| '+r['label']+' | '+' | '.join(display(cell(r,s)) for s in counterparts)+' |')
 lines+=['','These place configurations are canonically identical across the rows, including the released evaluator. Released DOM performs better on these two than the fine-tuned DOM snapshots; that warrants checking checkpoint adaptation and inference compatibility. It does not identify the cause, and released/full-data pretraining is not a matched training-budget comparison.','',
  '## All 18 common configurations: completed scored trials','',
  '| Scenario | KRV best | KRV latest | DOM best | DOM latest | SmolVLA | SmolVLA KR |','|---|---:|---:|---:|---:|---:|---:|']
 for s in scenarios:lines.append('| '+s['dimension']+' '+s['task']+' · '+s['target_asset']+' | '+' | '.join(display(cell(r,s)) for r in primary)+' |')
 lines+=['','Every cell above has 20 scored repeats. There is **no scenario with nonzero success in all six completed comparison runs**. The strongest cross-family contrast is MP lemon placement: KRV best 1/20, KRV latest 3/20, DOM best/latest 3/20 each, SmolVLA 7/20 and SmolVLA KR 0/20. SmolVLA KR has sparse successes on VU beer pick, SR orange pick and VG peach pick; VG uses an intentionally unseen peach asset and conflicts with the requested exclusion of visual OOD.','',
  '## What this supports for a new subset','',
  '1. Use the two can scenarios and their place counterparts as **diagnostic anchors**, alongside MP lemon placement to cover a different cross-family success pattern. Keep their existing settings and labels for that diagnostic; do not silently relabel MG or two-body pick as IID.',
  '2. Define the final subset by task/asset/count/nominal-dynamics coverage. Include successes and failures, distractor levels and a reserved long-horizon stratum. The two can-pick cases alone favor the observed DOM/KRV pattern and do not represent the 20k pick context.',
  '3. Complete genuinely comparable diagnostic coverage for VLASH+KR and π0.5 only after a new user instruction. Their current coverage is too sparse to decide whether every architecture shares the same easy/hard cases.',
  '4. Treat these historical outcomes as development evidence. Freeze selection rules before collecting final evaluation data on new configurations/seeds; do not reuse outcome-selected historical scores as independent final evidence. Keep intended velocity shifts separate from nominal cases.','',
  '## Accounting and scope','']
 lines += ['- '+x for x in audit['limitations']]
 lines+=['','The full raw inventory, exact configuration fingerprints, checkpoint metadata, per-scenario counts and training support are in `audit.json`; tidy results are in `all_scenario_results.csv`. No simulations, training jobs, uploads or checkpoint changes were performed.','']
 (OUT/'comparison.md').write_text('\n'.join(lines))
 # A compact self-contained comparison page with exact counts on hover.
 h=['<!doctype html><meta charset="utf-8"><title>Cross-VLA scenario audit</title><style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:8px;text-align:center}th:first-child,td:first-child{text-align:left}small{color:#555}</style>',
  '<h1>Historical scenario audit</h1><p>Existing scored trials only. Same configuration hashes; historical prompts were not paired. Evaluation remains paused.</p><table><tr><th>Scenario</th>']
 h += ['<th>'+html.escape(r['label'])+'</th>' for r in primary];h+=['</tr>']
 for s in scenarios:
  h+=['<tr><td>'+html.escape(s['dimension']+' '+s['task']+' · '+s['target_asset'])+'</td>']
  for r in primary:
   c=cell(r,s);a=c['percent']/100 if c else 0;bg=f'rgb({int(250-110*a)},{int(250-35*a)},{int(250-100*a)})'
   h.append(f'<td style="background:{bg}" title="{html.escape(s["name"])}; {display(c)}">{display(c)}</td>')
  h+=['</tr>']
 h+=['</table><p><small>20 repeats per cell, sharing fixed scenario resets. Counts are descriptive, not independent scene samples or statistical equivalence tests.</small></p><p>See comparison.md for scene descriptions, partial coverage, historical exclusions, and subset guidance.</p>']
 (OUT/'comparison.html').write_text(''.join(h))
 # Link existing videos directly without copying large media.
 video_lines=['# Existing videos on the same scenarios','','Examples are selected by scenario and PASS/FAIL label for diagnosis; they are not additional scored trials.','']
 examples=[]
 for r in primary:
  for s in focus+[next(x for x in scenarios if x['dimension']=='MP' and x['task']=='place')]:
   matches=sorted((Path(r['root'])/'full').rglob(s['name']+'-*.mp4'))
   for outcome in ['SUCCESS','FAIL']:
    candidates=[p for p in matches if p.stem.endswith('-'+outcome)]
    if candidates:
     p=candidates[0];examples.append({'method':r['label'],'scenario':s['name'],'outcome':outcome,'path':str(p)})
     video_lines.append(f'- {r["label"]}, {s["dimension"]} {s["task"]}, {outcome}: [video]({p})')
 write(OUT/'video_examples.json',examples);(OUT/'videos.md').write_text('\n'.join(video_lines)+'\n')
 print(json.dumps({'report':str(OUT/'comparison.md'),'runs':len(runs),'trial_files_in_inventory':len(inv),'focus_scenarios':len(focus),'focus_successes':15,'shared_nonzero_scenarios_all_six':len(shared_nonzero),'video_links':len(examples)},indent=2))


if __name__=='__main__':main()
