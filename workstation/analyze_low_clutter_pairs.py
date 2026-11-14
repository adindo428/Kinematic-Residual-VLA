#!/usr/bin/env python3
"""Analyze baseline/KR outcome transitions only after both full suites finish."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

PAIRS=[('dom100_best','krv100_best'),('dom100_latest','krv100_latest'),('smolvla_baseline','smolvla_kr')]


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def load(root):
 result={};prompts={};metrics={};videos={}
 for folder in sorted((root/'full').iterdir()):
  if not folder.is_dir():continue
  v=read(folder/'verification.json');assert v['matched_prompts_verified'] and v['trajectory_metrics_verified']
  for row in rows(folder/'trial_results.jsonl'):
   key=(row['env_name'],row['trial']);assert key not in result;result[key]=row
  for row in rows(folder/'prompt_trace.jsonl'):prompts[(row['env_name'],row['trial'])]=row
  for row in rows(folder/'trajectory_metrics.jsonl'):metrics[(row['env_name'],row['trial'])]=row
  names={r['env_name'] for r in rows(folder/'trial_results.jsonl')}
  all_videos=sorted((folder/'videos').rglob('*.mp4'))
  for name in names:
   files=[p for p in all_videos if p.name.startswith(name+'-')]
   assert len(files)==20,(name,'Missing video records')
   videos.update({(name,i+1):str(path) for i,path in enumerate(files)})
 assert len(result)==len(prompts)==len(metrics)==1800
 return result,prompts,metrics,videos


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
 out=root/'paired_analysis';out.mkdir(exist_ok=True);states=[]
 for baseline,kr in PAIRS:
  br=root/'results'/baseline;rr=root/'results'/kr
  if not (br/'summary.json').exists() or not (rr/'summary.json').exists():
   states.append({'baseline':baseline,'kr':kr,'state':'waiting_for_both_complete_1800_trial_suites'});continue
  assert read(br/'summary.json')['verified_trials']==read(rr/'summary.json')['verified_trials']==1800
  b,bp,bm,bv=load(br);k,kp,km,kv=load(rr);assert b.keys()==k.keys()
  all_rows=[];joint=[];counts=Counter()
  for key in sorted(b):
   assert bp[key]['text']==kp[key]['text'] and bp[key]['seed']==kp[key]['seed']
   assert b[key]['status'] in [0,1] and k[key]['status'] in [0,1]
   category=('both_success' if b[key]['success'] and k[key]['success'] else
    'kr_success_baseline_failure' if k[key]['success'] else
    'baseline_success_kr_failure' if b[key]['success'] else 'both_failure')
   counts[category]+=1
   row={'scenario':key[0],'repeat':key[1],'prompt':bp[key]['text'],'outcome':category,
    'baseline_success':b[key]['success'],'kr_success':k[key]['success'],
    'baseline_time_s':b[key]['task_time_s'],'kr_time_s':k[key]['task_time_s'],
    'time_saved_s':b[key]['task_time_s']-k[key]['task_time_s'],
    'baseline_ee_path_m':bm[key]['ee_path_length_m'],'kr_ee_path_m':km[key]['ee_path_length_m'],
    'ee_path_reduction_m':bm[key]['ee_path_length_m']-km[key]['ee_path_length_m'],
    'baseline_video':bv[key],'kr_video':kv[key]}
   all_rows.append(row)
   if category=='both_success':joint.append(row)
  assert sum(counts.values())==1800
  directory=out/(baseline+'__vs__'+kr);directory.mkdir(exist_ok=True)
  for filename,values in [('all_paired_trials.csv',all_rows),('jointly_successful_trials.csv',joint),
                         ('kr_faster_joint_successes.csv',[r for r in joint if r['time_saved_s']>1e-6])]:
   with (directory/filename).open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(values)
  summary={'baseline':baseline,'kr':kr,'paired_trials':1800,'counts':dict(counts),
   'baseline_success_percent':100*sum(r['baseline_success'] for r in all_rows)/1800,
   'kr_success_percent':100*sum(r['kr_success'] for r in all_rows)/1800,
   'joint_success_trials':len(joint),'kr_faster_joint_successes':sum(r['time_saved_s']>1e-6 for r in joint),
   'baseline_faster_joint_successes':sum(r['time_saved_s']<-1e-6 for r in joint),
   'mean_time_saved_on_joint_success_s':statistics.mean(r['time_saved_s'] for r in joint) if joint else None,
   'mean_ee_path_reduction_on_joint_success_m':statistics.mean(r['ee_path_reduction_m'] for r in joint) if joint else None,
   'interpretation':'Joint-success timing/path results are outcome-conditioned secondary analyses; all-trial success is primary. Shorter paths are descriptive, not guaranteed higher quality.',
   'dom_pair_confound':'DOM/KRV differs in neural velocity conditioning and residualization.' if baseline.startswith('dom') else None}
  (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
  text=f'# {baseline} vs {kr}\n\n1,800 matched scenario/repeat/prompt/reset conditions. Model noise is not paired.\n\n'
  text+='| Outcome | Trials |\n|---|---:|\n'
  for category in ['both_success','kr_success_baseline_failure','baseline_success_kr_failure','both_failure']:
   text+=f'| {category} | {counts[category]} |\n'
  text+=f"\nBaseline success: {summary['baseline_success_percent']:.2f}%; KR success: {summary['kr_success_percent']:.2f}%.\n"
  text+=f"\nOf {len(joint)} jointly successful trials, KR finishes faster in {summary['kr_faster_joint_successes']}. See the CSVs for both improvement directions and all failures. These selected examples do not replace full-suite success rates.\n"
  if summary['dom_pair_confound']:text+='\n'+summary['dom_pair_confound']+'\n'
  text+='\n| Scenario / repeat | Time saved (s) | Baseline | KR |\n|---|---:|---|---|\n'
  for r in sorted([r for r in joint if r['time_saved_s']>1e-6],key=lambda r:-r['time_saved_s'])[:12]:
   text+=f"| {r['scenario']} / {r['repeat']} | {r['time_saved_s']:.2f} | [video]({r['baseline_video']}) | [video]({r['kr_video']}) |\n"
  (directory/'comparison.md').write_text(text)
  states.append({'baseline':baseline,'kr':kr,'state':'complete','paired_trials':1800,'joint_successes':len(joint)})
 (out/'status.json').write_text(json.dumps({'pairs':states},indent=2)+'\n')
 print(json.dumps({'paired_analysis':states}))


if __name__=='__main__':main()
