"""User-selected MG-only KRV follow-up, then finish all original DOM trials."""
import argparse,collections,copy,fcntl,json,signal,sys
from pathlib import Path
import run_krv_full_dom_ablation as full
base=full.base
ROOT=full.ROOT
DIMS=['CR','DA','LS','VU','SR','MP','VG','MG','DR']

def read(p):return json.loads(p.read_text())
def rows(j):
 p=Path(j['folder'])/'trial_results.jsonl'
 return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
def render():
 plan=read(ROOT/'plan.json');result={m['key']:{d:{'successes':0,'trials':0,'target':100} for d in DIMS} for m in plan['models']}
 for j in plan['jobs']:
  s=result[j['model']][j['dimension']];data=rows(j);s['successes']+=sum(bool(x['success']) for x in data);s['trials']+=len(data)
 report={'models':result,'per_category_target':100,'DOM_total_target':900,'partial_scores_are_not_full_suite_scores':True}
 base.low.write(ROOT/'category_scores.json',report)
 lines=['# Full DOM evaluation: DOM completion','','DOM retains its 900-trial target (100 per category). KRV, VPD and VI evaluation arms are stopped; their completed results are retained as partial. Historical KRV 900-trial results are reported separately.','','| Method | CR | DA | LS | VU | SR | MP | VG | MG | DR | All completed |','|---|'+'---:|'*10]
 scores={}
 for m in plan['models']:
  dims=result[m['key']];success=sum(v['successes'] for v in dims.values());n=sum(v['trials'] for v in dims.values());scores[m['key']]=[success,n]
  if m['key'] in ('full_vpd','full_vpd_vi'):continue
  cells=[str(dims[d]['successes'])+'/'+str(dims[d]['trials']) for d in DIMS]
  lines.append('| '+m['key']+' | '+' | '.join(cells)+f' | {success}/{n} |')
 lines+=['','Cells are successes/completed trials; each category target is 100. Percent equals the success count only when n=100.','', 'VPD/VI remain inference-only on the existing full-data epoch-3 KRV checkpoint. Stopped arms must not be presented as completed 900-trial experiments.','']
 (ROOT/'live_results.md').write_text('\n'.join(lines))
 from append_krv_history import append_history
 append_history(ROOT/'live_results.md')
 return scores

def verify(plan):
 for j in plan['jobs']:
  data=rows(j);assert len(data)==j['repeats'],j['id'];assert read(Path(j['folder'])/'resident_client_completion.json')['complete']
 scores=render();dom=scores['dom'];assert dom[1]==900
 base.low.write(ROOT/'scope-completion.json',{'verified':True,'DOM_scores':dom,'MG_models':plan['scope']['MG_models'],'scores':scores,'KRV_900_not_claimed':True})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mg-models',nargs='*',default=[]);parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
 if args.report_only:render();return
 assert set(args.mg_models)<= {'full_kra','full_vpd','full_vpd_vi'}
 original=read(ROOT/'plan.json');plan=copy.deepcopy(original)
 mg=[j for j in original['jobs'] if j['dimension']=='MG' and j['model'] in args.mg_models]
 dom=[j for j in original['jobs'] if j['model']=='dom']
 plan['jobs']=mg+dom;plan['scope']={'MG_models':args.mg_models,'DOM_all_categories':True,'user_requested_stop_after_MG_for_KRV':bool(args.mg_models),'KRV_evaluation_cancelled':not args.mg_models}
 selected_models={j['model'] for j in plan['jobs']}
 plan['models']=[m for m in original['models'] if m['key'] in selected_models]
 plan['scope']['resident_models']=[m['key'] for m in plan['models']]
 # Reuse wording already recorded in this evaluation when restarting workers.
 for j in sorted(original['jobs'],key=lambda j:j['model']=='dom'):
  trace=Path(j['folder'])/'prompt_trace.jsonl'
  if trace.exists():
   cache=plan.setdefault('saved_prompts',{}).setdefault(j['scene'],{})
   for line in trace.read_text().splitlines():
    row=json.loads(line);cache[str(row.get('global_trial',row['trial']))]=row['text']
 plan['scope']['recorded_prompt_cache_reused']=True
 # The simulator reads its block metadata from the original report plan.
 original['saved_prompts']=plan['saved_prompts']
 base.low.write(ROOT/'plan.json',original)
 # Preserve all already scored records; no incomplete DOM block may be silently overwritten.
 for j in plan['jobs']:
  data=rows(j)
  if data and not (Path(j['folder'])/'resident_client_completion.json').exists():
   raise RuntimeError('Partial selected block requires explicit resumable tail: '+j['id'])
 base.low.write(ROOT/'mg-then-dom-plan.json',plan)
 base.render=render;base.verify=verify;base.prepare=lambda:plan;base.launch=full.launch
 sys.argv=[sys.argv[0]]
 base.main()

if __name__=='__main__':main()
