"""Verify complete trial accounting and compare manipulation metrics to Table 1."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import math
import statistics
from prepare import write_json

def verify_records(records,expected_names,trials_per_environment,horizons):
    expected=Counter({n:trials_per_environment for n in expected_names})
    assert Counter(r['env_name'] for r in records)==expected,'Missing, extra or duplicated environment trials'
    assert len({(r['env_name'],r['trial']) for r in records})==len(records),'Duplicate trial IDs'
    for name in expected_names:
        assert {r['trial'] for r in records if r['env_name']==name}==set(range(1,trials_per_environment+1))
    for r in records:
        assert r['status'] in (0,1),'No-action/infrastructure outcomes cannot become benchmark failures'
        assert r['success']==(r['status']==0)
        assert r['n_steps']>1 and r['step_dt_s']>0
        assert math.isfinite(r['task_time_s'])
        assert abs(r['task_time_s']-r['n_steps']*r['step_dt_s'])<1e-8
        assert 0<r['task_time_s']<=horizons[r['env_name']]+2*r['step_dt_s']
    return {'verified':True,'trials':len(records),'environments':len(expected),'successes':sum(r['success'] for r in records),
            'success_percent':100*statistics.mean(r['success'] for r in records),
            'mean_task_time_s':statistics.mean(r['task_time_s'] for r in records),'infrastructure_failures':0}

def read_records(folder):
    return [json.loads(line) for line in (Path(folder)/'trial_results.jsonl').read_text().splitlines()]

def verify_folder(folder,root,env_list,n_trials):
    protocol=json.loads((Path(root)/'protocol.json').read_text())
    tests={t['name']:t for t in protocol['tests']}
    names=[Path(line).stem for line in Path(env_list).read_text().splitlines() if line.strip()]
    result=verify_records(read_records(folder),names,n_trials,{n:tests[n]['horizon_seconds'] for n in names})
    log=(Path(folder)/'client.log').read_text()
    assert f'Evaluation completed. Total tests run: {len(names)*n_trials}' in log,'Client did not acknowledge every trial'
    assert 'Test suite done with success rates:' in log
    write_json(Path(folder)/'verification.json',result)
    return result

def summarize(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text());gpu_label=p.get('gpu_label', 'L40S');records=[]
    for i,d in enumerate(p['dimension_order']):
        folder=root/'full'/f'{i:02d}_{d}'
        verify_folder(folder,root,root/'test_lists'/f'{i:02d}_{d}.txt',20)
        records.extend(read_records(folder))
    assert len(records)==1800
    test_dimension={t['name']:t['dimension'] for t in p['tests']}
    observed={};rows=[]
    for d in p['dimension_order']:
        group=[r for r in records if test_dimension[r['env_name']]==d];assert len(group)==200
        sr=100*statistics.mean(r['success'] for r in group)
        observed[d]=sr
        rows.append({'Metric':d,'Published':p['paper_reference'][d],'Observed':sr,
                     'Difference':sr-p['paper_reference'][d],'Trials':len(group),
                     'Observed_mean_time_s':statistics.mean(r['task_time_s'] for r in group)})
    observed['Success']=100*statistics.mean(r['success'] for r in records)
    observed['Time']=statistics.mean(r['task_time_s'] for r in records)
    for key in ('Success','Time'):
        rows.append({'Metric':key,'Published':p['paper_reference'][key],'Observed':observed[key],
                     'Difference':observed[key]-p['paper_reference'][key],'Trials':1800,
                     'Observed_mean_time_s':observed['Time']})
    write_json(root/'summary.json',{'complete':True,'verified_trials':1800,'protocol':p,'observed':observed,'comparison':rows})
    with (root/'table1_comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    md='Official DynamicVLA DOM manipulation benchmark: Table 1 comparison\n\n'
    md+=f'| Metric | Paper | {gpu_label} evaluation | Difference |\n|---|---:|---:|---:|\n'
    for r in rows:
        unit=' s' if r['Metric']=='Time' else '%'
        delta_unit=' s' if r['Metric']=='Time' else ' pp'
        md+=f"| {r['Metric']} | {r['Published']:.2f}{unit} | {r['Observed']:.2f}{unit} | {r['Difference']:+.2f}{delta_unit} |\n"
    md+='\n1800 verified trials, 20 per released configuration. Success is instructed manipulation completion. Time averages successful and failed trials. No tracking-distance criterion is applied.\n\n'
    md+='Official checkpoint revision: '+p['checkpoint_revision']+'; upstream code: '+p['official_source_commit']+'.\n\n'
    md+=f'Hardware differs from the paper RTX A6000. The official simulator and continuous inference/latency-aware action streaming are retained. Released configurations and reset seeds are unchanged. Numerical deviations are reported without claiming statistical equivalence or independent randomized repeats.\n'
    (root/'table1_comparison.md').write_text(md)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=[*p['dimension_order'],'Success','Time']
    fig,ax=plt.subplots(figsize=(14,3));ax.axis('off')
    cells=[['Published DynamicVLA',*[f"{p['paper_reference'][k]:.2f}" for k in keys]],
           [f'Released checkpoint · {gpu_label}',*[f"{observed[k]:.2f}" for k in keys]]]
    table=ax.table(cellText=cells,colLabels=['Method',*p['dimension_order'],'Success\n(%) ↑','Time\n(s) ↓'],
                   loc='center',cellLoc='center',colWidths=[.27]+[.062]*9+[.08,.08])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.8)
    ax.set_title('Official DOM manipulation benchmark: Table 1 reproduction',pad=22)
    fig.text(.06,.03,f'1,800 manipulation trials. Original test configurations and streaming execution; {gpu_label} hardware differs from paper RTX A6000.',fontsize=9)
    fig.savefig(root/'table1_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print(json.dumps(observed,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--folder',type=Path);p.add_argument('--env-list',type=Path);p.add_argument('--trials',type=int)
    a=p.parse_args()
    if a.folder:print(verify_folder(a.folder,a.root,a.env_list,a.trials))
    else:summarize(a.root)
