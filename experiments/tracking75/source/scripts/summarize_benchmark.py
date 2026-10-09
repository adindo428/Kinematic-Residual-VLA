#!/usr/bin/env python3
"""Publish a Table 1-style tracking table only after all 7200 trials verify."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from tracking_online_common import write_json
from tracking_benchmark import DIMENSIONS, MODELS, SCORING_VERSION, SUCCESS_DEFINITION, TIME_DEFINITION, mean_episode_success_percent


def summarize(root):
    root=Path(root); p=json.loads((root/'protocol.json').read_text())
    assert p['scoring_version']==SCORING_VERSION
    all_records=[]
    for shard in range(90):
        folder=root/'full'/f'{shard:03d}'
        v=json.loads((folder/'verification.json').read_text())
        if v.get('scoring_version')!=SCORING_VERSION:
            from verify_benchmark import verify
            v=verify(folder,folder/'manifest.json')
        assert v['verified'] and v['pairing_valid'] and v['rollouts']==80
        assert v['scoring_version']==SCORING_VERSION
        all_records+=json.loads((folder/'rollouts.json').read_text())
    assert len(all_records)==p['expected_rollouts']
    assert all(r['metrics']['scoring_version']==SCORING_VERSION for r in all_records)
    assert len({(r['model'],r['dimension'],r['environment'],r['episode']) for r in all_records})==7200
    table=[]; details={}
    for model in MODELS:
        rows=[r for r in all_records if r['model']==model]
        assert len(rows)==1800
        out={'Method':p['model_labels'][model]}
        details[model]={}
        for d in DIMENSIONS:
            group=[r for r in rows if r['dimension']==d]; assert len(group)==200
            out[d]=mean_episode_success_percent([r['metrics'] for r in group])
            details[model][d]={
                'mean_episode_success_percent':out[d],'trials':200,
                'completed':sum(r['metrics']['completed'] for r in group),
                'mean_object_speed_mps':float(np.mean([r['object_mean_speed_mps'] for r in group])),
                'mean_whole_episode_within_3cm_fraction':float(np.mean([r['metrics']['whole_episode_within_3cm_fraction'] for r in group])),
                'mean_post_acquisition_within_3cm_fraction':float(np.mean([r['metrics']['post_acquisition_within_3cm_fraction'] or 0 for r in group])),
                'completed_mean_hover_rmse_m':float(np.mean([r['metrics']['hover_rmse_m'] for r in group if r['metrics']['completed']])) if any(r['metrics']['completed'] for r in group) else None,
                'acquisition_time_capped_s':float(np.mean([r['metrics']['acquisition_time_capped_s'] for r in group]))}
        out['Success']=mean_episode_success_percent([r['metrics'] for r in rows])
        out['Time']=float(np.mean([r['metrics']['acquisition_time_capped_s'] for r in rows]))
        out['Completion']=100*sum(r['metrics']['completed'] for r in rows)/len(rows)
        table.append(out)
    write_json(root/'summary.json',{'complete':True,'pairing_valid':True,'expected_rollouts':7200,
        'recorded_rollouts':len(all_records),'protocol':p,'table':table,'dimensions':details})
    columns=['Method',*DIMENSIONS,'Success','Time','Completion']
    with (root/'tracking_table1.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(table)
    best={k:max(r[k] for r in table) for k in [*DIMENSIONS,'Success']}
    best['Time']=min(r['Time'] for r in table)
    md='Table 1-style tracking adaptation (not the DOM manipulation benchmark).\n\n'
    md+='| Method | '+' | '.join([*DIMENSIONS,'Success ↑','Acquisition time (s) ↓'])+' |\n'
    md+='|---|'+'---:|'*11+'\n'
    for r in table:
        values=[]
        for k in [*DIMENSIONS,'Success','Time']:
            val=f'{r[k]:.2f}'
            values.append(f'**{val}**' if r[k]==best[k] else val)
        md+='| '+r['Method']+' | '+' | '.join(values)+' |\n'
    md+='\nSuccess: '+SUCCESS_DEFINITION+'\n\n'
    md+='Time: '+TIME_DEFINITION+' Completion rate is reported separately in the CSV and JSON.\n\n'
    md+=p['scope']+'\n\n'
    for d,description in DIMENSIONS.items(): md+=f'- {d}: {description}\n'
    (root/'tracking_table1.md').write_text(md)
    # A shareable scientific figure, with categories grouped like the paper.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(15,4.8));ax.axis('off')
    cells=[[r['Method'],*[f'{r[k]:.2f}' for k in [*DIMENSIONS,'Success','Time']]] for r in table]
    t=ax.table(cellText=cells,colLabels=['Method',*DIMENSIONS,'Success\n(%) ↑','Acquisition\n(s) ↓'],loc='center',cellLoc='center',colWidths=[.27]+[.065]*9+[.075,.075])
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,1.9)
    for (i,j),cell in t.get_celld().items():
        cell.set_edgecolor('#dddddd')
        if i==0:cell.set_facecolor('#dde5ef');cell.get_text().set_weight('bold')
        if i>0 and j>0:
            key=[*DIMENSIONS,'Success','Time'][j-1]
            if table[i-1][key]==best[key]:cell.get_text().set_weight('bold')
    ax.set_title('Closed-loop hover tracking: Table 1-style adaptation',fontsize=15,pad=35)
    fig.text(.16,.80,'Interaction: CR / DA / LS      Perception: VU / SR / MP      Generalization: VG / MG / DR',fontsize=10)
    fig.text(.04,.06,'Success: mean per-episode time within 3 cm (%), including the initial second; no pass/fail gate.\nTime: first entry within 3 cm (s), capped at horizon if never acquired. Terminated remainder earns zero tracking credit.\nOne training room; ten seeded setups; VG uses unseen apple meshes. LS horizon 30 s; others 10 s.',fontsize=9)
    fig.savefig(root/'tracking_table1.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('Completed 7200-trial table:',root/'tracking_table1.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();summarize(a.root)
