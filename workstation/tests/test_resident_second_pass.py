"""Keep pass-two repeat IDs disjoint and final averages balanced."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_low_clutter_table1
import table1_two_pass as passes
from table1_resume_records import compose
import table1_display_state as display


def write_trials(folder,names):
    folder.mkdir(parents=True)
    rows=[{'env_name':name,'trial':i,'status':int(i%2==0),'success':bool(i%2),
           'n_steps':10,'step_dt_s':.04,'task_time_s':.4} for name in names for i in range(1,11)]
    (folder/'trial_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (folder/'client.log').write_text(f'Evaluation completed. Total tests run: {len(rows)}\nTest suite done with success rates:\n')


def test_second_pass_composition_merges_global_ids_11_through_20(tmp_path):
    root=tmp_path/'root';root.mkdir();names=['a','b']
    (root/'protocol.json').write_text(json.dumps({'tests':[{'name':n,'horizon_seconds':12} for n in names]}))
    (root/'two_pass_plan.json').write_text(json.dumps({'repeats_per_pass':[10,10]}))
    listing=root/'cases.txt';listing.write_text('a.json\nb.json\n')
    first=root/'passes/01/full/00_CR';write_trials(first,names)
    original=(first/'trial_results.jsonl').read_bytes()
    sources=[]
    for name in names:
        folder=tmp_path/('second_'+name);write_trials(folder,[name])
        case=folder/'case.txt';case.write_text(name+'.json\n')
        sources.append({'folder':str(folder),'listing':str(case),'repeats':10,'repeat_offset':0})
    second=root/'passes/02/full/00_CR';compose(second,root,listing,10,sources)
    assert passes.verify(second,root,listing,10,10,False)['repeat_offset']==10
    merged=passes.merge(root,'CR',0,listing,False)
    assert merged['trials']==40 and merged['successes']==20
    rows=passes.read_records(root/'full/00_CR')
    for name in names:
        assert {r['trial'] for r in rows if r['env_name']==name and r['pass_index']==2}==set(range(11,21))
    assert (first/'trial_results.jsonl').read_bytes()==original


def test_mixed_100_and_200_trial_columns_do_not_produce_an_overall(tmp_path):
    for i,d in enumerate(display.DIMS):
        for p in [1,2] if i==0 else [1]:
            path=tmp_path/'passes'/f'{p:02d}'/'full'/f'{i:02d}_{d}'/'verification.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'verified':True,'trials':100,'successes':50,'mean_task_time_s':1}))
    values=display.provisional(tmp_path)
    assert values['CR']=='50.00'
    assert 'provisional' in values['DA']
    assert 'Success' not in values and 'Time' not in values


def test_complete_200_trial_columns_publish_final_overall(tmp_path):
    for i,d in enumerate(display.DIMS):
        for p in [1,2]:
            path=tmp_path/'passes'/f'{p:02d}'/'full'/f'{i:02d}_{d}'/'verification.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'verified':True,'trials':100,'successes':50,'mean_task_time_s':1}))
    values=display.provisional(tmp_path)
    assert values['Success']=='50.00' and values['Time']=='1.00'
