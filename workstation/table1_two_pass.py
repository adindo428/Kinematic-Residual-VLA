"""Verify two disjoint repeat blocks and publish provisional/final results."""
import json
from pathlib import Path
import statistics
from report import verify_records, verify_folder, read_records


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)

def plan(root):return read(Path(root)/'two_pass_plan.json')
def block(root,index):
    counts=plan(root)['repeats_per_pass'];return counts[index-1],sum(counts[:index-1])

def verify(folder,root,listing,repeats,offset,low_clutter):
    if (Path(folder)/'composition.json').exists():
        assert not low_clutter and offset in (0,10)
        if read(Path(folder)/'composition.json').get('kind')=='resume_prefix':
            from table1_resume_records import verify as verify_composed
        else:
            from table1_trial_reuse import verify_composed
        result=verify_composed(folder,root,listing,repeats)
    else:
        result=verify_folder(folder,root,listing,repeats)
    if low_clutter:
        configs={t['name']:read(t['path']) for t in read(Path(root)/'protocol.json')['tests']}
        expected={(Path(n).stem,i) for n in Path(listing).read_text().splitlines() if n for i in range(1,repeats+1)}
        for file in ['prompt_trace.jsonl','trajectory_metrics.jsonl']:
            rows=[json.loads(l) for l in (Path(folder)/file).read_text().splitlines()]
            assert len(rows)==len(expected) and {(r['env_name'],r['trial']) for r in rows}==expected
            for row in rows:
                if file=='prompt_trace.jsonl':
                    cfg=configs[row['env_name']]
                    assert row['text']==cfg['evaluation_prompts'][offset+row['trial']-1] and row['seed']==cfg['seed']
                else:assert Path(row['trajectory_path']).exists() and row['ee_path_length_m']>=0
        result.update(matched_prompts_verified=True,trajectory_metrics_verified=True)
    result.update(repeat_offset=offset,repeats=repeats,pass_block=True)
    write(Path(folder)/'verification.json',result);return result

def merge(root,dimension,index,listing,low_clutter):
    root=Path(root);name=f'{index:02d}_{dimension}';records=[];extras={f:[] for f in ['prompt_trace.jsonl','trajectory_metrics.jsonl']};sources=[]
    for pass_index in [1,2]:
        count,offset=block(root,pass_index);folder=root/'passes'/f'{pass_index:02d}'/'full'/name
        verify(folder,root,listing,count,offset,low_clutter);sources.append(str(folder))
        for row in read_records(folder):records.append({**row,'trial':row['trial']+offset,'pass_index':pass_index,'local_trial':row['trial']})
        if low_clutter:
            for file in extras:
                extras[file].extend({**row,'trial':row['trial']+offset} for row in [json.loads(l) for l in (folder/file).read_text().splitlines()])
    tests={t['name']:t for t in read(root/'protocol.json')['tests']};names=[Path(n).stem for n in Path(listing).read_text().splitlines() if n]
    result=verify_records(records,names,20,{n:tests[n]['horizon_seconds'] for n in names})
    result.update(aggregation='two verified pass blocks; client acknowledgements verified separately',source_pass_folders=sources)
    if low_clutter:result.update(matched_prompts_verified=True,trajectory_metrics_verified=True)
    target=root/'full'/name;target.mkdir(parents=True,exist_ok=True)
    (target/'trial_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    if low_clutter:
        for file,rows in extras.items():(target/file).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(target/'verification.json',result)
    return result

def summarize(root,dimensions,checkpoint):
    root=Path(root);records=[];observed={}
    for i,d in enumerate(dimensions):
        v=read(root/'full'/f'{i:02d}_{d}'/'verification.json');assert v['verified'] and v['trials']==200
        observed[d]=v['success_percent'];records+=read_records(root/'full'/f'{i:02d}_{d}')
    assert len(records)==1800
    observed.update(Success=100*statistics.mean(r['success'] for r in records),Time=statistics.mean(r['task_time_s'] for r in records))
    write(root/'summary.json',{'complete':True,'verified_trials':1800,'observed':observed,'checkpoint':checkpoint,'two_pass_plan':plan(root)})
