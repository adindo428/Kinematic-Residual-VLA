"""Build scene-first missing-trial jobs and retain existing checkpoint evidence."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
import re
from copy import deepcopy
from pathlib import Path

import run_low_clutter_table1 as low
import run_krv_full_two_suites as krv
import table1_two_pass as passes
from report import read_records,verify_records
from table1_resume_records import compose

ROOT=krv.ROOT/'resident_schedule'

def coverage(ledger,model,dimension):
    rows=[];seen=set()
    for source in ledger['sources'][model][dimension]:
        path=Path(source['folder'])/'trial_results.jsonl'
        if source.get('prefix'):assert low.sha(path)==source['records_sha256']
        for row in read_records(path.parent):
            row={**row,'trial':row['trial']+source.get('repeat_offset',0)}
            key=(row['env_name'],row['trial']);assert key not in seen,'Duplicate resident evidence'
            seen.add(key);rows.append(row)
    return rows

def check_rows(model,rows):
    tests={t['name']:t for t in low.read(Path(model['root'])/'protocol.json')['tests']}
    grouped=defaultdict(list)
    for r in rows:grouped[r['env_name']].append(r)
    for name,rs in grouped.items():
        assert len(rs)<=10
        verify_records(rs,[name],len(rs),{name:tests[name]['horizon_seconds']})

def prefix(folder,offset=0):
    folder=Path(folder);rows=read_records(folder)
    return {'folder':str(folder),'prefix':True,'counts':dict(Counter(r['env_name'] for r in rows)),
            'records_sha256':low.sha(folder/'trial_results.jsonl'),'repeat_offset':offset,
            'basis':'Durable completed records; no fabricated full-suite client acknowledgement'}

def saved_prompts(ledger):
    """Recover actual instructions for already-scored trials, never regenerate them."""
    result={}
    def visit(source):
        folder=Path(source['folder']);offset=source.get('repeat_offset',0)
        if (folder/'composition.json').exists():
            for child in low.read(folder/'composition.json')['sources']:visit(child)
            return
        rows=read_records(folder);pending=None;texts=[]
        trace=folder/'prompt_trace.jsonl'
        if trace.exists():texts=[json.loads(l)['text'] for l in trace.read_text().splitlines()]
        else:
            for line in (folder/'client.log').read_text().splitlines():
                match=re.search(r'Received new task: (.*)',line)
                if match:pending=match[1]
                if re.search(r'\[Test\d+\] Test done with ',line):
                    assert pending is not None
                    texts.append(pending);pending=None
        assert len(texts)>=len(rows),str(folder)
        for row,text in zip(rows,texts):
            candidates=[p for d in low.DIMS for p in (krv.BASE/'test_lists'/f'{low.DIMS.index(d):02d}_{d}.txt').read_text().splitlines() if Path(p).stem==row['env_name']]
            assert len(candidates)==1
            prompts=result.setdefault(candidates[0],{});trial=str(row['trial']+offset)
            assert trial not in prompts or prompts[trial]==text,'Previously scored prompts differ'
            prompts[trial]=text
    for model_sources in ledger['sources'].values():
        for sources in model_sources.values():
            for source in sources:visit(source)
    return result

def freeze():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'ledger.json').exists():return low.read(ROOT/'ledger.json')
    previous=low.read(krv.ROOT/'constant_resume/request.json')
    selected=low.read(krv.ROOT/'constant_resume/task_selection.json')
    models=previous['models'];sources={m['key']:{d:[] for d in low.DIMS} for m in models}
    for m in models:
        cp=Path(m['checkpoint']);assert low.sha(cp/'model.safetensors')==m['weight_sha256']
        assert low.sha(cp/'config.json')==m['config_sha256']
        cfg=low.read(cp/'config.json');assert cfg.get('residual_target_lead_s',0)==0
        assert cfg['residual_phase_gate_mode']==('causal_rules' if m['gated'] else 'constant')
        root=Path(m['root'])
        for i,d in enumerate(low.DIMS):
            folder=root/'passes/01/full'/f'{i:02d}_{d}'
            listing=root/'test_lists'/f'{i:02d}_{d}.txt'
            if (folder/'verification.json').exists():
                passes.verify(folder,root,listing,10,0,False)
                sources[m['key']][d].append(prefix(folder));continue
            if m['key']=='krv_best_constant' and d in previous['prefixes']:
                sources[m['key']][d].append(previous['prefixes'][d])
            for task in previous['tasks'][m['key']]:
                if task['dimension']!=d:continue
                key=m['key']+'/'+task['key']
                if key in selected:
                    folder=Path(selected[key]);low.runtime.verify_trial_folder(folder,root,Path(task['listing']),task['repeats'])
                    sources[m['key']][d].append({'folder':str(folder),'listing':task['listing'],'repeats':task['repeats'],
                                               'repeat_offset':task['repeat_offset']})
                else:
                    for attempt in (root/'resume_tasks'/task['key']).glob('attempt*'):
                        if not (attempt/'trial_results.jsonl').exists():continue
                        failure=low.read(attempt/'attempt_failure.json') if (attempt/'attempt_failure.json').exists() else {}
                        # Recover only the intentional user-requested transition,
                        # never runtime-disqualified parallel attempts.
                        if 'InterruptedError' not in failure.get('error',''):continue
                        runtime=low.read(attempt/'runtime.json')
                        assert runtime['checkpoint_sha256']==m['weight_sha256'] and not runtime['parallel']
                        if read_records(attempt):sources[m['key']][d].append(prefix(attempt,task['repeat_offset']))
    ledger={'models':models,'sources':sources,'accepted_jobs':[],
            'created_utc':datetime.now(timezone.utc).isoformat(),'budget':{'per_column':100,'per_checkpoint':900}}
    for m in models:
        for d in low.DIMS:check_rows(m,coverage(ledger,m['key'],d))
    low.write(ROOT/'ledger.json',ledger);return ledger

def make_plan(ledger,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    jobs=[];history={m['key']:[] for m in ledger['models']}
    by_dimension={}
    for m in ledger['models']:
        for d in low.DIMS:
            rows=coverage(ledger,m['key'],d);check_rows(m,rows)
            by_dimension[m['key'],d]=Counter(r['env_name'] for r in rows)
            history[m['key']].extend({r['env_name'] for r in rows if r['success']})
    for i,d in enumerate(low.DIMS):
        listing=krv.BASE/'test_lists'/f'{i:02d}_{d}.txt'
        for scene in listing.read_text().splitlines():
            name=Path(scene).stem
            for m in ledger['models']:
                offset=by_dimension[m['key'],d][name]
                if offset==10:continue
                uid=f'{i:02d}-{name}-{m["key"]}'
                folder=directory/'blocks'/uid;folder.mkdir(parents=True)
                (folder/'case.txt').write_text(scene+'\n')
                jobs.append({'id':uid,'model':m['key'],'scene':scene,'scene_sha256':low.sha(scene),'dimension':d,
                             'repeats':10-offset,'repeat_offset':offset,'folder':str(folder),'scored':True})
    result={'models':ledger['models'],'jobs':jobs,'success_history':history,'saved_prompts':saved_prompts(ledger),'all_trials_excluded_from_table1':False,
            'schedule':'scene first; each checkpoint receives an independent reset',
            'second_ten_repeats_deferred':True,'target_lead_s':0}
    low.write(directory/'plan.json',result);return result

def recover_interrupted(ledger):
    """Retain completed episodes from a stopped block without inventing a suite ACK."""
    for plan_path in sorted((ROOT/'runs').glob('run*/plan.json')):
        run=plan_path.parent
        state=low.read(run/'run_status.json') if (run/'run_status.json').exists() else {}
        if state.get('state') not in ('paused','failed'):continue
        for job in low.read(plan_path)['jobs']:
            if job['id'] in ledger['accepted_jobs']:continue
            folder=Path(job['folder']);records=folder/'trial_results.jsonl'
            if not records.exists():continue
            if (folder/'resident_client_completion.json').exists():
                accept(ledger,job);continue
            # No background writer survives launch()'s process-group cleanup.
            sources=ledger['sources'][job['model']][job['dimension']]
            if any(s['folder']==str(folder) for s in sources):continue
            rows=read_records(folder)
            if not rows:continue
            block=low.read(folder/'resident_block.json');assert block['job']==job
            worker=low.read(run/'execution/workers/loading.json')['receipts'][job['model']]
            model=next(m for m in ledger['models'] if m['key']==job['model'])
            assert worker['checkpoint']==model['checkpoint'] and len(rows)<=job['repeats']
            runtime=low.read(folder/'runtime.json')
            assert runtime['checkpoint_sha256']==model['weight_sha256'] and not runtime['parallel_inference']
            candidate=deepcopy(ledger)
            candidate['sources'][job['model']][job['dimension']].append(prefix(folder,job['repeat_offset']))
            check_rows(model,coverage(candidate,job['model'],job['dimension']))
            low.write(ROOT/'ledger.json',candidate);ledger.clear();ledger.update(candidate)

def accept(ledger,job):
    if job['id'] in ledger['accepted_jobs']:return
    m=next(m for m in ledger['models'] if m['key']==job['model']);root=Path(m['root']);folder=Path(job['folder'])
    completion=low.read(folder/'resident_client_completion.json')
    assert completion['complete'] and completion['checkpoint_sha256']==m['weight_sha256']
    runtime=low.read(folder/'runtime.json')
    assert runtime['checkpoint_sha256']==m['weight_sha256'] and runtime['config_sha256']==m['config_sha256']
    low.runtime.verify_trial_folder(folder,root,folder/'case.txt',job['repeats'])
    candidate=deepcopy(ledger)
    candidate['sources'][m['key']][job['dimension']].append({'folder':str(folder),'listing':str(folder/'case.txt'),
           'repeats':job['repeats'],'repeat_offset':job['repeat_offset']})
    candidate['accepted_jobs'].append(job['id'])
    rows=coverage(candidate,m['key'],job['dimension']);check_rows(m,rows)
    low.write(ROOT/'ledger.json',candidate);ledger.clear();ledger.update(candidate)
    dimension=job['dimension'];index=low.DIMS.index(dimension)
    if len(rows)==100:
        target=root/'passes/01/full'/f'{index:02d}_{dimension}';listing=root/'test_lists'/f'{index:02d}_{dimension}.txt'
        if not (target/'verification.json').exists():
            compose(target,root,listing,10,ledger['sources'][m['key']][dimension])
            passes.verify(target,root,listing,10,0,False)
    if all(len(coverage(ledger,m['key'],d))==100 for d in low.DIMS):
        low.write(root/'first_pass_summary.json',{'first_pass_complete':True,'verified_trials':900,
                  'checkpoint':m,'remaining_trials_deferred':900,'resident_scene_schedule':True})
