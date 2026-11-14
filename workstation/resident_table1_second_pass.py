"""Schedule disjoint repeats 11--20 and merge them with the verified first pass."""
from copy import deepcopy
from pathlib import Path

import resident_table1_schedule as first

low=first.low
passes=first.passes
ROOT=first.ROOT/'pass02'
coverage=first.coverage
check_rows=first.check_rows


def validate_first_pass(ledger):
    for model in ledger['models']:
        root=Path(model['root'])
        for i,dimension in enumerate(low.DIMS):
            assert len(coverage(ledger,model['key'],dimension))==100
            passes.verify(root/'passes/01/full'/f'{i:02d}_{dimension}',root,
                          root/'test_lists'/f'{i:02d}_{dimension}.txt',10,0,False)


def freeze():
    if (ROOT/'ledger.json').exists():return low.read(ROOT/'ledger.json')
    prior=low.read(first.ROOT/'ledger.json');validate_first_pass(prior)
    ROOT.mkdir(parents=True,exist_ok=True)
    low.write(ROOT/'first_pass_ledger.json',prior)
    ledger={'models':prior['models'],'sources':{m['key']:{d:[] for d in low.DIMS} for m in prior['models']},
            'accepted_jobs':[],'pass_index':2,'global_repeat_offset':10,
            'budget':{'per_column':100,'per_checkpoint':900,'combined_per_checkpoint':1800},
            'first_pass_ledger_sha256':low.sha(ROOT/'first_pass_ledger.json')}
    for m in ledger['models']:
        checkpoint=Path(m['checkpoint'])
        assert low.sha(checkpoint/'model.safetensors')==m['weight_sha256']
        assert low.sha(checkpoint/'config.json')==m['config_sha256']
    low.write(ROOT/'ledger.json',ledger)
    return ledger


def make_plan(ledger,directory):
    # Reuse the first-pass planner's source verification and missing-prefix
    # accounting. Its local repeat numbers remain 1--10 within this pass.
    plan=first.make_plan(ledger,directory)
    prior=low.read(ROOT/'first_pass_ledger.json')
    assert low.sha(ROOT/'first_pass_ledger.json')==ledger['first_pass_ledger_sha256']
    prompts=first.saved_prompts(prior)
    for scene,values in first.saved_prompts(ledger).items():
        prompts.setdefault(scene,{}).update({str(int(k)+10):v for k,v in values.items()})
    plan.update(pass_index=2,global_repeat_range=[11,20],saved_prompts=prompts,
                second_ten_repeats_deferred=False,first_pass_ledger_sha256=ledger['first_pass_ledger_sha256'])
    expected={t['name']:t['config_sha256'] for t in low.read(first.krv.BASE/'protocol.json')['tests']}
    for job in plan['jobs']:
        assert job['scene_sha256']==expected[Path(job['scene']).stem],'Original scene configuration changed'
        job['pass_index']=2
        job['pass_local_repeat_offset']=job['repeat_offset']
        job['repeat_offset']+=10
        assert 10<=job['repeat_offset']<20 and job['repeat_offset']+job['repeats']==20
    for model in ledger['models']:
        successes={r['env_name'] for d in low.DIMS for r in coverage(prior,model['key'],d) if r['success']}
        plan['success_history'][model['key']]=sorted(successes|set(plan['success_history'][model['key']]))
    low.write(Path(directory)/'plan.json',plan)
    return plan


def finalize_dimension(ledger,model,dimension):
    root=Path(model['root']);index=low.DIMS.index(dimension)
    if len(coverage(ledger,model['key'],dimension))!=100:return
    listing=root/'test_lists'/f'{index:02d}_{dimension}.txt'
    target=root/'passes/02/full'/f'{index:02d}_{dimension}'
    if not (target/'composition.json').exists():
        first.compose(target,root,listing,10,ledger['sources'][model['key']][dimension])
    passes.verify(target,root,listing,10,10,False)
    full=root/'full'/f'{index:02d}_{dimension}'
    if (full/'trial_results.jsonl').exists():
        receipt=low.read(full/'verification.json')
        expected=[str(root/'passes'/f'{p:02d}'/'full'/f'{index:02d}_{dimension}') for p in (1,2)]
        assert receipt.get('source_pass_folders')==expected,'Preserve unrelated full-suite results'
    passes.merge(root,dimension,index,listing,False)
    if all(len(coverage(ledger,model['key'],d))==100 for d in low.DIMS):
        passes.summarize(root,low.DIMS,model)


def accept(ledger,job):
    if job['id'] in ledger['accepted_jobs']:
        return
    model=next(m for m in ledger['models'] if m['key']==job['model'])
    folder=Path(job['folder']);done=low.read(folder/'resident_client_completion.json')
    assert done['complete'] and done['job']==job and done['checkpoint_sha256']==model['weight_sha256']
    runtime=low.read(folder/'runtime.json')
    assert runtime['checkpoint_sha256']==model['weight_sha256']
    assert runtime['config_sha256']==model['config_sha256'] and runtime['repeat_offset']==job['repeat_offset']
    assert 10<=job['repeat_offset']<20
    low.runtime.verify_trial_folder(folder,Path(model['root']),folder/'case.txt',job['repeats'])
    candidate=deepcopy(ledger)
    candidate['sources'][model['key']][job['dimension']].append({
        'folder':str(folder),'listing':str(folder/'case.txt'),'repeats':job['repeats'],
        'repeat_offset':job['repeat_offset']-10})
    candidate['accepted_jobs'].append(job['id'])
    check_rows(model,coverage(candidate,model['key'],job['dimension']))
    low.write(ROOT/'ledger.json',candidate);ledger.clear();ledger.update(candidate)
    finalize_dimension(ledger,model,job['dimension'])


def recover_interrupted(ledger):
    for model in ledger['models']:
        for dimension in low.DIMS:finalize_dimension(ledger,model,dimension)
    for path in sorted((ROOT/'runs').glob('run*/plan.json')):
        run=path.parent;state=low.read(run/'run_status.json') if (run/'run_status.json').exists() else {}
        if state.get('state') not in ('paused','failed'):continue
        for job in low.read(path)['jobs']:
            if job['id'] in ledger['accepted_jobs']:continue
            folder=Path(job['folder']);file=folder/'trial_results.jsonl'
            if not file.exists():continue
            if (folder/'resident_client_completion.json').exists():accept(ledger,job);continue
            if any(s['folder']==str(folder) for s in ledger['sources'][job['model']][job['dimension']]):continue
            if not first.read_records(folder):continue
            assert low.read(folder/'resident_block.json')['job']==job
            model=next(m for m in ledger['models'] if m['key']==job['model'])
            runtime=low.read(folder/'runtime.json')
            assert runtime['checkpoint_sha256']==model['weight_sha256'] and not runtime['parallel_inference']
            candidate=deepcopy(ledger)
            candidate['sources'][job['model']][job['dimension']].append(first.prefix(folder,job['repeat_offset']-10))
            check_rows(model,coverage(candidate,job['model'],job['dimension']))
            low.write(ROOT/'ledger.json',candidate);ledger.clear();ledger.update(candidate)


def prepare():
    ledger=freeze()
    if not (ROOT/'request.json').exists():
        request=deepcopy(low.read(first.ROOT/'request.json'))
        names=['resident_table1_worker.py','resident_table1_server.py','resident_table1_client.py',
               'resident_table1_schedule.py','resident_table1_second_pass.py','run_resident_table1_second_pass.py',
               'run_resident_table1_queue.py','pilot_resident_table1.py','table1_resume_records.py','table1_two_pass.py']
        request.update(models=ledger['models'],pass_index=2,global_repeat_range=[11,20],
                       remaining_ten_repeats_deferred=False,
                       code_sha256={str(low.REPO/'workstation'/n):low.sha(low.REPO/'workstation'/n) for n in names})
        low.write(ROOT/'request.json',request)
    return ledger
