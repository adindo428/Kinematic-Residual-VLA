"""Preserve scored evidence and resume only checkpoints with ten completed epochs."""
from collections import defaultdict
import json
from pathlib import Path

import run_resident_adapter_table1 as runner


def main():
    root=runner.ROOT;low=runner.low
    if (root/'checkpoint_policy.json').exists():
        raise RuntimeError('Ten-epoch policy already prepared; resume its existing plan')
    original=low.read(root/'plan.json')
    runner.MODELS[:]=[m['key'] for m in original['models']]
    records=runner.collect(original,validate=True)
    tests={t['name']:t for t in low.read(low.ROOT/'protocol.json')['tests']}
    partial=[]
    # Completed server records survive an interrupted client. Do not count an
    # in-flight episode without its outcome, prompt and trajectory evidence.
    for job in original['jobs']:
        folder=Path(job['folder']);path=folder/'trial_results.jsonl'
        if not job['scored'] or not path.exists() or (folder/'resident_client_completion.json').exists():continue
        raw=runner.rows(path)
        if not raw:continue
        assert len(raw)<=job['repeats']
        runner.validate_rows(raw,runner.rows(folder/'prompt_trace.jsonl'),runner.rows(folder/'trajectory_metrics.jsonl'),
                             tests[Path(job['scene']).stem],job['repeat_offset'])
        records[job['model']]+=[{**r,'trial':r['trial']+job['repeat_offset'],'dimension':job['dimension'],'source':str(folder)} for r in raw]
        partial.append({'job':job['id'],'preserved_trials':len(raw),'folder':str(folder)})
    eligible=[m for m in original['models'] if m['entry']['progress']['epoch']>=10]
    selected=[m['key'] for m in eligible]
    assert selected==['smolvla_kr','vlash_kr','pi05_baseline']
    for name,rr in records.items():
        keys=[(r['env_name'],r['trial']) for r in rr]
        assert len(keys)==len(set(keys)),name
        assert all(1<=trial<=20 for _,trial in keys)
    # Source edits here concern queue policy/display only. Model, recipe,
    # normalization, simulator and inference pins must remain unchanged.
    driver=str(Path(runner.__file__).resolve())
    for path,digest in original['files_sha256'].items():
        if path!=driver:assert low.sha(path)==digest,path
    stamp=runner.stamp();tag=stamp[:19].replace(':','').replace('-','')
    archive=root/'archives'/('before-ten-epoch-only-'+tag);archive.mkdir(parents=True)
    for name in ['plan.json','existing.json','request.json','live_results.json','failure.json',
                 'completion.json','failed_four_model_qualification.json','controller.log','execution']:
        path=root/name
        if path.exists():path.rename(archive/name)
    sources=[]
    for name in selected:
        for source in sorted({r['source'] for r in records[name]}):
            sources.append({'model':name,'folder':source,'records_sha256':low.sha(Path(source)/'trial_results.jsonl')})
    low.write(root/'existing.json',{'records':{name:records[name] for name in selected},'sources':sources})
    low.write(root/'held_records.json',{name:rr for name,rr in records.items() if name not in selected})
    held={m['key']:{'progress':m['entry']['progress'],'reason':'Awaiting checkpoint with ten completed epochs'}
          for m in original['models'] if m['key'] not in selected}
    held['vlash_baseline']={'reason':'Latest retained checkpoint has fewer than ten completed epochs'}
    policy={'minimum_completed_epochs':10,'enabled_models':selected,'held_models':held,
            'notify_on_completion':'desktop','updated_utc':stamp,'user_authorized':True}
    low.write(root/'checkpoint_policy.json',policy)
    jobs=[];lists={dim:[s for s in (low.ROOT/'test_lists'/(dim+'.txt')).read_text().splitlines() if s] for dim in low.DIMS}
    pilot=lists['CR'][0]
    for i,name in enumerate([*selected,selected[0]]):
        key=f'epoch10-pilot-{i}-{name}'
        jobs.append({'id':key,'model':name,'scene':pilot,'scene_sha256':tests[Path(pilot).stem]['config_sha256'],
                     'repeats':1,'repeat_offset':0,'folder':str(root/'blocks'/key),'dimension':'CR','scored':False})
    for pass_index in (1,2):
        begin,end=(1,11) if pass_index==1 else (11,21)
        for dim in low.DIMS:
            for scene in lists[dim]:
                for name in selected:
                    present={r['trial'] for r in records[name] if r['env_name']==Path(scene).stem}
                    missing=[i for i in range(begin,end) if i not in present]
                    if not missing:continue
                    assert missing==list(range(missing[0],end))
                    key=f'epoch10-pass{pass_index}-{dim}-{Path(scene).stem}-{name}'
                    jobs.append({'id':key,'model':name,'scene':scene,'scene_sha256':tests[Path(scene).stem]['config_sha256'],
                                 'repeats':len(missing),'repeat_offset':missing[0]-1,'folder':str(root/'blocks'/key),
                                 'dimension':dim,'scored':True,'pass_index':pass_index})
    for job in jobs:
        folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
        assert not (folder/'trial_results.jsonl').exists()
        (folder/'case.txt').write_text(job['scene']+'\n')
    pins={**original['files_sha256'],driver:low.sha(driver)}
    plan={**original,'models':eligible,'jobs':jobs,'files_sha256':pins,
          'success_history':{name:sorted({r['env_name'] for r in records[name] if r['success']}) for name in selected},
          'new_scored_trials':sum(j['repeats'] for j in jobs if j['scored']),
          'previous_plan':str(archive/'plan.json'),'minimum_completed_training_epochs':10,'created_utc':stamp}
    low.write(root/'plan.json',plan)
    low.write(root/'request.json',{'enabled':True,'models':selected,'user_authorized':True,'minimum_completed_epochs':10,
              'minimum_latency_retention':.9,'scene_first':True,'concurrent_inference':False,'block_repeats_maximum':10,
              'automatic_second_pass_authorized':True,'target_per_model':1800,'notify_on_completion':True,'created_utc':stamp})
    catalog=low.read(low.ROOT/'catalog.json');catalog.update(enabled_models=selected,held_models=list(held),queue_order=selected)
    low.write(low.ROOT/'catalog.json',catalog)
    low.write(root/'eligibility_change_receipt.json',{'eligible_models':selected,'held_models':held,'preserved_records':{k:len(v) for k,v in records.items()},
              'partial_blocks_preserved':partial,'remaining_trials':plan['new_scored_trials'],'archive':str(archive),'created_utc':stamp})
    runner.MODELS[:]=selected
    low.write(root/'execution/status.json',{'stage':'preparing','updated_utc':stamp})
    runner.publish(plan)
    print(json.dumps({'eligible_models':selected,'preserved_records':{k:len(v) for k,v in records.items()},
                      'partial_blocks_preserved':partial,'remaining_trials':plan['new_scored_trials']},indent=2))


if __name__=='__main__':main()
