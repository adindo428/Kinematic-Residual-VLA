"""Resume the explicitly authorized partial-training Pi05 KR checkpoint locally."""
from pathlib import Path
import json

import run_resident_adapter_table1 as runner


def main():
    root=runner.ROOT;low=runner.low
    previous=low.read(root/'plan.json')
    assert low.read(root/'completion.json')['complete'],'Do not replace an active evaluation plan'
    runner.MODELS[:]=[m['key'] for m in previous['models']]
    complete=runner.collect(previous,validate=True)
    assert all(len(rr)==1800 for rr in complete.values())
    saved=low.read(root/'held_records.json')['pi05_kr']
    original=low.read(previous['previous_plan'])
    spec=next(m for m in original['models'] if m['key']=='pi05_kr')
    assert spec['entry']['progress']=={'updates':545349,'samples':17451168,'epoch':6,'cursor':1090528}
    assert len(saved)==1130 and len({(r['env_name'],r['trial']) for r in saved})==1130
    sources=[]
    for source in sorted({r['source'] for r in saved}):
        sources.append({'model':'pi05_kr','folder':source,'records_sha256':low.sha(Path(source)/'trial_results.jsonl')})
    driver=str(Path(runner.__file__).resolve())
    for path,digest in previous['files_sha256'].items():
        if path!=driver:assert low.sha(path)==digest,path
    stamp=runner.stamp();tag=stamp[:19].replace(':','').replace('-','')
    archive=root/'archives'/('before-pi05-kr-resume-'+tag);archive.mkdir(parents=True)
    for name in ['plan.json','existing.json','request.json','live_results.json','failure.json','completion.json',
                 'completion_notification.json','checkpoint_policy.json','held_records.json','controller.log','execution']:
        p=root/name
        if p.exists():p.rename(archive/name)
    low.write(root/'existing.json',{'records':{'pi05_kr':saved},'sources':sources})
    low.write(root/'held_records.json',complete)
    policy={'minimum_completed_epochs':10,'enabled_models':['pi05_kr'],
            'held_models':{'vlash_baseline':{'reason':'Awaiting completed ten-epoch checkpoint'}},
            'authorized_exceptions':{'pi05_kr':{'user_authorized':True,'checkpoint_sha256':spec['entry']['checkpoint_sha256'],
                'progress':spec['entry']['progress'],'reason':'User requested existing partial checkpoint evaluation as a fallback before further training resources are available'}},
            'notify_on_completion':'desktop','updated_utc':stamp}
    low.write(root/'checkpoint_policy.json',policy)
    tests={t['name']:t for t in low.read(low.ROOT/'protocol.json')['tests']}
    lists={d:[s for s in (low.ROOT/'test_lists'/(d+'.txt')).read_text().splitlines() if s] for d in low.DIMS}
    jobs=[];pilot=lists['CR'][0]
    for i in range(2):
        key=f'pi05kr-fallback-pilot-{i}'
        jobs.append({'id':key,'model':'pi05_kr','scene':pilot,'scene_sha256':tests[Path(pilot).stem]['config_sha256'],
            'repeats':1,'repeat_offset':0,'folder':str(root/'blocks'/key),'dimension':'CR','scored':False})
    for pass_index in (1,2):
        begin,end=(1,11) if pass_index==1 else (11,21)
        for dim in low.DIMS:
            for scene in lists[dim]:
                present={r['trial'] for r in saved if r['env_name']==Path(scene).stem}
                missing=[i for i in range(begin,end) if i not in present]
                if not missing:continue
                assert missing==list(range(missing[0],end))
                key=f'pi05kr-fallback-pass{pass_index}-{dim}-{Path(scene).stem}'
                jobs.append({'id':key,'model':'pi05_kr','scene':scene,'scene_sha256':tests[Path(scene).stem]['config_sha256'],
                    'repeats':len(missing),'repeat_offset':missing[0]-1,'folder':str(root/'blocks'/key),
                    'dimension':dim,'scored':True,'pass_index':pass_index})
    assert sum(j['repeats'] for j in jobs if j['scored'])==670
    for job in jobs:
        folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True)
        assert not (folder/'trial_results.jsonl').exists()
        (folder/'case.txt').write_text(job['scene']+'\n')
    pins={**previous['files_sha256'],driver:low.sha(driver)}
    plan={**previous,'models':[spec],'jobs':jobs,'files_sha256':pins,'new_scored_trials':670,
          'success_history':{'pi05_kr':sorted({r['env_name'] for r in saved if r['success']})},
          'previous_plan':str(archive/'plan.json'),'created_utc':stamp,'partial_training_exception':policy['authorized_exceptions']}
    low.write(root/'plan.json',plan)
    low.write(root/'request.json',{'enabled':True,'user_authorized':True,'models':['pi05_kr'],
        'minimum_completed_epochs':10,'authorized_exceptions':policy['authorized_exceptions'],'minimum_latency_retention':.9,
        'scene_first':True,'concurrent_inference':False,'block_repeats_maximum':10,'automatic_second_pass_authorized':True,
        'target_per_model':1800,'notify_on_completion':True,'created_utc':stamp})
    catalog=low.read(low.ROOT/'catalog.json');catalog.update(enabled_models=['pi05_kr'],held_models=['vlash_baseline'],queue_order=['pi05_kr'])
    low.write(low.ROOT/'catalog.json',catalog)
    low.write(root/'pi05_kr_resume_receipt.json',{'progress':spec['entry']['progress'],'checkpoint_sha256':spec['entry']['checkpoint_sha256'],
        'retained_trials':len(saved),'remaining_trials':670,'completed_models_preserved':{k:len(v) for k,v in complete.items()},
        'user_authorized':True,'archive':str(archive),'updated_utc':stamp})
    runner.MODELS[:]=['pi05_kr']
    low.write(root/'execution/status.json',{'stage':'preparing','updated_utc':stamp})
    runner.publish(plan)
    print(json.dumps({'resume_model':'pi05_kr','retained_trials':len(saved),'remaining_trials':670,'progress':spec['entry']['progress']},indent=2))


if __name__=='__main__':main()
