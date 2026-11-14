"""Queue the verified ten-epoch VLASH baseline on the frozen low-clutter suite."""
import argparse
from pathlib import Path
import shutil

import run_resident_adapter_table1 as runner
from evaluation_training_policy import POLICY, require_eligible


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    low=runner.low;root=runner.ROOT;receipt=low.read(a.receipt)
    assert receipt['verified'] and receipt['progress']['epoch']==10 and receipt['progress']['cursor']==0
    previous=low.read(root/'plan.json');assert low.read(root/'completion.json')['complete']
    runner.MODELS[:]=[m['key'] for m in previous['models']]
    completed=low.read(root/'held_records.json')
    completed.update(runner.collect(previous,validate=True))
    assert all(len(rr)==1800 for rr in completed.values())
    legacy=low.read(low.WORK/'evaluations/multivla20k-20261029/catalog.json')
    entry={**legacy['models']['vlash_baseline'],'checkpoint':receipt['checkpoint'],'weight_path':receipt['checkpoint'],
        'checkpoint_sha256':receipt['sha256'],'sha256':receipt['sha256'],'progress':receipt['progress'],
        'identity':receipt['identity'],'checkpoint_bytes':receipt['bytes'],'weight_bytes':receipt['bytes'],
        'checkpoint_mtime_ns':receipt['mtime_ns'],'weight_mtime_ns':receipt['mtime_ns'],
        'selection':'Completed epoch-10 latest checkpoint; not validation-selected','label':'VLASH (epoch 10)',
        'client_kind':'resident_adapter','transfer_receipt':str(a.receipt.resolve()),
        'training_scope':'20k_subset_with_prior_full_data_exposure'}
    assert entry['identity']=='e02ca1ddf295d045cab81269cbc5c69420d2ccce2df0151c7650262bb29b9874'
    stat=Path(entry['checkpoint']).stat();assert (stat.st_size,stat.st_mtime_ns)==(entry['checkpoint_bytes'],entry['checkpoint_mtime_ns'])
    driver=str(Path(runner.__file__).resolve())
    for path,digest in previous['files_sha256'].items():
        if path!=driver:assert low.sha(path)==digest,path
    protocol=low.read(low.ROOT/'protocol.json');assert low.sha(low.ROOT/'protocol.json')==previous['protocol_sha256']
    for test in protocol['tests']:assert low.sha(test['path'])==test['config_sha256']
    stamp=runner.stamp();archive=root/'archives'/('before-vlash-baseline-'+stamp[:19].replace(':','').replace('-',''));archive.mkdir(parents=True)
    for name in ['plan.json','existing.json','request.json','live_results.json','failure.json','completion.json',
                 'completion_notification.json','checkpoint_policy.json','held_records.json','controller.log','execution']:
        path=root/name
        if path.exists():path.rename(archive/name)
    policy=low.read(POLICY);low.write(archive/'training_history_policy.before.json',policy)
    policy['models']['vlash_baseline'].update(training_complete=True,evaluation_eligible=True,
        completed_checkpoint=entry['checkpoint'],completed_progress=entry['progress'],completion_receipt=str(a.receipt.resolve()))
    if 'unfinished_models_held' in policy:policy['unfinished_models_held']=[m for m in policy['unfinished_models_held'] if m!='vlash_baseline']
    policy['updated_utc']=stamp;low.write(POLICY,policy);require_eligible('vlash_baseline',entry)
    low.write(root/'existing.json',{'records':{'vlash_baseline':[]},'sources':[]})
    low.write(root/'held_records.json',completed)
    low.write(root/'checkpoint_policy.json',{'minimum_completed_epochs':10,'enabled_models':['vlash_baseline'],
        'held_models':{},'updated_utc':stamp,'user_authorized':True})
    tests={t['name']:t for t in protocol['tests']};lists={d:[s for s in (low.ROOT/'test_lists'/(d+'.txt')).read_text().splitlines() if s] for d in low.DIMS}
    jobs=[];pilot=lists['CR'][0]
    for i in range(2):
        key=f'vlash-baseline-epoch10-pilot-{i}'
        jobs.append({'id':key,'model':'vlash_baseline','scene':pilot,'scene_sha256':tests[Path(pilot).stem]['config_sha256'],
            'repeats':1,'repeat_offset':0,'folder':str(root/'blocks'/key),'dimension':'CR','scored':False})
    for pass_index,offset in [(1,0),(2,10)]:
        for dim in low.DIMS:
            for scene in lists[dim]:
                key=f'vlash-baseline-epoch10-pass{pass_index}-{dim}-{Path(scene).stem}'
                jobs.append({'id':key,'model':'vlash_baseline','scene':scene,'scene_sha256':tests[Path(scene).stem]['config_sha256'],
                    'repeats':10,'repeat_offset':offset,'folder':str(root/'blocks'/key),'dimension':dim,'scored':True,'pass_index':pass_index})
    for job in jobs:
        folder=Path(job['folder']);folder.mkdir(parents=True,exist_ok=True);assert not (folder/'trial_results.jsonl').exists()
        (folder/'case.txt').write_text(job['scene']+'\n')
    pins={**previous['files_sha256'],driver:low.sha(driver)}
    pins.update({str(path):low.sha(path) for path in Path(entry['code_root']).rglob('*') if path.is_file() and path.suffix in ('.py','.json')})
    spec={'key':'vlash_baseline','entry':entry,'catalog':{k:legacy[k] for k in ['official_source','vlash_source']}}
    plan={**previous,'models':[spec],'jobs':jobs,'files_sha256':pins,'new_scored_trials':1800,
        'success_history':{'vlash_baseline':[]},'previous_plan':str(archive/'plan.json'),'created_utc':stamp}
    plan.pop('partial_training_exception',None)
    low.write(root/'plan.json',plan)
    low.write(root/'request.json',{'enabled':True,'models':['vlash_baseline'],'user_authorized':True,'minimum_completed_epochs':10,
        'minimum_latency_retention':.9,'scene_first':True,'concurrent_inference':False,'block_repeats_maximum':10,
        'automatic_second_pass_authorized':True,'target_per_model':1800,'notify_on_completion':True,'created_utc':stamp})
    catalog=low.read(low.ROOT/'catalog.json');catalog['models']['vlash_baseline']=entry
    if 'vlash_baseline' not in catalog['order']:catalog['order'].insert(catalog['order'].index('vlash_kr'),'vlash_baseline')
    catalog.update(enabled_models=['vlash_baseline'],held_models=[],queue_order=['vlash_baseline']);low.write(low.ROOT/'catalog.json',catalog)
    result=low.ROOT/'results/vlash_baseline';result.mkdir(parents=True,exist_ok=True)
    assert not (result/'summary.json').exists(),'Existing baseline results need explicit provenance review'
    low.write(result/'protocol.json',{**protocol,'checkpoint':entry})
    shutil.copytree(low.ROOT/'test_lists',result/'test_lists',dirs_exist_ok=True)
    low.write(result/'suite_status.json',{'state':'queued','scored_trials_target':1800,'checkpoint':entry['checkpoint']})
    low.write(root/'vlash_baseline_start_receipt.json',{'checkpoint':entry,'retained_completed_models':{k:len(v) for k,v in completed.items()},'scored_trials':1800,'updated_utc':stamp})
    runner.MODELS[:]=['vlash_baseline'];low.write(root/'execution/status.json',{'stage':'preparing','updated_utc':stamp});runner.publish(plan)
    print('Prepared completed VLASH baseline: 90 frozen scenes, 10 + 10 repeats, 1,800 scored trials; prior results retained.')


if __name__=='__main__':main()
