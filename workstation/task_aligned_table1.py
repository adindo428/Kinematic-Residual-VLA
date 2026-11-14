#!/usr/bin/env python3
"""Prepare and run a task-aligned DOM mini-suite after KRV100 epoch ten."""
import argparse
from collections import Counter,defaultdict
import copy
import csv
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'evaluation/official_dom/scripts'))
import local as official
from prepare import sha256,write_json
from report import read_records,verify_folder
from epoch10_heldout_evaluation import dependency as epoch10_dependency


def dependency(request):
    if request.get('checkpoint_selection')!='best_validation_after_user_stop':
        return epoch10_dependency(request)
    active=subprocess.run(['systemctl','--user','is-active',request['training_service']],
                          capture_output=True,text=True,check=False).stdout.strip()
    if active in ('active','activating','deactivating'):
        return None
    if sha256(request['stop_receipt'])!=request['stop_receipt_sha256']:
        raise ValueError('Checkpointed training-stop receipt changed')
    receipt=read(request['stop_receipt'])
    if not receipt.get('checkpoint_saved') or receipt['selected_epoch']!=request['target_epoch']:
        raise ValueError('Training stop and checkpoint selection are not confirmed')
    run=Path(request['family_root'])/'runs/krv100'
    if read(run/'best.json')!=request['selected_pointer']:
        raise ValueError('Best validation checkpoint changed after selection')
    checkpoint=run/'generations'/request['selected_pointer']['generation']
    for name,expected in request['selected_files_sha256'].items():
        if sha256(checkpoint/name)!=expected:
            raise ValueError('Selected checkpoint content changed')
    return checkpoint


def read(path):
    return json.loads(Path(path).read_text())


def state(root,phase,**details):
    value={'state':phase,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
           'expected_trials':360,**details}
    write_json(root/'suite_status.json',value)
    return value


def category(name):
    stem=Path(name).stem
    return re.sub(r'\d+d?$','',stem)


def prepare(stage,root):
    if (root/'request.json').exists():
        validate(read(root/'request.json'))
        print('Existing task-aligned suite reused',flush=True)
        return
    plan=read(stage/'plan.json');family=stage/'family'
    records=read(family/'episodes.json')
    train=set(read(family/'split.json')['splits']['train'])
    by_id={r['episode_index']:r for r in records}
    support=defaultdict(Counter)
    data=Path(read(family/'manifest.json')['dataset_root'])
    with (data/'meta/camera.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);ep=row['episode_index']
            if ep in train:
                # The collection filename identifies the principal object.
                token=row['filename'].split('_')[2]
                support[by_id[ep]['task']][category(token)]+=1
    baseline=Path(plan['parent']).parents[1]/'results/dom-paper45-a6000-20261010-02'
    original=read(baseline/'protocol.json')
    root.mkdir(parents=True,exist_ok=True)
    configs=root/'configs';configs.mkdir(exist_ok=True)
    def eligible(test,task):
        cfg=read(test['path'])
        objects=[v for key,v in cfg['scene'].items() if re.fullmatch(r'object\d*',key)
                 and isinstance(v,dict) and v.get('spawn',{}).get('usd_path')]
        cats=[category(v['spawn']['usd_path']) for v in objects]
        needed=('long_horizon',) if task=='long_horizon' else ('pick','place')
        return all(support[t][c]>0 for t in needed for c in cats)
    def rank(test):
        return hashlib.sha256(('task-aligned-v1|42|'+test['name']).encode()).hexdigest()
    chosen=[];used_categories=set()
    for dim in original['dimension_order']:
        task='long_horizon' if dim=='LS' else 'place'
        candidates=sorted([t for t in original['tests'] if t['dimension']==dim and eligible(t,task)],key=rank)
        if not candidates:
            raise ValueError(f'No locally supported training-task template for {dim}')
        # Favor distinct object categories using metadata alone, before outcomes.
        candidates.sort(key=lambda t:category(read(t['path'])['scene']['object']['spawn']['usd_path']) in used_categories)
        count=2 if dim=='LS' else 1
        for test in candidates[:count]:
            cfg=read(test['path'])
            used_categories.add(category(cfg['scene']['object']['spawn']['usd_path']))
            for actual_task in (('long_horizon',) if dim=='LS' else ('pick','place')):
                variant=copy.deepcopy(cfg)
                name=test['name']
                if actual_task=='pick':
                    name=name.replace('_place_','_pick_',1)
                    placed=variant['terminations'].pop('objects_placed')
                    variant['terminations'].pop('container_dropping',None)
                    variant['terminations']['object_picked']={
                        'func':'configs.termination_cfg:is_object_picked',
                        'params':{'goal_position':placed['params']['goal_position'],
                                  'tolerance':placed['params']['tolerance']},'time_out':False}
                    variant['instruction']={'task':'pick','objects':cfg['instruction']['objects']}
                path=configs/(name+'.json')
                write_json(path,variant)
                chosen.append({'name':name,'path':str(path),'dimension':dim,'task':actual_task,
                    'seed':variant['seed'],'horizon_seconds':variant['episode_length_s'],
                    'config_sha256':sha256(path),'parent_config':test['path'],
                    'parent_config_sha256':test['config_sha256'],
                    'object_category':category(cfg['scene']['object']['spawn']['usd_path']),
                    'changes':(['instruction.task/objects','placement success replaced with upstream pick success']
                               if actual_task=='pick' else [])})
    if len(chosen)!=18 or Counter(t['task'] for t in chosen)!=Counter(pick=8,place=8,long_horizon=2):
        raise ValueError('Expected 8 pick, 8 place and 2 sequencing configurations')
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('frozen_dom_experiment',Path(plan['source'])/'utils/dom_experiment.py')
    module=module_from_spec(spec);spec.loader.exec_module(module)
    assets={}
    for case in chosen:
        cfg=read(case['path'])
        for key,node in cfg['scene'].items():
            if not isinstance(node,dict) or not node.get('spawn',{}).get('usd_path'):
                continue
            usd=Path(node['spawn']['usd_path'])
            if key=='house':
                asset=REPO/'scratch/dom45/assets/scenes'/usd.name
            elif node.get('class_type')=='isaaclab.assets.rigid_object.rigid_object:RigidObject':
                asset=REPO/'scratch/dom45/assets/objects'/usd.parent.name/usd.name
            else:
                continue
            if not asset.is_file():
                raise FileNotFoundError(asset)
            assets[str(asset)]=sha256(asset)
    lists=root/'test_lists';lists.mkdir(exist_ok=True)
    for i,dim in enumerate(original['dimension_order']):
        paths=[t['path'] for t in chosen if t['dimension']==dim]
        (lists/f'{i:02d}_{dim}.txt').write_text('\n'.join(paths)+'\n')
    pilot=[]
    for i,dim in enumerate(original['dimension_order']):
        options=[t for t in chosen if t['dimension']==dim]
        pilot.append(options[i%len(options)]['path'])
    (lists/'pilot.txt').write_text('\n'.join(pilot)+'\n')
    (lists/'all.txt').write_text('\n'.join(t['path'] for t in chosen)+'\n')
    coverage={'train_episode_counts':dict(Counter(by_id[e]['task'] for e in train)),
        'training_object_categories_by_task':{k:dict(v) for k,v in support.items()},
        'eval_config_counts':dict(Counter(t['task'] for t in chosen)),
        'eval_trials':{'pick':160,'place':160,'long_horizon':40},
        'selection':'Seed-42 SHA256 ranking of eligible public templates, preferring object-category diversity. No policy outcomes used.',
        'speed_policy':'Preserve the original per-condition dynamics, forces, resets and physics; no claim of unseen-speed exclusion.',
        'object_policy':'Object categories must occur in both pick/place training data (or long-horizon data for LS). VG retains unseen asset instances of trained categories.',
        'scope':'New scenario construction from public DOM templates; not a replay of exact training/validation episodes.'}
    write_json(root/'training_task_coverage.json',coverage)
    request={'user_authorized':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'target_epoch':10,'family_root':str(family),'manifest':plan['manifest'],
        'experiment_fingerprint':plan['fingerprint'],'policy_source':plan['source'],
        'results':str(root),'expected_trials':360,'pilot_trials':9,'repeats_per_case':20,
        'partition':'new_task_aligned_scenarios','cases':chosen,'asset_sha256':assets,
        'dimension_order':original['dimension_order'],'official_stage':str(REPO/'scratch/dom45/official'),
        'baseline_results':str(baseline),'assets':str(REPO/'scratch/dom45/assets'),
        'isaaclab':str(REPO/'workstation/isaaclab-paper45.sh'),
        'policy_python':str(REPO/'scratch/dom45/envs/train/bin/python'),
        'training_service':'krvla-krv100-finish-epoch10-20261024.service','wandb_id':'2qh0ijb1',
        'kit_args':read(Path(plan['parent'])/'epoch1_table1_request.json')['kit_args'],
        'prior_exposure':'Checkpoint has two preceding full-data epochs, then 20k continuation; no pristine 20k-only or held-out claim.',
        'report_scope':'Custom task-aligned Table-1-style mini-suite: 18 configurations x 20 repeats, 40 trials per condition. Not the official 1800-trial Table 1 benchmark.'}
    files=[Path(__file__),Path(__file__).with_name('epoch10_heldout_evaluation.py')]
    files.extend(Path(__file__).with_name(n) for n in
                 ('krvla_table1_server.py','krvla_table1_probe.py','krvla_policy_runtime.py','inference_krvla.py',
                  'run-krv100-task-table1.sh'))
    files.extend(REPO/'evaluation/official_dom/scripts'/n for n in
                 ('local.py','prepare.py','report.py','server_observer.py'))
    files.extend(lists.iterdir());files.append(root/'training_task_coverage.json')
    request['files_sha256']={str(p):sha256(p) for p in files}
    write_json(root/'request.json',request)
    write_json(stage/'epoch10_task_evaluation_request.json',request)
    validate(request)
    state(root,'prepared_waiting_epoch10')
    print(json.dumps({'prepared':True,'configurations':18,'trials':360,'tasks':coverage['eval_trials']}),flush=True)


def validate(request):
    if (not request.get('user_authorized')
            or (request['target_epoch']!=10 and request.get('checkpoint_selection')!='best_validation_after_user_stop')
            or request['partition']!='new_task_aligned_scenarios' or request['expected_trials']!=360):
        raise ValueError('Missing task-aligned evaluation authorization')
    for path,expected in request['files_sha256'].items():
        if sha256(path)!=expected:
            raise ValueError(f'Frozen evaluation implementation changed: {path}')
    for test in request['cases']:
        if sha256(test['path'])!=test['config_sha256'] or sha256(test['parent_config'])!=test['parent_config_sha256']:
            raise ValueError('Scenario changed after selection')
    for path,expected in request['asset_sha256'].items():
        if sha256(path)!=expected:
            raise ValueError('Scenario asset changed')
    sys.path.insert(0,request['policy_source'])
    from core.dom_training import load_family
    manifest=load_family(request['family_root'])
    if manifest['fingerprint']!=request['manifest']:
        raise ValueError('Checkpoint training family changed')
    return manifest


def bind(request,root,checkpoint):
    if (root/'protocol.json').exists():
        protocol=read(root/'protocol.json')
        if (protocol['trained_checkpoint']!=str(checkpoint)
                or protocol['checkpoint_weights_sha256']!=sha256(checkpoint/'model.safetensors')):
            raise ValueError('Existing suite belongs to another checkpoint')
        return protocol
    stage=root/'stage';(stage/'checkpoint').mkdir(parents=True,exist_ok=True)
    source=Path(request['official_stage'])/'source'
    if not (stage/'source').exists():
        (stage/'source').symlink_to(source,target_is_directory=True)
    for name in ('config.json','model.safetensors','provenance.json'):
        target=stage/'checkpoint'/name
        if not target.exists():
            os.link(checkpoint/name,target)
        elif sha256(target)!=sha256(checkpoint/name):
            raise ValueError('Staged checkpoint changed')
    config=read(checkpoint/'config.json')
    if (config['completed_epochs']!=request['target_epoch'] or config['experiment_fingerprint']!=request['experiment_fingerprint']
            or config['wandb_id']!=request['wandb_id'] or not config['enable_residual_flow']
            or not config['condition_on_object_velocity']):
        raise ValueError('Expected the selected validated KRV100 checkpoint')
    protocol=read(Path(request['baseline_results'])/'protocol.json')
    for key in ('paper_reference','test_jsons_identical_to_official_archive','unique_scene_usd_files_in_release'):
        protocol.pop(key,None)
    protocol.update(benchmark='Task-aligned DOM mini-suite',checkpoint=str(stage/'checkpoint'),
        trained_checkpoint=str(checkpoint),checkpoint_repo='Local KRV100 continuation',
        checkpoint_revision=checkpoint.name,checkpoint_kind='locally_trained',
        checkpoint_config_sha256=sha256(checkpoint/'config.json'),
        checkpoint_weights_sha256=sha256(checkpoint/'model.safetensors'),
        completed_training_epochs=request['target_epoch'],training_manifest=request['manifest'],
        checkpoint_selection=request.get('checkpoint_selection','epoch10_final'),
        expected_trials=360,configurations=18,configurations_per_dimension=2,
        dimension_order=request['dimension_order'],tests=request['cases'],asset_sha256=request['asset_sha256'],
        policy_source=request['policy_source'],evaluation_code_sha256=request['files_sha256'],
        prior_exposure=request['prior_exposure'],report_scope=request['report_scope'],
        velocity_source='Current simulator object pose and linear velocity; privileged analytic and neural inputs',
        custom_pick_success='Upstream is_object_picked: target object and end effector at the configured raised goal within tolerance.',
        limitations=['Adapted condition labels; pick variants do not test container-selection reasoning.',
                     'Two configurations per condition; 20 upstream repeats retain their original reset/instruction policy.',
                     request['report_scope'],request['prior_exposure']])
    write_json(root/'protocol.json',protocol)
    return protocol


def report(root,request):
    all_records=[];dimension_rows=[]
    for i,dim in enumerate(request['dimension_order']):
        folder=root/'full'/f'{i:02d}_{dim}'
        result=verify_folder(folder,root,root/'test_lists'/f'{i:02d}_{dim}.txt',20)
        if result['trials']!=40:
            raise ValueError('Incorrect per-condition trial count')
        all_records.extend(read_records(folder))
        dimension_rows.append({'Condition':dim,'Trials':40,'Success':result['success_percent'],
                               'Time':result['mean_task_time_s']})
    if len(all_records)!=360:
        raise ValueError('Incomplete mini-suite')
    tests={t['name']:t for t in request['cases']};task_records=defaultdict(list)
    for row in all_records:
        task_records[tests[row['env_name']]['task']].append(row)
    overall={'Success':100*statistics.mean(r['success'] for r in all_records),
             'Time':statistics.mean(r['task_time_s'] for r in all_records)}
    task_rows=[{'Task':task,'Trials':len(rows),'Success':100*statistics.mean(r['success'] for r in rows),
                'Time':statistics.mean(r['task_time_s'] for r in rows)} for task,rows in sorted(task_records.items())]
    observed={r['Condition']:r['Success'] for r in dimension_rows};observed.update(overall)
    write_json(root/'summary.json',{'complete':True,'verified_trials':360,'observed':observed,
        'conditions':dimension_rows,'tasks':task_rows,'scope':request['report_scope'],'protocol':read(root/'protocol.json')})
    columns=['Method',*request['dimension_order'],'Success','Time']
    with (root/'table1_task_subset.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        writer.writerow({'Method':f"KRV100 epoch {request['target_epoch']}",**observed})
    lines=[f"# KRV100 epoch {request['target_epoch']}: task-aligned Table 1 subset",'',request['report_scope'],'',
        '| '+' | '.join(columns)+' |','|'+'|'.join(['---']+['---:']*(len(columns)-1))+'|',
        f"| KRV100 epoch {request['target_epoch']} | "+' | '.join(f'{observed[c]:.2f}' for c in columns[1:])+' |','',
        '| Task | Trials | Success (%) | Time (s) |','|---|---:|---:|---:|']
    lines.extend(f"| {r['Task']} | {r['Trials']} | {r['Success']:.2f} | {r['Time']:.2f} |" for r in task_rows)
    lines.extend(['','Time includes failed trials and timeouts; pilot trials are excluded.',
                  'Original condition labels are inherited from scene templates; pick variants do not exercise container selection.',
                  'No published/full-suite baseline score is presented as a comparison on this custom cohort.',request['prior_exposure'],''])
    (root/'table1_task_subset.md').write_text('\n'.join(lines))


def run(root):
    request=read(root/'request.json');validate(request)
    with (root/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            checkpoint=dependency(request)
            phase='waiting_training_stop' if request.get('checkpoint_selection') else 'waiting_epoch10'
            if checkpoint is not None:
                gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                   capture_output=True,text=True,check=True)
                if not gpu.stdout.strip():break
                phase='waiting_gpu_release'
            record=state(root,phase)
            if phase!=previous:print(json.dumps(record),flush=True);previous=phase
            time.sleep(30)
        validate(request);protocol=bind(request,root,checkpoint)
        args=SimpleNamespace(stage=root/'stage',root=root,assets=Path(request['assets']),
            isaaclab=Path(request['isaaclab']),policy_python=Path(request['policy_python']),
            policy_source=Path(request['policy_source']),policy_probe=REPO/'workstation/krvla_table1_probe.py',
            port=27000,sim_gpu=0,policy_gpu=1,kit_args=request['kit_args'],dry_run=False,resume_verified=True)
        original=official.commands
        def commands(*positional,**keywords):
            server,client=original(*positional,**keywords)
            server[2]=str(REPO/'workstation/krvla_table1_server.py')
            client[2]=str(REPO/'workstation/inference_krvla.py')
            client.extend(['--source',request['policy_source']])
            client[client.index('-a')+1]=f"KRV100_epoch{request['target_epoch']:03d}_task_subset"
            return server,client
        official.commands=commands
        import signal
        def interrupted(signum,frame):raise KeyboardInterrupt(f'Evaluation interrupted by {signum}')
        signal.signal(signal.SIGTERM,interrupted)
        try:
            if not (root/'pilot_verified.json').exists():
                state(root,'pilot',checkpoint=str(checkpoint));official.run_one(args,'pilot')
            else:
                verify_folder(root/'pilot',root,root/'test_lists/pilot.txt',1)
            for i,dim in enumerate(protocol['dimension_order']):
                validate(request);state(root,'running',dimension=dim,checkpoint=str(checkpoint))
                official.run_one(args,'full',i)
            report(root,request);state(root,'complete',verified_trials=360,checkpoint=str(checkpoint))
        except BaseException as error:
            state(root,'failed',error=str(error));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run'))
    parser.add_argument('--stage',type=Path)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.stage.resolve(),args.root.resolve())
    else:run(args.root.resolve())


if __name__=='__main__':main()
