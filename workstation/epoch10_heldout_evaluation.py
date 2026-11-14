#!/usr/bin/env python3
"""Run 64 exact validation scenes after the resumed KRV100 reaches epoch ten."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
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


def read(path):
    return json.loads(Path(path).read_text())


def status(root,phase,**details):
    record={'state':phase,'updated_at_utc':datetime.now(timezone.utc).isoformat(),
            'expected_trials':64,**details}
    write_json(root/'suite_status.json',record)
    return record


def validate(request):
    if not request.get('user_authorized') or request['target_epoch']!=10 or request['partition']!='validation':
        raise ValueError('Exact epoch-ten validation authorization is required')
    for path,expected in request['files_sha256'].items():
        if sha256(path)!=expected:
            raise ValueError(f'Queued evaluation implementation changed: {path}')
    sys.path.insert(0,request['policy_source'])
    from core.dom_training import load_family
    family=Path(request['family_root'])
    manifest=load_family(family)
    if manifest['fingerprint']!=request['manifest'] or sha256(family/'validation_cases.json')!=request['validation_cases_sha256']:
        raise ValueError('Frozen validation membership changed')
    cases=read(request['cases_file'])
    ids=[row['episode_index'] for row in cases]
    if ids!=read(family/'validation_cases.json') or len(ids)!=64 or len(set(ids))!=64:
        raise ValueError('Expected the exact 64 reserved validation cases')
    split=read(family/'split.json')['splits']
    if not set(ids).issubset(split['validation']) or set(ids).intersection(split['train']):
        raise ValueError('Validation cases overlap the current 20k training set')
    return manifest,cases


def dependency(request):
    family=Path(request['family_root']);stage=family.parent;run=family/'runs/krv100'
    service=subprocess.run(['systemctl','--user','is-active',request['training_service']],
                           text=True,capture_output=True,check=False).stdout.strip()
    pointer=read(run/'latest.json')
    if service=='failed':
        write_json(stage/'training_status.json',{'phase':'failed','error':'Epoch-ten training service failed'})
        raise RuntimeError('Epoch-ten training service failed; evaluation remains gated')
    if service in ('active','activating','deactivating'):
        if pointer['progress']['updates']>158490:
            write_json(stage/'training_status.json',{'phase':'training','family_root':str(family),
                'wandb_id':request['wandb_id'],'early_stopping_disabled':True,'maximum_total_epochs':10,
                'updated_at_utc':datetime.now(timezone.utc).isoformat()})
        return None
    path=run/'training_complete.json'
    if not path.exists() or read(path)['epoch']<10:
        return None
    completed=read(path)
    final=read(run/'final.json')
    if (completed['epoch']!=10 or completed['cursor']!=0 or completed.get('pending_validation')
            or completed['fingerprint']!=request['experiment_fingerprint']
            or final!=pointer or final!=read(run/'epoch-010.json')
            or pointer['progress']['epoch']!=10
            or not (run/'validation-epoch-010.json').is_file()):
        raise ValueError('Epoch-ten validated final checkpoint does not match the request')
    write_json(stage/'training_status.json',{'phase':'complete','completed_epochs':10,
        'wandb_id':request['wandb_id'],'early_stopping_disabled':True,
        'updated_at_utc':datetime.now(timezone.utc).isoformat()})
    return run/'generations'/final['generation']


def mapped_cases(request,cases):
    mapping_path=Path(request['case_map'])
    if not mapping_path.is_file():
        return None,[r['episode_index'] for r in cases]
    mapping=read(mapping_path)
    rows=[];missing=[]
    for row in cases:
        ep=row['episode_index']
        if str(ep) not in mapping:
            missing.append(ep);continue
        path=Path(mapping[str(ep)])
        if not path.is_absolute():
            path=mapping_path.parent/path
        if not path.is_file():
            missing.append(ep);continue
        config=read(path)
        explicit=config.get('episode_index',config.get('dom_episode_index'))
        if (explicit is not None and explicit!=ep) or (explicit is None and path.stem!=row['original_filename']):
            raise ValueError(f'Configuration identity does not match validation episode {ep}')
        if not isinstance(config.get('seed'),int) or config.get('episode_length_s',0)<=0 or 'scene' not in config:
            raise ValueError(f'Invalid original simulator configuration: {path}')
        rows.append((row,path.resolve(),config))
    return rows,missing


def bind(request,root,checkpoint,manifest,rows):
    if (root/'protocol.json').exists():
        protocol=read(root/'protocol.json')
        if (protocol['trained_checkpoint']!=str(checkpoint)
                or protocol['checkpoint_weights_sha256']!=sha256(checkpoint/'model.safetensors')):
            raise ValueError('Existing validation evaluation is bound to another checkpoint')
        return protocol
    from scripts.dom_closed_loop import scene_asset_paths
    stage=root/'stage';(stage/'checkpoint').mkdir(parents=True,exist_ok=True)
    source=Path(request['official_stage'])/'source'
    if not (stage/'source').exists():
        (stage/'source').symlink_to(source,target_is_directory=True)
    for name in ('config.json','model.safetensors','provenance.json'):
        target=stage/'checkpoint'/name
        if target.exists():
            if sha256(target)!=sha256(checkpoint/name):
                raise ValueError('Staged checkpoint content changed')
        else:
            os.link(checkpoint/name,target)
    config=read(checkpoint/'config.json')
    if (config['completed_epochs']!=10 or config['experiment_fingerprint']!=request['experiment_fingerprint']
            or config['wandb_id']!=request['wandb_id'] or not config['enable_residual_flow']
            or not config['condition_on_object_velocity']):
        raise ValueError('Final model does not match the requested trained policy')
    tests=[];assets={}
    for row,path,scene in rows:
        for asset in scene_asset_paths(request['assets'],scene['scene']):
            if not asset.is_file():
                raise FileNotFoundError(f'Validation scene asset missing: {asset}')
            assets[str(asset)]=sha256(asset)
        tests.append({'name':path.stem,'path':str(path),'dimension':row['task'],
            'episode_index':row['episode_index'],'config_sha256':sha256(path),
            'seed':scene['seed'],'horizon_seconds':scene['episode_length_s']})
    if len({t['name'] for t in tests})!=64:
        raise ValueError('Validation scene names are not unique')
    lists=root/'test_lists';lists.mkdir(exist_ok=True)
    # Reuse the existing one-trial-per-scene runner. Its historical output
    # directory is named "pilot"; here it contains the actual 64-case evaluation.
    (lists/'pilot.txt').write_text('\n'.join(t['path'] for t in tests)+'\n')
    protocol=read(Path(request['baseline_results'])/'protocol.json')
    for key in ('paper_reference','configurations_per_dimension','test_jsons_identical_to_official_archive'):
        protocol.pop(key,None)
    protocol.update(benchmark='KRV100 epoch 10: reserved validation subset from committed 20k split',
        checkpoint=str(stage/'checkpoint'),trained_checkpoint=str(checkpoint),
        checkpoint_revision=checkpoint.name,checkpoint_kind='locally_trained',
        checkpoint_repo='Local KRV100 continuation',completed_training_epochs=10,
        checkpoint_weights_sha256=sha256(checkpoint/'model.safetensors'),
        checkpoint_config_sha256=sha256(checkpoint/'config.json'),
        training_manifest=manifest['fingerprint'],policy_source=request['policy_source'],
        policy_source_sha256=manifest['source_sha256'],evaluation_code_sha256=request['files_sha256'],
        expected_trials=64,trials_per_configuration=1,configurations=64,
        dimension_order=['Validation'],tests=tests,asset_sha256=assets,
        velocity_source='Current simulator target-object pose and linear velocity; privileged input',
        prior_exposure=request['prior_exposure'],report_scope=request['report_scope'])
    write_json(root/'protocol.json',protocol)
    return protocol


def evaluate(request,root,checkpoint,manifest,rows):
    protocol=bind(request,root,checkpoint,manifest,rows)
    args=SimpleNamespace(stage=root/'stage',root=root,assets=Path(request['assets']),
        isaaclab=Path(request['isaaclab']),policy_python=Path(request['policy_python']),
        policy_source=Path(request['policy_source']),policy_probe=REPO/'workstation/krvla_table1_probe.py',
        port=27000,sim_gpu=0,policy_gpu=1,kit_args=request['kit_args'],dry_run=False,resume_verified=False)
    original=official.commands
    def commands(*positional,**keywords):
        server,client=original(*positional,**keywords)
        server[2]=str(REPO/'workstation/krvla_table1_server.py')
        client[2]=str(REPO/'workstation/inference_krvla.py')
        client.extend(['--source',request['policy_source']])
        client[client.index('-a')+1]='KRV100_epoch010_validation64'
        return server,client
    official.commands=commands
    status(root,'running_validation64',checkpoint=str(checkpoint))
    if not (root/'pilot/verification.json').exists():
        official.run_one(args,'pilot')
    verified=verify_folder(root/'pilot',root,root/'test_lists/pilot.txt',1)
    records=read_records(root/'pilot')
    metadata={t['name']:t for t in protocol['tests']}
    groups=defaultdict(list)
    for record in records:
        groups[metadata[record['env_name']]['dimension']].append(record)
    groups['Overall']=records
    table=[{'Task':name,'Trials':len(group),'Success_percent':100*statistics.mean(r['success'] for r in group),
            'Time_seconds':statistics.mean(r['task_time_s'] for r in group)} for name,group in groups.items()]
    write_json(root/'summary.json',{'complete':True,'verified_trials':verified['trials'],
        'rows':table,'protocol':protocol,'scope':request['report_scope']})
    import csv
    with (root/'table1_validation_subset.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    lines=['# Epoch 10: 20k-split validation subset','',request['report_scope'],'',
        '| Task | Trials | Success (%) | Time (s) |','|---|---:|---:|---:|']
    lines.extend(f"| {r['Task']} | {r['Trials']} | {r['Success_percent']:.2f} | {r['Time_seconds']:.2f} |" for r in table)
    lines.extend(['','Time includes manipulation failures and timeouts.',request['prior_exposure'],''])
    (root/'table1_validation_subset.md').write_text('\n'.join(lines))
    status(root,'complete',verified_trials=64,checkpoint=str(checkpoint))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args();request=read(args.request);root=Path(request['results'])
    with (root/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            manifest,cases=validate(request)
            previous=None
            while True:
                checkpoint=dependency(request)
                rows,missing=mapped_cases(request,cases)
                phase='waiting_epoch10' if checkpoint is None else 'waiting_original_validation_scene_configs'
                if checkpoint is not None and not missing:
                    gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],
                                       capture_output=True,text=True,check=True)
                    if not gpu.stdout.strip():
                        break
                    phase='waiting_gpu_release'
                state=status(root,phase,missing_scene_episode_ids=missing,case_map=request['case_map'])
                if phase!=previous:
                    print(json.dumps(state),flush=True);previous=phase
                time.sleep(30)
            validate(request)
            evaluate(request,root,checkpoint,manifest,rows)
        except BaseException as error:
            status(root,'failed',error=str(error));raise


if __name__=='__main__':
    main()
