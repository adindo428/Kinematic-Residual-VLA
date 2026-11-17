"""Four exploratory frozen-policy arms on the unchanged low-clutter Table1 suite."""
import argparse,copy,fcntl,inspect,json,os,signal,sys,time
from pathlib import Path
import run_closed_loop_pilot as runner
low=runner.low
ROOT=low.WORK/'evaluations/krv-frozen-kra-vpd-vi-20261114'
TRAIN=low.WORK/'training/krv-distance-gate-history-20261113'
SOURCE=TRAIN/'source'


def render():
    plan=low.read(ROOT/'plan.json');scores={m['key']:[0,0] for m in plan['models']}
    for job in plan['jobs']:
        path=Path(job['folder'])/'trial_results.jsonl'
        if path.exists():
            for line in path.read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                scores[job['model']][0]+=bool(r['success']);scores[job['model']][1]+=1
    lines=['# Frozen-checkpoint KRA / VPD / VI ablation','','Exploratory inference-only comparison; 200 low-clutter CR trials per arm. KRV checkpoint: epoch3.943, update26959. DOMV has different training/history conditioning. Supplied35/41/43/46values are not measurements here.','','| Arm | KRA | VPD | VI | Successes / trials | SR (%) |','|---|---|---|---|---:|---:|']
    for m in plan['models']:
        s,n=scores[m['key']];rate=f'{100*s/n:.2f}' if n else 'pending';lines.append(f"| {m['label']} | {m['kr']} | {m['vpd']} | {m['vi']} | {s}/{n} | {rate} |")
    (ROOT/'live_results.md').write_text('\n'.join(lines)+'\n')
    return scores


def status(stage,**kwargs):
    low.write(ROOT/'status.json',{'stage':stage,'updated_utc':runner.stamp(),**kwargs});render()


def prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'plan.json').exists():return low.read(ROOT/'plan.json')
    assert low.read(TRAIN/'training_result.json')['interrupted']
    pointer=low.read(TRAIN/'family/runs/krv100/latest.json');assert pointer['progress']['updates']==26959
    parent=TRAIN/'family/runs/krv100/generations'/pointer['generation'];snapshot=ROOT/'checkpoints/krv';snapshot.mkdir(parents=True,exist_ok=True)
    for name in ('model.safetensors','training_state.pt','config.json','provenance.json'):
        if not (snapshot/name).exists():os.link(parent/name,snapshot/name)
    cfg=low.read(snapshot/'config.json');assert cfg['velocity_conditioning_mode']=='history' and cfg['residual_phase_gate_mode']=='distance_threshold'
    assert cfg['residual_distance_threshold_m']==.05 and cfg['enable_residual_flow']
    # Native export of the existing20k DOMV adapter; no weights/normalization changes.
    baseline=ROOT/'checkpoints/domv';baseline.mkdir(parents=True,exist_ok=True)
    import torch
    from safetensors.torch import save_file
    native=low.WORK/'evaluations/multivla20k-20261029/model_code/domv100'
    source=low.WORK/'training/palmetto-vla-20261026/main8-h200-only-20261022/runs/domv100/latest.pt'
    saved=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    weights={k.removeprefix('policy.'):v for k,v in saved['model'].items()};assert all(k.startswith('policy.') for k in saved['model'])
    if not (baseline/'model.safetensors').exists():save_file(weights,str(baseline/'model.safetensors'))
    bc=low.read(native/'parents/domv100/config.json');bc['delta_timestamps']={'observation':[-2,0]};low.write(baseline/'config.json',bc)
    low.write(baseline/'provenance.json',{'source_checkpoint':str(source),'identity':saved['identity'],'progress':saved['progress'],'native_export_no_weight_change':True})
    del weights,saved
    definitions=[('domv','DOMV baseline',False,False,False,baseline),('kra','KRV trained CV restoration',True,False,False,snapshot),('vpd','KRV + causal profile',True,True,False,snapshot),('vpd_vi','KRV + causal profile + VI',True,True,True,snapshot)]
    models=[]
    for key,label,kr,vpd,vi,cp in definitions:
        models.append({'key':key,'label':label,'kr':kr,'vpd':vpd,'vi':vi,'gated':kr,'source':str(SOURCE),'checkpoint':str(cp),'weight_sha256':low.sha(cp/'model.safetensors'),'config_sha256':low.sha(cp/'config.json')})
    protocol=low.read(low.ROOT/'protocol.json');assert len(protocol['tests'])==90
    jobs=[];prompts={}
    for pass_id in range(2):
        for test in protocol['tests']:
            scene=test['path'];config=low.read(scene);assert low.sha(scene)==test['config_sha256']
            prompts[scene]={str(i+1):text for i,text in enumerate(config['evaluation_prompts'])}
            shift=(test['seed']+pass_id)%4;order=models[shift:]+models[:shift]
            for m in order:
                name=f"p{pass_id+1}-{test['name']}-{m['key']}";folder=ROOT/'blocks'/name;folder.mkdir(parents=True,exist_ok=True)
                (folder/'case.txt').write_text(scene+'\n')
                jobs.append({'id':name,'model':m['key'],'scene':scene,'scene_sha256':test['config_sha256'],'repeats':10,'repeat_offset':pass_id*10,'folder':str(folder),'dimension':test['dimension'],'scored':True,'seed':test['seed'],'vi':m['vi']})
    files=list((low.REPO/'workstation').glob('krv_profile*.py'))+[Path(__file__),low.REPO/'workstation/resident_table1_client.py',low.REPO/'workstation/resident_table1_worker.py']+list(SOURCE.rglob('*.py'))
    plan={'models':models,'jobs':jobs,'tests':protocol['tests'],'saved_prompts':prompts,'created_utc':runner.stamp(),'training_checkpoint_progress':pointer['progress'],'files_sha256':{str(p.resolve()):low.sha(p) for p in files},'protocol':{'suite':'Exact original low-clutter Table1:90configurations x20repeats per arm','trials_per_arm':1800,'total_trials':7200,'inference_only':True,'matched_training_claim':False,'baseline_caveat':'Existing DOMV uses current velocity; KRV uses trained two-sample history and different prior exposure','krv_gate':'Original trained5cm distance threshold held fixed for all KRV arms','vpd':'v_plan(tau)=v0+a*tau, a=(v0-vprev)/0.08s, fixed at chunk planning; added displacement0.5*a*tau^2 capped to5cm; original CV restoration/centering retained','vi':'Integral of newly observed velocity minus the active frozen planned displacement; additive innovation cap5cm, slew5mm/step, original gripper-close fade','oracle_future_used':False,'neural_inputs_unchanged_from_each_checkpoint_training':True,'original_scene_physics_prompts_and_execution_unchanged':True}}
    low.write(ROOT/'plan.json',plan);low.write(ROOT/'protocol.json',plan['protocol']);status('prepared',total=7200)
    return plan


def verify(plan):
    import numpy as np
    resets={};scores=render()
    for job in plan['jobs']:
        folder=Path(job['folder']);receipt=folder/'resident_client_completion.json'
        if receipt.exists():
            r=low.read(receipt);assert r['complete'] and r['trials']==job['repeats']
        else:
            assert job.get('reused_partial') and len((folder/'trial_results.jsonl').read_text().splitlines())==job['repeats']
        for line in (folder/'reset_states.jsonl').read_text().splitlines():
            row=json.loads(line);key=(job['scene'],job['repeat_offset']+row['trial']);s=row['state']
            if key in resets:
                for field in ('robot_joint_pos','robot_joint_vel'):np.testing.assert_allclose(resets[key][field],s[field],atol=1e-5,rtol=0)
                for field in s['objects']:np.testing.assert_allclose(resets[key]['objects'][field],s['objects'][field],atol=1e-5,rtol=0)
            else:resets[key]=s
    assert all(n==plan['protocol']['trials_per_arm'] for s,n in scores.values())
    low.write(ROOT/'completion.json',{'verified':True,'scores':scores,'matched_resets_verified':True,'inference_only':True,'matched_training':False})


def launch(plan):
    runner.ROOT=ROOT;runner.status=status;runner.verify=verify
    text=inspect.getsource(runner.launch).replace("'closed-loop-pilot'","'krv-profile-ablation'").replace('closed_loop_pilot_server.py','krv_profile_ablation_server.py').replace('closed_loop_pilot_client.py','krv_profile_ablation_client.py').replace('total=75',"total=len(plan['jobs'])").replace('completed=75',"completed=len(plan['jobs'])")
    exec(compile(text,str(Path(__file__).resolve()),'exec'),runner.__dict__)
    runner.launch(plan)


def main():
    args=argparse.ArgumentParser();args.add_argument('--prepare-only',action='store_true');a=args.parse_args();plan=prepare()
    if a.prepare_only:print(json.dumps({'models':len(plan['models']),'trials':7200}));return
    signal.signal(signal.SIGTERM,lambda *_:runner.STOP.set());signal.signal(signal.SIGINT,lambda *_:runner.STOP.set())
    with (ROOT/'runner.lock').open('a') as own,(low.ROOT/'queue.lock').open('a') as gpu:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);status('waiting_for_gpu_lock')
        while not runner.STOP.is_set():
            try:fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:runner.STOP.wait(5)
        launch(plan)

if __name__=='__main__':main()
