"""Audit recorded VLA data/live geometry and normalization contracts without hardware."""
import argparse
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from real_robot.vx300s_kinematics import forward,joints_from_raw_ticks


def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def distribution(values):
    a=np.asarray(values,float)
    return dict(count=len(a),percentiles_mm=dict(zip(['min','p01','p10','median','p90','p99','max'],np.percentile(a,[0,1,10,50,90,99,100]).__mul__(1000).tolist()))) if len(a) else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    campaign=read(PROJECT/'scratch/real_robot/family-finetuning-v6/campaign.json');dataset=Path(campaign['training_recipe']['local_verified_dataset'])
    train=set(read(dataset/'meta/splits.json')['train']);quality=read(dataset/'meta/quality_policy.json')['usable_frames']
    episodes={x['episode_index']:x for x in map(json.loads,(dataset/'meta/episodes.jsonl').read_text().splitlines())}
    sources=list(map(json.loads,(dataset/'meta/sources.jsonl').read_text().splitlines()))
    arrays=defaultdict(lambda:defaultdict(list));phase_counts=Counter();eligible=[]
    for ep in sorted(train):
        usable=set(quality[str(ep)]);indices=[i for i in sorted(usable) if max(0,i-2) in usable]
        if not indices:continue
        eligible.append(ep);task=json.loads(episodes[ep]['tasks'][0])['objects']
        t=pq.read_table(dataset/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet',columns=['action','observation.state','observation.environment_state'])
        acts=np.asarray(t['action'].to_pylist());state=np.asarray(t['observation.state'].to_pylist());env=np.asarray(t['observation.environment_state'].to_pylist())
        q=list(map(json.loads,(dataset/f'quality/episode_{ep:06d}.jsonl').read_text().splitlines()))
        for i in indices:
            arrays[task]['state_z'].append(state[i,2]);arrays[task]['action_z'].append(acts[i,2]);arrays[task]['object_z'].append(env[i,2])
            phase=q[i]['manipulation_phase'];phase_counts[phase]+=1
            arrays[task]['phase_'+phase+'_action_z'].append(acts[i,2])
            arrays[task]['phase_'+phase+'_action_minus_state_z'].append(acts[i,2]-state[i,2])
    result=dict(hardware_accessed=False,dataset=str(dataset),eligible_training_episodes=len(eligible),phase_counts=dict(phase_counts),
        trained_target_heights={task:{k:distribution(v) for k,v in data.items()} for task,data in arrays.items()})
    # Check original raw-encoder FK against stored collection TCP matrices.
    sessions={sources[ep]['raw_session'] for ep in eligible};fk=[]
    for name in sorted(sessions):
        root=Path(name);manifest=root/'manifest.json';robot=root/'robot.jsonl'
        if not manifest.exists() or not robot.exists():fk.append(dict(session=name,available=False));continue
        m=read(manifest);offset=m['tcp_offset_m'];errors=[];qerrors=[]
        with robot.open() as f:
            for index,line in enumerate(f):
                if index%250:continue
                row=json.loads(line)
                if 'follower_raw_ticks' not in row or 'T_base_tcp_candidate' not in row:continue
                q=joints_from_raw_ticks(row['follower_raw_ticks']);pose=forward(q,offset);stored=np.asarray(row['T_base_tcp_candidate'])
                errors.append(float(np.linalg.norm(pose[:3,3]-stored[:3,3])))
                if 'q_model_rad' in row:qerrors.append(float(np.max(abs(q-np.asarray(row['q_model_rad'])))))
                if len(errors)>=32:break
        fk.append(dict(session=name,available=True,tcp_offset_m=offset,samples=len(errors),max_stored_tcp_position_difference_m=max(errors,default=None),max_stored_joint_difference_rad=max(qerrors,default=None)))
    result['raw_collection_fk_checks']=fk
    rig=read(PROJECT/'real_robot/calibration/collection-rig.json')
    result['live_rig']=dict(tcp_offset_m=rig.get('validated_tcp_offset_m',[.16,0,0]),transform_sha256=digest(rig['T_base_front_candidate']),
        depth_model_sha256=digest(rig.get('depth_corrections',{})),independently_base_calibration_validated=rig['base_calibration_validated'])
    cameras=list(map(json.loads,(dataset/'meta/camera.jsonl').read_text().splitlines()));camera_by_ep={x['episode_index']:x for x in cameras}
    transform_counts=Counter();depth_counts=Counter()
    for ep in eligible:
        calibration=next(x for x in camera_by_ep[ep]['cameras'] if x['source_role']=='front')['calibration']
        transform_counts[digest(calibration.get('T_base_front_candidate'))]+=1
        depth_counts[digest(calibration.get('depth_corrections',{}))]+=1
    result['training_camera_calibration_identities']=dict(base_transforms=dict(transform_counts),depth_models=dict(depth_counts))
    statistics=read(dataset/'meta/train_stats.json');checks={};runs=PROJECT/'scratch/real_robot/finetuning-runs/action-v5-fullsampler-20261110'
    for model in ['dom','domv','krv100']:
        selected=read(runs/model/'best_checkpoint.json');path=Path(selected['path']);config=read(path/'config.json');errors={}
        with safe_open(path/'model.safetensors',framework='pt',device='cpu') as tensors:
            for prefix,key in [('normalize_inputs','observation.state'),('normalize_targets','action'),('unnormalize_outputs','action')]:
                for moment in ['mean','std']:
                    name=prefix+'.buffer_'+key.replace('.','_')+'.'+moment
                    actual=tensors.get_tensor(name).numpy();expected=np.asarray(statistics[key][moment],dtype=np.float32)
                    errors[name]=float(np.max(abs(actual-expected)))
        checks[model]=dict(checkpoint=str(path),normalizer_max_abs_differences=errors,use_delta_action=config['use_delta_action'],
            euler_delta_convention=config['euler_delta_convention'],adapt_to_pi_aloha=config['adapt_to_pi_aloha'],
            object_position_direct_neural_input=False,velocity_neural_input=config['condition_on_object_velocity'])
    result['normalization_checks']=checks
    image=read(dataset/'meta/image_preprocessing.json');result['image_preprocessing']=image
    result['limits']='Contract consistency cannot establish physical camera/base/TCP accuracy; requires an independent observed reference.'
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Saved',a.output/'summary.json')
    print('FK maximum:',max(x['max_stored_tcp_position_difference_m'] or 0 for x in fk if x.get('available')))
    print('Base transforms:',dict(transform_counts),'live',result['live_rig']['transform_sha256'])
    print('Depth models:',dict(depth_counts),'live',result['live_rig']['depth_model_sha256'])
    print('Normalizer maximum differences:',{m:max(x['normalizer_max_abs_differences'].values()) for m,x in checks.items()})
    for task,data in arrays.items():
        print(task,'action',distribution(data['action_z']))
        for key,value in data.items():
            if key.startswith('phase_') and key.endswith('_action_z'):print(key,distribution(value))


if __name__=='__main__':main()
