"""Saved-observation integration check for evaluation transport and LAAS; no devices."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import cv2
from smoke_real_finetune import LeRobotDataset, PROJECT
sys.path.insert(0,str(PROJECT))
from real_robot.policy_stream import ActionStream
from real_robot.evaluate import prepare_checkpoint,read_run_contract


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--checkpoint',choices=['best','last'],default='best')
    p.add_argument('--model',choices=['krv100','domv'])
    args=p.parse_args()
    contract=read_run_contract(args.run);data=LeRobotDataset(contract['repo_id'],
        root=Path(contract['local_verified_dataset']),split='validation',revision=contract['revision'],
        euler_delta_convention='wrapped_xyz_v1',required_features=['observation.images.opst_cam','observation.images.wrist_cam',
        'observation.state','observation.environment_state','action'],delta_timestamps={'observation':[-2,0],'action':list(range(20))})
    for i in range(len(data)):
        item=data[i]
        if 'striped ball' in item['task'].lower():break
    else:raise RuntimeError('Missing striped-ball sample')
    payload={}
    for role,key in [('front','observation.images.opst_cam'),('wrist','observation.images.wrist_cam')]:
        # Export is top-padded 384x384; reconstruct 640x480 for exercising raw-image transport.
        images=item[key].permute(0,2,3,1).numpy()
        payload[role]=np.stack([cv2.resize(np.rint(im[96:]*255).clip(0,255).astype(np.uint8)[:,:,::-1],(640,480)) for im in images])
    for short,key in [('state','observation.state'),('environment','observation.environment_state'),
        ('required','dynamic_object.velocity_required'),('valid','dynamic_object.velocity_valid')]:payload[short]=item[key].numpy()
    report={'hardware_accessed':False,'sample_task':item['task'],'note':'Saved validation images reconstructed to raw size for transport; not a closed-loop trial or accuracy measurement.','models':{}}
    for family in ([args.model] if args.model else ['krv100','domv']):
        checkpoint,selected=prepare_checkpoint(args.run,family,args.checkpoint)
        exchange=args.run/'evaluation'/('transport-'+family+'-'+str(time.time_ns()));exchange.mkdir()
        command=[str(PROJECT/'workstation/finetune-python.sh'),'-u',str(PROJECT/'workstation/real_policy_worker.py'),
            '--checkpoint',str(checkpoint.resolve()),'--exchange',str(exchange.resolve()),
            '--image-spec',str(Path(contract['local_verified_dataset'])/'meta/image_preprocessing.json'),'--once']
        with (exchange/'worker.log').open('w') as log:
            child=subprocess.Popen(command,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+60
                while not (exchange/'worker.json').exists():
                    if child.poll() is not None:raise RuntimeError((exchange/'worker.log').read_text()[-1500:])
                    if time.monotonic()>deadline:raise RuntimeError('Worker startup timeout')
                    time.sleep(.05)
                meta=dict(token='transport',index=0,created_monotonic_s=time.monotonic(),instruction=item['task'])
                with (exchange/'observation.tmp').open('wb') as stream:np.savez(stream,metadata=json.dumps(meta),**payload)
                (exchange/'observation.tmp').replace(exchange/'observation.npz')
                child.wait(timeout=30)
                if child.returncode:raise RuntimeError((exchange/'worker.log').read_text()[-1500:])
                with np.load(exchange/'actions.npz',allow_pickle=False) as z:actions=z['actions'];result=json.loads(str(z['metadata']))
                delay=result['inference_finished_s']-meta['created_monotonic_s'];current=math.ceil(delay/.04)
                queue=ActionStream('transport');skipped=queue.merge('transport',0,actions,current)
                selected_action=queue.pop(current)
                assert np.isfinite(actions).all() and actions.shape==(20,7)
                assert selected_action is not None or current>=20
                report['models'][family]=dict(absolute_action_shape=list(actions.shape),finite=True,
                    checkpoint_selection=args.checkpoint,checkpoint_epoch=selected['epoch'],
                    inference_seconds=result['inference_seconds'],transport_to_completion_seconds=delay,
                    elapsed_control_steps=current,discarded_prefix=skipped,valid_action_available=selected_action is not None,
                    absolute_xyz_min=actions[:,:3].min(0).tolist(),absolute_xyz_max=actions[:,:3].max(0).tolist())
                print(json.dumps(report['models'][family]),flush=True)
            finally:
                if child.poll() is None:child.terminate();child.wait(timeout=5)
    filename='transport_check.json' if args.checkpoint=='best' else 'transport_check_last.json'
    (args.run/'evaluation'/filename).write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
