"""Supervised square-center hover sweep; no grasping or dynamic manipulation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse
import json
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from .aloha_hover_check import RightArm,check_start,validate_path,SETTLE_TOLERANCE_RAD
from .vx300s_kinematics import forward,LOWER,UPPER


class GridFollower:
    """Bounded compensation and speed/height guards for a reviewed hover route."""
    def __init__(self,board,nominal_start,initial_goal):
        self.board=np.asarray(board,dtype=float)
        self.up=-self.board[:3,2]
        self.goal=np.asarray(initial_goal,dtype=float).copy()
        self.correction=np.clip(self.goal-nominal_start,-np.deg2rad(3),np.deg2rad(3))

    def height(self,q):
        return float(self.up@(forward(q,[.16,0,0])[:3,3]-self.board[:3,3]))

    def step(self,nominal,actual,dt):
        nominal,actual=np.asarray(nominal),np.asarray(actual)
        if (nominal.shape!=(6,) or actual.shape!=(6,) or not np.isfinite(nominal).all()
                or not np.isfinite(actual).all() or not np.isfinite(dt) or not 0<dt<=.15):
            raise ValueError('Invalid grid control input or timing')
        if np.max(np.abs(nominal-actual))>np.deg2rad(4):
            raise RuntimeError('Joint following error exceeds 4 degrees')
        if self.height(actual)<.04:
            raise RuntimeError('Measured/model fingertip standoff below 40 mm')
        self.correction=np.clip(self.correction+.6*dt*(nominal-actual),
                                -np.deg2rad(3),np.deg2rad(3))
        requested=np.clip(nominal+self.correction,LOWER,UPPER)
        goal=self.goal+np.clip(requested-self.goal,-np.deg2rad(3)*dt,np.deg2rad(3)*dt)
        if self.height(goal)<.045:
            raise RuntimeError('Proposed goal/model standoff below 45 mm')
        self.goal=goal
        return goal.copy()


def validate_grid(plan):
    if plan.get('execution_blocked'):
        raise ValueError(plan.get('execution_blocked_reason','Grid execution is blocked'))
    if (plan.get('schema')!='right_aloha_hover_grid_v1' or plan['tcp_offset_m']!=[.16,0,0]
            or plan['hover_height_m']<.08 or plan['new_modeled_contacts']):
        raise ValueError('Not a reviewed 160 mm TCP / 80 mm standoff hover plan')
    if plan['minimum_nominal_tip_board_height_m']<.079 or plan['minimum_modeled_table_clearance_m']<.06:
        raise ValueError('Insufficient offline clearance')
    expected={f'R{r}C{c}' for r in range(1,7) for c in range(1,7)}
    if len(plan['route'])!=36 or set(plan['route'])!=expected or len(plan['segments'])!=35:
        raise ValueError('Expected the reviewed 36-square route')
    previous=np.asarray(plan['initial_nominal_q_rad'])
    for label,segment in zip(plan['route'][1:],plan['segments']):
        if segment['square']!=label:raise ValueError('Route/segment labels disagree')
        path=validate_path(segment['trajectory_q_rad'])
        if np.max(np.abs(path[0]-previous))>np.deg2rad(.1):
            raise ValueError('Disconnected grid trajectory')
        speed=np.max(np.abs(np.diff(path,axis=0)))*(len(path)-1)/segment['duration_s']
        if speed>np.deg2rad(2)+1e-6:
            raise ValueError('Nominal grid path exceeds 2 degrees/s')
        previous=path[-1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text());validate_grid(plan)
    print(f'35 remaining square centers; nominal movement {plan["estimated_motion_time_s"]:.0f}s plus settling/dwell',flush=True)
    if not args.execute:return
    from .camera import RGBDCamera
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    (root/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    stop=threading.Event();pause=threading.Event();done=threading.Event()
    state={'status':'preparing','square':'R3C3','phase':'preparing','completed':[],
           'error':None,'motion_sent':False,'last_preview_s':time.monotonic(),
           'expert_episode':False,'calibration_validated':False}
    camera=arm=thread=None
    cv2.setNumThreads(1)
    try:
        camera=RGBDCamera('234222300865');frame,_,_,_=camera.read()
        cv2.imshow('Right ALOHA grid hover - P pause - Q stop',frame);cv2.waitKey(1)
        arm=RightArm();actual=arm.positions();check_start(actual,np.array(plan['initial_actual_q_rad']))
        if np.max(np.abs(actual-plan['initial_nominal_q_rad']))>np.deg2rad(1):
            raise RuntimeError('Starting hover is not close to nominal R3C3')
        follower=GridFollower(plan['T_right_base_board'],np.array(plan['initial_nominal_q_rad']),arm.goals())
        state['last_preview_s']=time.monotonic()
        def run():
            last=time.monotonic();fault_time=0.;nominal=np.array(plan['initial_nominal_q_rad'])
            try:
                with (root/'motion.jsonl').open('w') as log:
                    for segment in plan['segments']:
                        if stop.is_set():break
                        path=np.asarray(segment['trajectory_q_rad']);elapsed=0.;phase='moving'
                        stable_since=None;settle_elapsed=dwell_elapsed=0.
                        state.update(status='running',square=segment['square'],phase=phase)
                        print('Moving to '+segment['square'],flush=True)
                        while not stop.is_set():
                            now=time.monotonic();dt=now-last
                            if dt<.001:stop.wait(.01);continue
                            last=now
                            if now-state['last_preview_s']>1:
                                raise RuntimeError('Operator camera preview stalled')
                            actual=arm.positions()
                            if now-fault_time>.5:
                                for motor in range(1,10):
                                    if arm.read_register(motor,70,1):raise RuntimeError('Motor hardware fault')
                                fault_time=now
                            if phase=='moving' and not pause.is_set():
                                elapsed=min(elapsed+dt,segment['duration_s']);index=elapsed/segment['duration_s']*(len(path)-1)
                                i=min(int(index),len(path)-2);u=index-i;nominal=(1-u)*path[i]+u*path[i+1]
                                if elapsed>=segment['duration_s']:phase='settling'
                            goal=follower.step(nominal,actual,dt)
                            arm.goal(goal);state['motion_sent']=True
                            err=float(np.max(np.abs(actual-nominal)));height=follower.height(actual)
                            state.update(phase=phase,paused=pause.is_set(),model_height_m=height,
                                         max_joint_error_deg=float(np.rad2deg(err)),
                                         movement_fraction=elapsed/segment['duration_s'])
                            log.write(json.dumps({'host_monotonic_s':now,'square':segment['square'],
                                                 'phase':phase,'paused':pause.is_set(),
                                                 'actual_q_rad':actual.tolist(),'nominal_q_rad':nominal.tolist(),
                                                 'goal_q_rad':goal.tolist(),'correction_rad':follower.correction.tolist(),
                                                 'model_height_m':height})+'\n')
                            if phase=='settling' and not pause.is_set():
                                settle_elapsed+=dt
                                if err<SETTLE_TOLERANCE_RAD:
                                    if stable_since is None:stable_since=now
                                    if now-stable_since>.75:
                                        phase='inspection pause';dwell_elapsed=0.
                                        result={'square':segment['square'],'host_monotonic_s':now,
                                                'model_height_m':height,'actual_q_rad':actual.tolist(),
                                                'max_joint_error_deg':float(np.rad2deg(err)),
                                                'visual_acceptance_recorded':False}
                                        state['completed'].append(result)
                                        (root/'completed-squares.json').write_text(json.dumps(state['completed'],indent=2)+'\n')
                                        print('Reached '+segment['square']+f' | {len(state["completed"])}/35 remaining squares',flush=True)
                                else:stable_since=None
                                if settle_elapsed>15 and phase=='settling':
                                    raise RuntimeError('Square did not settle within bounded correction; sweep stopped')
                            elif phase=='inspection pause' and not pause.is_set():
                                dwell_elapsed+=dt
                                if dwell_elapsed>=plan['dwell_s']:break
                            stop.wait(max(0.,1/30-(time.monotonic()-now)))
                    state['status']='complete' if len(state['completed'])==35 else 'operator stop'
                    state['final_q_rad']=arm.positions().tolist();state['final_goal_rad']=follower.goal.tolist()
            except BaseException as exc:
                state['status']='stopped on error';state['error']=str(exc)
            finally:
                state['hold_behavior']='No new goals; hardware faults may independently disable torque. Inspect fault and torque status before recovery.'
                done.set()
        thread=threading.Thread(target=run,name='right-grid-hover',daemon=True);thread.start()
        snapshots=set()
        while not done.is_set():
            frame,_,_,_=camera.read();state['last_preview_s']=time.monotonic()
            view=frame.copy()
            cv2.putText(view,f'{state["square"]} | {state["phase"]} | {len(state["completed"])+1}/36 checked',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            cv2.putText(view,'P: pause/resume | Q: stop | hover only',(8,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            if pause.is_set():cv2.putText(view,'PAUSED - holding current target',(8,75),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            cv2.imshow('Right ALOHA grid hover - P pause - Q stop',view)
            key=cv2.waitKey(1)&0xff
            if key==ord('q'):stop.set()
            if key==ord('p'):
                if pause.is_set():pause.clear()
                else:pause.set()
            for result in list(state['completed']):
                if result['square'] not in snapshots:
                    cv2.imwrite(str(root/(result['square']+'.jpg')),view);snapshots.add(result['square'])
            # Atomic status file permits monitoring without a second serial owner.
            tmp=root/'live-status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'live-status.json')
        thread.join(timeout=15)
        if thread.is_alive():raise RuntimeError('Control thread did not finish')
        frame,_,_,_=camera.read();cv2.imwrite(str(root/'final-pose.jpg'),frame)
        print(json.dumps({k:v for k,v in state.items() if k!='completed'},indent=2),flush=True)
    finally:
        stop.set()
        if thread is not None:thread.join(timeout=15)
        if thread is not None and thread.is_alive():raise RuntimeError('Control thread still active')
        if arm is not None:arm.close()
        if camera is not None:camera.close()
        cv2.destroyAllWindows()
        (root/'summary.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':main()
