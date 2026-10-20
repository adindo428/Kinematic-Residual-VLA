"""Execute one explicitly supplied, sampled right-arm hover-check trajectory.

This is a supervised geometric verification, not a dynamic manipulation backend.
No torque/mode/calibration/gripper writes. Goal positions go only to six right-arm
master joints, with verified secondary IDs for the paired shoulder/elbow motors.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time
import cv2
import numpy as np
from .vx300s_kinematics import LOWER, UPPER, MOTOR_IDS, forward

SETTLE_TOLERANCE_RAD = np.deg2rad(.7)


def validate_path(qpath):
    qpath = np.asarray(qpath, float)
    if qpath.ndim != 2 or qpath.shape[1] != 6 or len(qpath) < 2 or not np.isfinite(qpath).all():
        raise ValueError('Need a finite sampled six-joint trajectory')
    initial_excess = np.maximum(LOWER-qpath[0], qpath[0]-UPPER)
    if np.max(initial_excess) > np.deg2rad(2):
        raise ValueError('Initial model-limit discrepancy exceeds 2 degrees')
    for j in range(6):
        # An existing small parked-pose overrun may only decrease into the range.
        if qpath[0,j] > UPPER[j]:
            violating = qpath[:,j] > UPPER[j]
            if np.any(np.diff(qpath[:,j])[violating[1:]] > 1e-9):
                raise ValueError('Path worsens initial upper-limit overrun')
        elif qpath[0,j] < LOWER[j]:
            violating = qpath[:,j] < LOWER[j]
            if np.any(np.diff(qpath[:,j])[violating[1:]] < -1e-9):
                raise ValueError('Path worsens initial lower-limit overrun')
        lo, hi = min(qpath[0,j],LOWER[j]), max(qpath[0,j],UPPER[j])
        if np.any(qpath[:,j] < lo-1e-9) or np.any(qpath[:,j] > hi+1e-9):
            raise ValueError('Trajectory leaves the joint bounds')
    if np.max(np.abs(qpath[-1]-qpath[0])) > np.deg2rad(150):
        raise ValueError('Requested single-check joint change is too large')
    if np.max(np.abs(np.diff(qpath,axis=0))) > np.deg2rad(2):
        raise ValueError('Trajectory sampling is too coarse')
    return qpath


def check_start(actual, expected):
    if not np.isfinite(actual).all() or np.max(np.abs(actual-expected)) > np.deg2rad(.5):
        raise RuntimeError('Arm moved since planning; make a fresh plan')


def duration_for(qpath, rate_rad_s=np.deg2rad(3)):
    # Trajectory samples are uniformly spaced in quintic trajectory time.
    return max(15., float(np.max(np.abs(np.diff(qpath,axis=0)))*(len(qpath)-1)/rate_rad_s))


class BoundedSettler:
    """Slow outer integral correction for an already-nearby hover target.

    Does not change motor gains. Command correction is limited to 3 degrees;
    command slew is limited to 2 degrees/s. Used only at a fixed hover pose.
    """
    def __init__(self,target,initial_goal):
        self.target=np.asarray(target,dtype=float)
        self.goal=np.asarray(initial_goal,dtype=float).copy()
        self.correction=np.zeros(6)

    def step(self,actual,dt):
        actual=np.asarray(actual,dtype=float)
        if (actual.shape!=(6,) or not np.isfinite(actual).all()
                or not np.isfinite(dt) or not 0<dt<=.15):
            raise ValueError('Invalid fixed-pose feedback')
        if np.max(np.abs(actual-self.target))>np.deg2rad(5):
            raise RuntimeError('Arm is not close enough for bounded settling')
        self.correction=np.clip(self.correction+.6*dt*(self.target-actual),
                                -np.deg2rad(3),np.deg2rad(3))
        target_goal=np.clip(self.target+self.correction,LOWER,UPPER)
        self.goal+=np.clip(target_goal-self.goal,-np.deg2rad(2)*dt,np.deg2rad(2)*dt)
        return self.goal.copy()


class RightArm:
    def __init__(self, require_torque=True):
        import dynamixel_sdk as dxl
        self.dxl = dxl
        path = '/dev/ttyDXL_follower_right'
        owner = subprocess.run(['fuser',path],capture_output=True,text=True)
        if owner.returncode == 0:
            raise RuntimeError('Right follower serial port is already owned: '+owner.stdout.strip())
        self.port = dxl.PortHandler(path)
        self.packet = dxl.PacketHandler(2.)
        if not self.port.openPort():
            raise RuntimeError('Cannot open right follower port')
        try:
            if not self.port.setBaudRate(1_000_000):
                raise RuntimeError('Cannot set host serial baud rate')
            for motor in range(1,10):
                if require_torque and self.read_register(motor,64,1) != 1:
                    raise RuntimeError(f'Motor {motor} torque is off; no torque settings changed')
                if self.read_register(motor,70,1) != 0:
                    raise RuntimeError(f'Motor {motor} has a hardware error')
                if motor != 9 and self.read_register(motor,11,1) != 4:
                    raise RuntimeError('Arm operating mode differs from the audited extended-position mode')
            for shadow, master in [(3,2),(5,4)]:
                if self.read_register(shadow,12,1) != master:
                    raise RuntimeError('Paired-motor secondary ID is not configured as expected')
            expected_models = {i:1120 for i in range(1,8)} | {8:1020,9:1020}
            for motor, model in expected_models.items():
                if self.read_register(motor,0,2) != model:
                    raise RuntimeError('Motor model differs from audited right arm')
            self.reader = dxl.GroupSyncRead(self.port,self.packet,132,4)
            for motor in MOTOR_IDS:
                self.reader.addParam(motor)
            self.writer = dxl.GroupSyncWrite(self.port,self.packet,116,4)
        except BaseException:
            self.port.closePort()
            raise

    def read_register(self,motor,address,width):
        value,comm,error = getattr(self.packet,f'read{width}ByteTxRx')(self.port,motor,address)
        if comm != self.dxl.COMM_SUCCESS or error&127:
            raise RuntimeError(f'Motor {motor} register read failed: {comm}/{error}')
        if error&128:
            if address==70:
                kind='overload' if value&32 else 'hardware fault'
                raise RuntimeError(f'Motor {motor} {kind}: hardware error 0x{value:02x}; torque may be disabled')
            raise RuntimeError(f'Motor {motor} hardware alert during register {address} read; inspect hardware error and torque state')
        return value

    def positions(self):
        if self.reader.txRxPacket() != self.dxl.COMM_SUCCESS:
            raise RuntimeError('Joint feedback failed')
        raw=[]
        for motor in MOTOR_IDS:
            if not self.reader.isAvailable(motor,132,4):
                raise RuntimeError('Incomplete joint feedback')
            value=self.reader.getData(motor,132,4)
            raw.append(value if value < 2**31 else value-2**32)
        return (np.array(raw)-2048)*2*np.pi/4096

    def goals(self):
        raw=[]
        for motor in MOTOR_IDS:
            value=self.read_register(motor,116,4)
            raw.append(value if value<2**31 else value-2**32)
        return (np.array(raw)-2048)*2*np.pi/4096

    def goal(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError('Invalid six-joint goal')
        raw=np.rint(2048+q*4096/(2*np.pi)).astype(int)
        self.writer.clearParam()
        for motor,value in zip(MOTOR_IDS,raw):
            value=int(value)&0xffffffff
            if not self.writer.addParam(motor,list(value.to_bytes(4,'little'))):
                raise RuntimeError('Could not build joint goal packet')
        if self.writer.txPacket() != self.dxl.COMM_SUCCESS:
            raise RuntimeError('Goal packet failed')

    def close(self):
        self.port.closePort()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--settle-only',action='store_true',
                        help='Bounded feedback correction only; requires already-nearby target')
    args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    if plan.get('execution_blocked'):
        raise RuntimeError(plan.get('execution_blocked_reason','Plan execution is blocked'))
    path=validate_path(plan['trajectory_q_rad'])
    duration=duration_for(path)
    if args.settle_only:
        print('Fixed R3C3 correction only: maximum 15s, correction <=3 deg, slew <=2 deg/s',flush=True)
    else:
        print(f'One hover at {plan["target_square"]}; duration {duration:.1f}s; maximum commanded rate 3 deg/s',flush=True)
    if not args.execute:
        return
    if plan['tcp_offset_m'] != [.16,0,0] or plan['target_square'] != 'R3C3':
        raise RuntimeError('This first-check executor is restricted to the reviewed R3C3/160mm TCP plan')
    if plan['minimum_tip_height_above_estimated_board_m'] < .079:
        raise RuntimeError('Insufficient board standoff')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    arm=camera=None
    stop=threading.Event();done=threading.Event();thread=None
    state={'status':'preparing','fraction':0.,'error':None,'motion_sent':False}
    from .camera import RGBDCamera
    cv2.setNumThreads(1)
    try:
        # Require a working operator preview before starting the movement worker.
        camera=RGBDCamera('234222300865')
        frame,_,_,_=camera.read()
        cv2.imshow('Right arm FIRST HOVER CHECK - Q to stop',frame)
        cv2.waitKey(1)
        arm=RightArm()
        actual=arm.positions()
        initial_goal=arm.goals()
        if args.settle_only:
            if np.max(np.abs(actual-path[-1]))>np.deg2rad(5):
                raise RuntimeError('Settle-only requires being within 5 degrees of the reviewed hover')
        else:
            check_start(actual,path[0])
        def move():
            last_actual=actual.copy();previous_goal=actual.copy();last_fault_check=0.
            try:
                start=time.monotonic();previous=start
                with (root/'motion.jsonl').open('w') as log:
                    state['status']='moving'
                    if args.settle_only:
                        state['status']='bounded feedback settling'
                        controller=BoundedSettler(path[-1],initial_goal)
                        stable_since=None
                        while not stop.is_set() and time.monotonic()-start<15:
                            now=time.monotonic();dt=now-previous
                            if dt<.001:
                                stop.wait(.01);continue
                            previous=now
                            q=arm.positions();last_actual=q
                            for motor in range(1,10):
                                if now-last_fault_check>.5 and arm.read_register(motor,70,1):
                                    raise RuntimeError('Motor hardware fault')
                            if now-last_fault_check>.5:last_fault_check=now
                            goal=controller.step(q,dt)
                            arm.goal(goal);state['motion_sent']=True
                            elapsed=now-start;state['fraction']=min(elapsed/15,1.)
                            log.write(json.dumps({'elapsed_s':elapsed,'q_actual_rad':q.tolist(),
                                                 'q_goal_rad':goal.tolist(),'correction_rad':controller.correction.tolist()})+'\n')
                            if np.max(np.abs(q-path[-1]))<SETTLE_TOLERANCE_RAD:
                                if stable_since is None:stable_since=now
                                if now-stable_since>.75:
                                    state['status']='target reached';state['fraction']=1.;break
                            else:stable_since=None
                            stop.wait(max(0.,1/30-(time.monotonic()-now)))
                        if state['status']!='target reached' and not stop.is_set():
                            raise RuntimeError('Bounded correction did not settle; no further squares commanded')
                        state['final_q_rad']=arm.positions().tolist()
                        state['final_goal_rad']=controller.goal.tolist()
                        return
                    while not stop.is_set():
                        now=time.monotonic();dt=now-previous
                        if dt > .15:
                            raise RuntimeError('Control timing gap; stop motion')
                        previous=now
                        q=arm.positions();last_actual=q
                        if np.max(np.abs(q-previous_goal)) > np.deg2rad(4):
                            raise RuntimeError('Joint tracking error exceeds 4 degrees')
                        if now-last_fault_check > .5:
                            for motor in range(1,10):
                                if arm.read_register(motor,70,1):
                                    raise RuntimeError('Motor hardware fault')
                            last_fault_check=now
                        fraction=min((now-start)/duration,1.)
                        index=fraction*(len(path)-1);i=min(int(index),len(path)-2);u=index-i
                        goal=path[i]*(1-u)+path[i+1]*u
                        # Feedback and packet limits also bound any scheduling anomaly.
                        if np.max(np.abs(goal-previous_goal)) > np.deg2rad(.5):
                            raise RuntimeError('Goal step exceeds half a degree')
                        arm.goal(goal);state['motion_sent']=True;previous_goal=goal
                        state['fraction']=fraction
                        log.write(json.dumps({'elapsed_s':now-start,'q_actual_rad':q.tolist(),
                                             'q_goal_rad':goal.tolist(),'fraction':fraction})+'\n')
                        if fraction >= 1:
                            state['status']='settling';break
                        stop.wait(max(0.,1/30-(time.monotonic()-now)))
                if state['status']=='settling' and not stop.is_set():
                    settle_start=time.monotonic()
                    while not stop.is_set() and time.monotonic()-settle_start<4:
                        settled=arm.positions()
                        if time.monotonic()-settle_start>1 and np.max(np.abs(settled-path[-1]))<SETTLE_TOLERANCE_RAD:
                            state['status']='target reached';break
                        stop.wait(.05)
                    if state['status']=='settling':
                        raise RuntimeError('Arm did not settle within 0.7 degrees of target')
                state['final_q_rad']=arm.positions().tolist()
            except BaseException as exc:
                state['error']=str(exc);state['status']='stopped on error'
            finally:
                # Stop generating setpoints and retain the last bounded motor goal.
                # Replacing a load-compensated goal with measured position would
                # remove the holding effort and let the loaded joints sag again.
                state['hold_behavior']='No new goals; last bounded goal retained. Hardware faults may independently disable torque.'
                done.set()
        thread=threading.Thread(target=move,name='single-right-hover',daemon=True)
        thread.start()
        printed=-1
        while not done.is_set():
            frame,_,_,_=camera.read()
            view=frame.copy()
            cv2.putText(view,f'R3C3 hover | {state["status"]} | {100*state["fraction"]:.0f}%',(8,25),
                        cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            cv2.putText(view,'Q: stop | fingertips stay above board',(8,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.imshow('Right arm FIRST HOVER CHECK - Q to stop',view)
            if cv2.waitKey(1)&0xff==ord('q'):
                stop.set();state['status']='operator stop'
            bucket=int(state['fraction']*10)
            if bucket>printed:
                printed=bucket;print(f'Hover progress {state["fraction"]*100:.0f}%',flush=True)
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError('Motion thread did not finish')
        if state['motion_sent']:
            frame,_,_,_=camera.read();cv2.imwrite(str(root/'after-hover.jpg'),frame)
        print(json.dumps(state,indent=2),flush=True)
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=15)
        if thread is not None and thread.is_alive():
            raise RuntimeError('Control thread still active; do not start another serial owner')
        if arm is not None:arm.close()
        if camera is not None:camera.close()
        cv2.destroyAllWindows()
        (root/'summary.json').write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    main()
