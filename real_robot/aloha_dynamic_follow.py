"""Bounded right-arm pilot following an independently running perception feed.

Requires a fresh reviewed approach/scene plan. No motor mode, gain, protection,
reboot or gripper changes. Supported start prepares unchanged holding goals and
enables torque; stopping retains the last bounded holding goals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .aloha_hover_check import check_start, duration_for, validate_path
from .aloha_tape_hover import TapeArm, CurrentGuard, GeometryGuard, HoverServo
from .controller import damped_joint_velocity
from .vx300s_kinematics import forward, LOWER, UPPER
from .geometry import validate_transform
from .tracking_config import validate_control, load_config


def validate_plan(plan):
    if (plan.get('schema') != 'right_aloha_dynamic_pilot_v1' or plan.get('execution_blocked', True)
            or not plan.get('scene_check_passed') or plan.get('arm') != 'right'
            or plan.get('tcp_offset_m') != [.16, 0, 0] or plan.get('hover_height_m') != .05
            or plan.get('base_calibration_validated') is not False
            or plan.get('max_nominal_tcp_speed_mps') != .015
            or plan.get('max_object_speed_mps') != .05):
        raise ValueError('Require a reviewed, bounded 50 mm right-arm pilot plan')
    if plan.get('plan_boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        raise ValueError('Plan is from another computer boot; capture and review a fresh scene')
    age=time.monotonic()-plan.get('scene_observation_host_monotonic_s',-float('inf'))
    if not 0<=age<=300: raise ValueError('Scene review is older than 5 minutes; replan')
    path = validate_path(plan['approach_q_rad'])
    if np.any(path < LOWER) or np.any(path > UPPER): raise ValueError('Pilot requires a fully in-limit starting path')
    if np.max(np.abs(path[0]-plan['initial_q_rad'])) > 1e-8: raise ValueError('Approach start mismatch')
    lo, hi = np.array(plan['object_workspace_min_m']), np.array(plan['object_workspace_max_m'])
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([*lo, *hi]).all() or np.any(hi <= lo) or np.any(hi-lo > [.081, .081, .031]):
        raise ValueError('Invalid pilot workspace')
    model = Path(plan['collision_model'])
    if hashlib.sha256(model.read_bytes()).hexdigest() != plan['collision_model_sha256']:
        raise ValueError('Collision model changed since review')
    target = validate_transform(plan['target_T_base_tcp']); endpoint = forward(path[-1], [.16, 0, 0])
    if not np.allclose(plan['hover_rotation'], target[:3,:3]): raise ValueError('Hover rotation mismatch')
    anchor=np.asarray(plan['initial_object_top_anchor_m'])
    if np.any(anchor<lo) or np.any(anchor>hi): raise ValueError('Initial object outside checked workspace')
    if (np.linalg.norm(target[:3, 3]-endpoint[:3, 3]) > .002
            or np.linalg.norm(cv2.Rodrigues(target[:3, :3]@endpoint[:3, :3].T)[0]) > np.deg2rad(.5)
            or not 0 <= plan['surface_top_offset_m'] <= .08
            or not np.allclose(target[:3, 3], np.array(plan['initial_object_top_anchor_m'])+[0, 0, .05])):
        raise ValueError('Hover endpoint or visible-top offset disagrees with plan')
    computed_floor = min(forward(q, [.16, 0, 0])[2, 3] for q in path)
    if not np.isclose(computed_floor, plan['minimum_tcp_z_m'], atol=1e-6) or computed_floor < .07:
        raise ValueError('TCP floor was not derived from the checked path')
    return path


def observation(path, plan, now):
    try: record = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError): return None, 'perception unavailable'
    if (record.get('schema') != 'front_side_observation_v1' or not record.get('running')
            or record.get('stale', True) or record.get('rig_sha256') != plan['rig_sha256']
            or record.get('output_frame') != 'right_base_estimate'):
        return None, 'perception stale or frame/calibration mismatch'
    timing = record.get('timing') or {}; arrival = timing.get('oldest_host_arrival_monotonic_s', -float('inf'))
    if not 0 <= now-arrival <= .15: return None, 'camera arrival age exceeds 150 ms'
    track = record.get('track')
    if not track or not track.get('accepted') or not track.get('ready') or track.get('frame') != 'right_base_estimate':
        return None, 'position filter unready'
    point = np.asarray(track['position_m'], float)
    if point.shape != (3,) or not np.isfinite(point).all(): return None, 'invalid position'
    point = point.copy(); point[2] += plan['surface_top_offset_m']
    if np.any(point < plan['object_workspace_min_m']) or np.any(point > plan['object_workspace_max_m']):
        lo,hi=np.array(plan['object_workspace_min_m']),np.array(plan['object_workspace_max_m'])
        axis=int(np.argmax(np.maximum(lo-point,point-hi)))
        return None, f"object {'XYZ'[axis]}={point[axis]:.3f} m outside checked [{lo[axis]:.3f}, {hi[axis]:.3f}] m"
    speed = record.get('velocity_report', {}).get('speed_mps')
    if speed is not None and (not np.isfinite(speed) or speed > plan['max_object_speed_mps']):
        return None, f"object exceeds {plan['max_object_speed_mps']*100:g} cm/s pilot cap"
    return {'point': point, 'record': record}, None


class DynamicStep:
    def __init__(self, q, rotation, floor_z, control=None, tcp_bounds=None):
        self.q = np.array(q, float); self.rotation = np.array(rotation, float)
        self.floor_z = floor_z; self.velocity = np.zeros(3)
        self.control=validate_control(control); self.integral=np.zeros(3)
        self.previous_position=forward(self.q,[.16,0,0])[:3,3].copy()
        self.tcp_bounds=tcp_bounds

    def step(self, target, dt):
        if not 0 < dt <= .1: raise RuntimeError('Invalid dynamic control timing')
        target = np.asarray(target, float)
        if target.shape != (3,) or not np.isfinite(target).all(): raise ValueError('Invalid target')
        t = forward(self.q, [.16, 0, 0]); error = target-t[:3, 3]
        c=self.control
        if np.linalg.norm(error) > c['max_lag_m']: raise RuntimeError('Hover lag exceeds configured bound; do not chase')
        self.integral=np.clip(self.integral+error*dt,-c['integral_limit_m_s'],c['integral_limit_m_s'])
        measured_velocity=(t[:3,3]-self.previous_position)/dt
        self.previous_position=t[:3,3].copy()
        requested=c['kp']*error+c['ki']*self.integral-c['kd']*measured_velocity
        requested *= min(1.,c['max_tcp_speed_mps']/max(np.linalg.norm(requested),1e-12))
        delta = requested-self.velocity; self.velocity += delta*min(1., c['max_tcp_acceleration_mps2']*dt/max(np.linalg.norm(delta), 1e-12))
        jacobian = np.empty((6, 6)); eps = 1e-5
        for i in range(6):
            shifted = self.q.copy(); shifted[i] += eps; other = forward(shifted, [.16, 0, 0])
            jacobian[:3, i] = (other[:3, 3]-t[:3, 3])/eps
            skew = ((other[:3, :3]-t[:3, :3])/eps)@t[:3, :3].T
            jacobian[3:, i] = [skew[2, 1], skew[0, 2], skew[1, 0]]
        omega = c['orientation_kp']*cv2.Rodrigues(self.rotation@t[:3, :3].T)[0].ravel()
        qdot = damped_joint_velocity(jacobian, np.r_[self.velocity, omega], c['damping'], np.deg2rad(c['max_joint_speed_deg_s']))
        tcp_speed=np.linalg.norm(jacobian[:3]@qdot)
        qdot*=min(1.,c['max_tcp_speed_mps']/max(tcp_speed,1e-12))
        proposed = self.q+dt*qdot
        pose = forward(proposed, [.16, 0, 0])
        if np.any(proposed < LOWER) or np.any(proposed > UPPER) or pose[2, 3] < self.floor_z:
            raise RuntimeError('Dynamic candidate violates joint or TCP floor bounds')
        if self.tcp_bounds is not None:
            lo,hi=self.tcp_bounds
            if np.any(pose[:3,3]<lo) or np.any(pose[:3,3]>hi): raise RuntimeError('TCP left checked dynamic workspace')
        self.q = proposed
        return proposed.copy()


def approach_duration(path,control):
    joint_duration=duration_for(path,np.deg2rad(control['max_joint_speed_deg_s']))
    xyz=np.array([forward(q,[.16,0,0])[:3,3] for q in path])
    tcp_duration=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).max()*(len(path)-1)/control['max_tcp_speed_mps'])
    return max(joint_duration,tcp_duration)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--execute-supported-start', action='store_true')
    parser.add_argument('--execute-held-start', action='store_true')
    parser.add_argument('--config', help='YAML software tracking control gains and bounded speeds')
    parser.add_argument('--seconds', type=float, default=120, help='Dynamic phase after initial hover settles')
    args = parser.parse_args(argv)
    control=validate_control() if args.config is None else load_config(args.config)['control']
    if not 0 < args.seconds <= 300: raise ValueError('Pilot duration must be <=300 seconds')
    plan = json.loads(Path(args.plan).read_text()); path = validate_plan(plan)
    plan['max_object_speed_mps']=min(plan['max_object_speed_mps'],control['max_object_speed_mps'])
    duration = approach_duration(path,control)
    if args.execute_supported_start and args.execute_held_start: raise ValueError('Choose one starting mode')
    if not args.execute_supported_start and not args.execute_held_start:
        print(json.dumps({'plan_valid': True, 'approach_duration_s': duration, 'motion_sent': False})); return
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (root/'control.json').write_text(json.dumps(control,indent=2)+'\n')
    state = {'status': 'preflight', 'motion_sent': False, 'error': None, 'base_calibration_validated': False,
             'expert_episode': False, 'torque_enabled_by_this_run': False}
    arm = None; last_view = None
    def save():
        (root/'status.tmp.json').write_text(json.dumps(state, indent=2)+'\n'); (root/'status.tmp.json').replace(root/'status.json')
    try:
        obs, reason = observation(plan['observation_file'], plan, time.monotonic())
        if obs is None: raise RuntimeError(reason)
        if np.linalg.norm(obs['point']-plan['initial_object_top_anchor_m']) > .012:
            raise RuntimeError('Object moved since planning')
        geometry = GeometryGuard(plan['collision_model'], path[0])
        for sample in path: geometry.check(sample, .03)
        arm = TapeArm(); actual, _, _ = arm.telemetry(); check_start(actual, path[0])
        for i in range(1, 10):
            if arm.read_register(i, 146, 1) >= 55: raise RuntimeError('Motor already warm')
        if args.execute_supported_start:
            actual = arm.enable_supported_hold(path[0]); state['torque_enabled_by_this_run'] = True
            initial_goal = actual
        else:
            arm.check_health(); initial_goal = arm.goals()
        # Bias is relative to the retained nominal reference, not measured
        # gravity sag. check_start already bounds actual-reference to 0.5deg.
        servo = HoverServo(path[0], plan['minimum_tcp_z_m']-.012, initial_goal,
                           max_goal_rate_deg_s=control['max_joint_speed_deg_s'])
        guard = CurrentGuard(); nominal = path[0].copy()
        dynamic = None; elapsed = 0.; stable = dynamic_start = lost_since = None
        start = last = time.monotonic(); last_fault = 0.
        print('Holding torque verified. Keep tape still during approach.', flush=True)
        with (root/'motion.jsonl').open('w', buffering=1) as log:
            while True:
                now = time.monotonic(); dt = now-last
                if dt < 1/30: time.sleep(max(0., 1/30-dt)); continue
                last = now
                if dt > .15: raise RuntimeError('Control loop exceeded 150 ms')
                actual, currents, _ = arm.telemetry(); guard.check(currents, dt)
                geometry.check(actual, .02)
                if now-last_fault > .25: arm.check_health(); last_fault = now
                obs, reason = observation(plan['observation_file'], plan, now)
                if obs is None:
                    if lost_since is None: lost_since = now
                    state['status'] = 'holding: '+str(reason)
                    if dynamic is not None:
                        dynamic.velocity[:] = 0; dynamic.integral[:] = 0
                    if now-lost_since > 2: raise RuntimeError(reason+' for over 2 seconds')
                    goal = servo.goal.copy()
                else:
                    lost_since = None
                    point = obs['point']; state['object_top_anchor_m'] = point.tolist()
                    state['velocity_report'] = obs['record']['velocity_report']
                    if dynamic is None:
                        if np.linalg.norm(point-plan['initial_object_top_anchor_m']) > .015:
                            raise RuntimeError('Tape moved during initial static approach')
                        if now-start >= 2: elapsed = min(duration, elapsed+dt)
                        index = elapsed/duration*(len(path)-1); i = min(int(index), len(path)-2); u = index-i
                        nominal = path[i]*(1-u)+path[i+1]*u
                        state['status'] = 'approach: keep tape still' if elapsed < duration else 'settling hover'
                        if elapsed >= duration and np.max(np.abs(actual-path[-1])) < np.deg2rad(.7):
                            if stable is None: stable = now
                            if now-stable > 1:
                                bounds=(np.array(plan['object_workspace_min_m'])+[0,0,.05]-.002,
                                        np.array(plan['object_workspace_max_m'])+[0,0,.05]+.002)
                                dynamic = DynamicStep(path[-1], plan['hover_rotation'], plan['minimum_tcp_z_m']-.012,control,bounds)
                                dynamic_start = now; state['hover_reached'] = True
                                print('Hover reached. Small slow tape movements may begin within the checked workspace.', flush=True)
                        else: stable = None
                        if elapsed >= duration and now-start > duration+25: raise RuntimeError('Initial hover failed to settle')
                    else:
                        target = point+[0, 0, .05]
                        if np.linalg.norm(target-forward(actual,[.16,0,0])[:3,3])>control['max_lag_m']:
                            raise RuntimeError('Measured TCP hover lag exceeds configured bound')
                        nominal = dynamic.step(target, dt)
                        state['status'] = f"following: max nominal TCP {control['max_tcp_speed_mps']*100:g} cm/s"
                    geometry.check(nominal, .03)
                    goal = servo.step(nominal, actual, dt)
                geometry.check(goal, .025); arm.goal(goal); state['motion_sent'] = True
                state.update(actual_q_rad=actual.tolist(), nominal_q_rad=nominal.tolist(), goal_q_rad=goal.tolist(),
                    currents_a=currents.tolist(), model_tcp_m=forward(actual, [.16, 0, 0])[:3, 3].tolist(),
                    approach_fraction=elapsed/duration, host_monotonic_s=now)
                log.write(json.dumps(state)+'\n'); save()
                view = np.full((240, 1000, 3), 24, np.uint8)
                labels = ['RIGHT ARM | '+state['status'], 'Q: stop new goals; holding torque remains ON',
                    f"Approach {100*elapsed/duration:.0f}% | object cap {control['max_object_speed_mps']*100:g} cm/s | TCP cap {control['max_tcp_speed_mps']*100:g} cm/s",
                    '50 mm above estimated visible top | provisional base/TCP calibration',
                    'Object workspace: '+str(plan['object_workspace_min_m'])+' to '+str(plan['object_workspace_max_m'])]
                for i,label in enumerate(labels): cv2.putText(view,label,(8,30+40*i),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1)
                cv2.imshow('Right arm bounded moving-object pilot - Q stop',view); last_view = view
                if cv2.waitKey(1)&255 == ord('q') or (root/'STOP').exists(): state['status'] = 'operator stopped / holding'; break
                if dynamic_start is not None and now-dynamic_start >= args.seconds:
                    state['status'] = 'pilot complete / holding'; break
                if now-start > duration+args.seconds+45: raise RuntimeError('Pilot wall-time limit')
    except BaseException as exc:
        state.update(status='stopped / last holding goals retained', error=f'{type(exc).__name__}: {exc}')
        print(state['error'], flush=True)
    finally:
        if arm is not None: arm.close()
        cv2.destroyAllWindows()
        state['running'] = False; state['hold_behavior'] = 'Last goals retained, torque not automatically released.'
        save(); (root/'summary.json').write_text(json.dumps(state, indent=2)+'\n')
        if last_view is not None: cv2.imwrite(str(root/'last_frame.jpg'), last_view)
    return 1 if state['error'] else 0


if __name__ == '__main__': raise SystemExit(main())
