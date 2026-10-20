"""Return a healthy, already-held right ALOHA to a checked compact parking pose.

No torque, gripper, mode, gain, reboot, or fault-clear writes. Retain holding
torque after parking; this is not an unsupported torque-off sleep procedure.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import argparse
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from .aloha_hover_check import check_start, duration_for, validate_path
from .aloha_tape_hover import CurrentGuard, GeometryGuard, TapeArm
from .vx300s_kinematics import LOWER, UPPER


class ParkFollower:
    """Follow the checked path from existing hardware goals, without extra I control.

    Using measured position as the first goal would remove the existing holding
    effort. The nominal path starts at the retained motor goal instead.
    """
    def __init__(self, initial_goal):
        self.goal = np.asarray(initial_goal, float).copy()
        if self.goal.shape != (6,) or not np.isfinite(self.goal).all():
            raise ValueError('Invalid initial holding goal')

    def step(self, nominal, actual, dt):
        nominal, actual = np.asarray(nominal, float), np.asarray(actual, float)
        if (nominal.shape != (6,) or actual.shape != (6,)
                or not np.isfinite([*nominal, *actual, dt]).all() or not 0 < dt <= .15):
            raise ValueError('Invalid parking feedback/timing')
        if np.max(np.abs(nominal-actual)) > np.deg2rad(4):
            raise RuntimeError('Parking following error exceeds 4 degrees')
        if np.any(nominal < LOWER) or np.any(nominal > UPPER):
            raise RuntimeError('Parking goal outside model joint limits')
        self.goal += np.clip(nominal-self.goal, -np.deg2rad(3)*dt, np.deg2rad(3)*dt)
        return self.goal.copy()


def validate_park(plan):
    if (plan.get('schema') != 'right_aloha_park_v1' or plan.get('execution_blocked', True)
            or plan.get('torque_writes_allowed') is not False or plan.get('new_modeled_contacts', True)):
        raise ValueError('Not a reviewed torque-preserving right-arm parking plan')
    path = validate_path(plan['trajectory_q_rad'])
    if np.any(path < LOWER) or np.any(path > UPPER):
        raise ValueError('Parking path leaves model joint limits')
    if (np.max(np.abs(path[0]-plan['initial_goal_q_rad'])) > 1e-9
            or np.max(np.abs(path[-1]-plan['park_q_rad'])) > 1e-9):
        raise ValueError('Parking path endpoints disagree')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text()); path = validate_park(plan)
    duration = duration_for(path)
    print(f'Right-arm parking: {duration:.1f}s, maximum nominal/goal rate 3 deg/s; torque remains ON', flush=True)
    if not args.execute: return
    from .camera import RGBDCamera
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = {'status': 'preparing', 'error': None, 'motion_sent': False,
             'parked': False, 'torque_changed': False, 'torque_after_parking': 'ON',
             'expert_episode': False, 'last_preview_s': time.monotonic()}
    stop, done = threading.Event(), threading.Event()
    camera = arm = worker = last_view = None
    def save():
        (root/'status.tmp.json').write_text(json.dumps(dict(state), indent=2)+'\n')
        (root/'status.tmp.json').replace(root/'status.json')
    try:
        geometry = GeometryGuard(plan['collision_model'], plan['initial_actual_q_rad'])
        for q in path: geometry.check(q, .04)
        camera = RGBDCamera('234222300865')
        frame, _, _, _ = camera.read()
        cv2.imshow('Right ALOHA parking - Q stop', frame); cv2.waitKey(1)
        arm = TapeArm(); arm.check_health()
        actual, currents, _ = arm.telemetry(); held = arm.goals()
        check_start(actual, np.array(plan['initial_actual_q_rad']))
        check_start(held, path[0])
        geometry.check(actual, .025)
        follower, guard = ParkFollower(held), CurrentGuard()
        state['last_preview_s'] = time.monotonic()
        def move():
            start = last = time.monotonic(); last_fault = 0.; stable = None; reached_at = None
            try:
                with (root/'motion.jsonl').open('w', buffering=1) as log:
                    while not stop.is_set():
                        now = time.monotonic(); dt = now-last
                        if dt < .005: stop.wait(.005); continue
                        last = now
                        actual, currents, _ = arm.telemetry()
                        state.update(actual_q_rad=actual.tolist(), currents_a=currents.tolist())
                        guard.check(currents, dt)
                        if now-last_fault > .25: arm.check_health(); last_fault = now
                        if now-state['last_preview_s'] > .35:
                            raise RuntimeError('Parking operator preview stale')
                        geometry.check(actual, .025)
                        fraction = min((now-start)/duration, 1.)
                        index = fraction*(len(path)-1); i = min(int(index), len(path)-2); u = index-i
                        nominal = (1-u)*path[i]+u*path[i+1]
                        goal = follower.step(nominal, actual, dt)
                        geometry.check(goal, .03)
                        arm.goal(goal); state['motion_sent'] = True
                        error = float(np.max(np.abs(actual-path[-1])))
                        state.update(status='parking' if fraction < 1 else 'settling', fraction=fraction,
                                     final_goal_q_rad=goal.tolist(), park_error_deg=float(np.rad2deg(error)))
                        log.write(json.dumps({'host_monotonic_s': now, 'actual_q_rad': actual.tolist(),
                                              'goal_q_rad': goal.tolist(), 'currents_a': currents.tolist(),
                                              'fraction': fraction})+'\n')
                        if fraction >= 1:
                            if error < np.deg2rad(2):
                                if stable is None: stable = now
                                if now-stable > 1:
                                    state['parked'] = True; state['status'] = 'parked / holding torque'
                                    if reached_at is None: reached_at = now
                                    if now-reached_at > 5: break
                            else: stable = None
                            if now-start > duration+15 and not state['parked']:
                                raise RuntimeError('Parking pose did not settle within 2 degrees')
                        stop.wait(max(0., 1/30-(time.monotonic()-now)))
                if state['parked']: print('Right arm parked; holding torque retained.', flush=True)
            except BaseException as exc:
                state['status'] = 'stopped on error'; state['error'] = str(exc)
                print('Parking stopped: '+str(exc), flush=True)
            finally:
                state['hold_behavior'] = 'Last bounded hardware goals retained; no torque or gripper writes.'
                done.set()
        worker = threading.Thread(target=move, daemon=True); worker.start()
        while not done.is_set() and not stop.is_set():
            frame, _, _, _ = camera.read(); state['last_preview_s'] = time.monotonic()
            last_view = frame.copy()
            cv2.putText(last_view, f'RIGHT ARM | {state["status"]} | {100*state.get("fraction", 0):.0f}%',
                        (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            cv2.putText(last_view, 'Q: stop new goals | holding torque remains enabled', (8, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
            cv2.imshow('Right ALOHA parking - Q stop', last_view)
            if cv2.waitKey(1) & 0xff == ord('q'): stop.set()
            save()
            cv2.imwrite(str(root/'live_frame.tmp.jpg'), last_view)
            (root/'live_frame.tmp.jpg').replace(root/'live_frame.jpg')
        if stop.is_set() and not state['error']: state['status'] = 'operator stop'
    except BaseException as exc:
        state['status'] = 'stopped before or during parking'; state['error'] = str(exc)
        raise
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=10)
            if worker.is_alive(): raise RuntimeError('Parking worker still active; do not open another serial owner')
        if arm is not None: arm.close()
        if camera is not None: camera.close()
        cv2.destroyAllWindows()
        if last_view is not None: cv2.imwrite(str(root/'after-park.jpg'), last_view)
        (root/'summary.json').write_text(json.dumps(dict(state), indent=2)+'\n'); save()


if __name__ == '__main__': main()
