"""Supervised single blue-tape hover; explicit supported-start torque activation.

Not a dynamic-following controller. Requires an independently checked path and
fresh matching tape/arm observations. Never reboots or changes modes/gains/limits.
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

from .aloha_hover_check import RightArm, check_start, duration_for, validate_path
from .geometry import transform_point, validate_transform
from .vx300s_kinematics import forward, LOWER, UPPER, MOTOR_IDS


def signed(value, bits):
    return value - (1 << bits) if value >= (1 << (bits-1)) else value


def observation_transform(plan, q):
    if plan.get('observation_camera', 'front') == 'right_wrist':
        return forward(q)@validate_transform(plan['T_gripper_wrist_camera'])
    return validate_transform(plan['T_base_camera_candidate'])


class CurrentGuard:
    """Conservative software abort thresholds, not certified hardware limits.

    Both model 1120 and 1020 report Present Current in 2.69 mA/count.
    The motor's own shutdown protection remains unchanged.
    """
    def __init__(self, profile='pilot'):
        self.high_s = np.zeros(9)
        if profile not in ('pilot','continuous'):raise ValueError('Unknown software current profile')
        self.profile=profile

    def check(self, currents_a, dt):
        currents = np.abs(np.asarray(currents_a, float))
        if currents.shape != (9,) or not np.isfinite(currents).all() or not 0 < dt <= .15:
            raise ValueError('Invalid current feedback or control timing')
        peak = np.array([1.5]*7+[.9, .8])
        sustained = np.array([.9]*7+[.5, .6])
        # October 20: healthy extended right-arm hold measured 0.95/1.15 A
        # at the paired shoulder, 37 C, no faults; hardware limit 5.506 A.
        # The continuous profile changes only these software sustained guards.
        # Peak thresholds and every motor register/protection stay unchanged.
        if self.profile=='continuous':sustained[1:3]=1.25
        if np.any(currents > peak):
            raise RuntimeError(f'Motor {int(np.argmax(currents/peak))+1}: peak current guard')
        self.high_s = np.where(currents > sustained, self.high_s+dt, 0.)
        if np.any(self.high_s >= .75):
            raise RuntimeError(f'Motor {int(np.argmax(self.high_s))+1}: sustained current guard')


class HoverServo:
    def __init__(self, initial, floor_z, initial_goal=None, max_goal_rate_deg_s=3.5):
        initial = np.asarray(initial, float)
        self.goal = initial.copy() if initial_goal is None else np.asarray(initial_goal, float).copy()
        if self.goal.shape != (6,) or not np.isfinite(self.goal).all():
            raise ValueError('Invalid initial holding goal')
        self.correction = self.goal-initial
        # Hardware goals have one encoder count of quantization. Preserve the
        # exact retained goal while clipping only the internal I state.
        if np.max(np.abs(self.correction)) > np.deg2rad(1.5)+2*np.pi/4096+1e-9:
            raise ValueError('Existing holding correction exceeds 1.5 degrees')
        self.correction=np.clip(self.correction,-np.deg2rad(1.5),np.deg2rad(1.5))
        self.floor_z = float(floor_z)
        if not np.isfinite(max_goal_rate_deg_s) or not 0<max_goal_rate_deg_s<=3.5:
            raise ValueError('Invalid bounded holding goal rate')
        self.max_goal_rate_rad_s=np.deg2rad(max_goal_rate_deg_s)
        self.lo = np.minimum(LOWER, initial)
        self.hi = np.maximum(UPPER, initial)

    def step(self, nominal, actual, dt):
        nominal, actual = np.asarray(nominal, float), np.asarray(actual, float)
        if (nominal.shape != (6,) or actual.shape != (6,)
                or not np.isfinite([*nominal, *actual, dt]).all() or not 0 < dt <= .15):
            raise ValueError('Invalid servo feedback/timing')
        if np.max(np.abs(nominal-actual)) > np.deg2rad(4):
            raise RuntimeError('Joint following error exceeds 4 degrees')
        if forward(actual, [.16, 0, 0])[2, 3] < self.floor_z:
            raise RuntimeError('Measured/model TCP below planned clearance floor')
        self.correction = np.clip(self.correction+.35*dt*(nominal-actual),
                                  -np.deg2rad(1.5), np.deg2rad(1.5))
        requested = np.clip(nominal+self.correction, LOWER, UPPER)
        goal = self.goal+np.clip(requested-self.goal, -self.max_goal_rate_rad_s*dt, self.max_goal_rate_rad_s*dt)
        if (np.any(goal < self.lo-1e-9) or np.any(goal > self.hi+1e-9)
                or forward(goal, [.16, 0, 0])[2, 3] < self.floor_z):
            raise RuntimeError('Proposed goal exceeds planned clearance or joint envelope')
        # Existing parked-limit overruns may only shrink, never increase.
        self.lo, self.hi = np.maximum(self.lo, np.minimum(LOWER, goal)), np.minimum(self.hi, np.maximum(UPPER, goal))
        self.goal = goal
        return goal.copy()


class GeometryGuard:
    """Check the same mesh model as the offline path, including camera housing.

    Initial convex-mesh overlaps may only diminish (2 mm numerical allowance).
    This is not a complete certificate for the physical scene or flexible cables.
    """
    def __init__(self, model_path, initial):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.baseline = {}
        self._pose(initial)
        for c in self.data.contact:
            if c.dist < -.001:
                self.baseline[tuple(sorted((c.geom1, c.geom2)))] = float(c.dist)

    def _pose(self, q):
        self.data.qpos[:6] = q
        self.data.qpos[6:] = [.021, -.021]
        self.mj.mj_forward(self.model, self.data)

    def check(self, q, minimum_table_clearance=.02):
        self._pose(q)
        m, d = self.model, self.data
        for c in d.contact:
            if c.dist >= -.001: continue
            pair = tuple(sorted((c.geom1, c.geom2)))
            if pair not in self.baseline or c.dist < self.baseline[pair]-.002:
                raise RuntimeError('New or worsening modeled self contact')
        for gid in range(m.ngeom):
            name = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_GEOM, gid) or ''
            if name == 'table' or '/1_base' in name or '/2_shoulder' in name: continue
            rotation = d.geom_xmat[gid].reshape(3, 3)
            if m.geom_type[gid] == self.mj.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gid]; start = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
                z = (m.mesh_vert[start:start+n]@rotation.T+d.geom_xpos[gid])[:, 2].min()
            elif m.geom_type[gid] == self.mj.mjtGeom.mjGEOM_BOX:
                z = d.geom_xpos[gid, 2]-np.abs(rotation[2])@m.geom_size[gid]
            else: continue
            if z < minimum_table_clearance:
                raise RuntimeError('Modeled geometry/table clearance guard')


class TapeArm(RightArm):
    def __init__(self):
        super().__init__(require_torque=False)
        try:
            self.telemetry_reader = self.dxl.GroupSyncRead(self.port, self.packet, 126, 10)
            for motor in range(1, 10):
                if not self.telemetry_reader.addParam(motor):
                    raise RuntimeError('Cannot prepare telemetry reader')
                if self.read_register(motor, 10, 1) & 8:
                    raise RuntimeError('Unexpected torque-on-goal-update drive mode')
            if self.read_register(9, 11, 1) != 5:
                raise RuntimeError('Unexpected gripper operating mode')
        except BaseException:
            self.close()
            raise

    def telemetry(self):
        r = self.telemetry_reader
        if r.txRxPacket() != self.dxl.COMM_SUCCESS:
            raise RuntimeError('Joint/current telemetry communication failed')
        raw, currents = {}, []
        for motor in range(1, 10):
            if not r.isAvailable(motor, 126, 10):
                raise RuntimeError('Incomplete motor telemetry')
            raw[motor] = signed(r.getData(motor, 132, 4), 32)
            currents.append(signed(r.getData(motor, 126, 2), 16)*.00269)
        q = (np.array([raw[i] for i in MOTOR_IDS])-2048)*2*np.pi/4096
        return q, np.array(currents), raw

    def check_health(self):
        for motor in range(1, 10):
            if self.read_register(motor, 70, 1) != 0:
                raise RuntimeError('Motor hardware fault')
            if self.read_register(motor, 64, 1) != 1:
                raise RuntimeError(f'Motor {motor} torque unexpectedly off')
            if self.read_register(motor, 146, 1) >= 55:
                raise RuntimeError(f'Motor {motor} temperature guard')

    def enable_supported_hold(self, expected):
        # No motion setpoint until the whole supported arm is verified torque-off.
        for motor in range(1, 10):
            if self.read_register(motor, 64, 1) != 0:
                raise RuntimeError('Supported-start requires all nine motors torque-off')
        q, currents, raw = self.telemetry()
        check_start(q, expected)
        for shadow, master in [(3, 2), (5, 4)]:
            if abs(raw[shadow]-raw[master]) > 12:
                raise RuntimeError('Paired encoders differ by over 12 ticks')
        self.goal(q)  # Six master goals; existing Secondary_ID mirrors paired goals.
        comm, error = self.packet.write4ByteTxRx(self.port, 9, 116, raw[9] & 0xffffffff)
        if comm != self.dxl.COMM_SUCCESS or error:
            raise RuntimeError('Cannot prepare unchanged gripper holding goal')
        # Observed torque-off behavior on this arm: Goal Position continuously
        # follows each motor's own encoder (including one-count jitter), instead
        # of retaining the supplied master goal. Verify a local holding target,
        # not equality to the other motor of a paired joint. The full evidence is
        # retained in blue-hover-20261019-04/holding-goal-probe.json.
        self.starting_hold_readback = {}
        for motor in range(1, 10):
            goal = signed(self.read_register(motor, 116, 4), 32)
            position = signed(self.read_register(motor, 132, 4), 32)
            self.starting_hold_readback[str(motor)] = {'goal': goal, 'position': position}
            if abs(goal-position) > 2:
                raise RuntimeError(f'Motor {motor}: holding goal {goal} differs from local encoder {position}')
        self.torque_packet(1)
        self.check_health()
        held = self.positions()
        check_start(held, q)
        return held

    def torque_packet(self, value):
        writer = self.dxl.GroupSyncWrite(self.port, self.packet, 64, 1)
        for motor in range(1, 10):
            if not writer.addParam(motor, [value]):
                raise RuntimeError('Cannot build torque packet')
        if writer.txPacket() != self.dxl.COMM_SUCCESS:
            raise RuntimeError('Torque packet communication failed')
        writer.clearParam()


def validate_plan(plan):
    if plan.get('execution_blocked', True):
        raise ValueError(plan.get('blocked_reason', 'Plan has not passed offline review'))
    if (plan.get('schema') != 'right_aloha_single_tape_hover_v1'
            or plan.get('tcp_offset_m') != [.16, 0, 0] or plan.get('hover_height_m') != .05
            or plan.get('camera_serial') != '234222300865'
            or plan.get('new_modeled_contacts', True)
            or plan.get('unresolved_scene_obstacles', True)
            or plan.get('minimum_modeled_geometry_table_clearance_m', 0) < .03):
        raise ValueError('Not a checked 50 mm right-arm tape hover plan')
    path = validate_path(plan['trajectory_q_rad'])
    target = validate_transform(plan['target_T_base_tcp'])
    validate_transform(plan['T_base_camera_candidate'])
    observation_camera = plan.get('observation_camera', 'front')
    if observation_camera not in ('front', 'right_wrist'):
        raise ValueError('Unsupported observation camera')
    if observation_camera == 'right_wrist':
        if plan.get('wrist_camera_serial') != '218622278247':
            raise ValueError('Require the calibrated right wrist camera')
        validate_transform(plan['T_gripper_wrist_camera'])
    if not np.isfinite(plan['minimum_model_tcp_z_m']):
        raise ValueError('Invalid TCP floor')
    p = np.asarray(plan['object_position_m'], float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('Invalid object position')
    if np.linalg.norm(target[:3, 3]-p-[0, 0, .05]) > 1e-6:
        raise ValueError('Target is not 50 mm along base Z above object')
    if np.linalg.norm(forward(path[-1], [.16, 0, 0])[:3, 3]-target[:3, 3]) > .001:
        raise ValueError('Path endpoint does not match hover target')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--execute-supported-start', action='store_true')
    execution.add_argument('--execute-held-start', action='store_true', help='Continue from a healthy torque-on hold without toggling torque')
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    path = validate_plan(plan)
    duration = duration_for(path)
    print(f'Single tape hover: {duration:.1f}s nominal approach, 3 deg/s maximum nominal rate', flush=True)
    if not (args.execute_supported_start or args.execute_held_start):
        return
    from .camera import RGBDCamera
    from .perception import detect_mask, depth_measurement
    from .tracking import VelocityTracker
    root = Path(args.output); root.mkdir(parents=True, exist_ok=False)
    (root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    state = {'status': 'preparing', 'error': None, 'motion_sent': False, 'torque_enabled_by_this_run': False,
             'calibration_validated': False, 'expert_episode': False, 'reached': False,
             'actual_q_rad': path[0].tolist()}
    observation = {'host_s': -float('inf'), 'point': None}
    stop, done = threading.Event(), threading.Event()
    camera = arm = worker = None
    transform = validate_transform(plan['T_base_camera_candidate'])
    wrist = plan.get('observation_camera', 'front') == 'right_wrist'
    wrist_transform = validate_transform(plan['T_gripper_wrist_camera']) if wrist else None
    observation_log = (root/'perception.jsonl').open('w', buffering=1)
    tracker = VelocityTracker(frame='right_base_estimate', stationary_updates=False)
    clock = last_stamp = None
    origin = None
    last_view = None
    def save_status():
        (root/'status.tmp.json').write_text(json.dumps(dict(state), indent=2)+'\n')
        (root/'status.tmp.json').replace(root/'status.json')
    def observe():
        nonlocal clock, last_stamp, origin, last_view
        bgr, depth, timing, deproject = camera.read()
        stamp = timing['sensor_timestamp_s']
        if ((clock is not None and clock != timing['timestamp_domain'])
                or (last_stamp is not None and stamp <= last_stamp)):
            raise RuntimeError('Camera clock changed or repeated')
        clock, last_stamp = timing['timestamp_domain'], stamp
        if origin is None: origin = stamp
        mask = detect_mask(bgr, 'blue_tape', hsv_low=plan.get('hsv_low'), hsv_high=plan.get('hsv_high'))
        point, quality = depth_measurement(mask, depth, deproject, min_depth_m=.07 if wrist else .1)
        q_for_camera = np.array(state['actual_q_rad'])
        current_transform = observation_transform(plan, q_for_camera)
        if point is not None: point = transform_point(current_transform, point)
        track = tracker.update(stamp-origin, point)
        valid = (track is not None and track.ready and track.accepted
                 and time.monotonic()-timing['host_arrival_monotonic_s'] < .12)
        # Replace the whole observation so the worker never reads half an update.
        observation.update(host_s=time.monotonic(), point=track.position_m.copy() if valid else None)
        observation_log.write(json.dumps({'timing': timing, 'quality': quality,
                                          'q_for_camera_rad': q_for_camera.tolist(),
                                          'base_point_m': None if point is None else point.tolist(),
                                          'track': None if track is None else track.as_dict()})+'\n')
        last_view = bgr.copy()
        if mask is not None:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(last_view, contours, -1, (0, 255, 0), 2)
        cv2.putText(last_view, 'RIGHT ARM | '+state['status'], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
        cv2.putText(last_view, 'Q: stop new goals | 5 cm nominal hover | provisional calibration', (8, 50), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        cv2.imshow('Right ALOHA single tape hover - Q stop', last_view)
        if cv2.waitKey(1) & 0xff == ord('q'): stop.set()
        return valid
    try:
        camera = RGBDCamera('218622278247' if wrist else '234222300865')
        for _ in range(25):
            if stop.is_set(): raise RuntimeError('Operator stopped before activation')
            observe()
        if observation['point'] is None or np.linalg.norm(observation['point']-plan['object_position_m']) > .012:
            raise RuntimeError('Fresh tape position no longer matches plan')
        geometry = GeometryGuard(plan['collision_model'], path[0])
        for q_sample in path: geometry.check(q_sample, .03)
        arm = TapeArm()
        if args.execute_supported_start:
            # Enabling holding torque is restricted to this explicitly supported start.
            state['torque_activation_attempted'] = True
            q = arm.enable_supported_hold(path[0]); state['torque_enabled_by_this_run'] = True
            initial_goal = q
            (root/'starting-hold-readback.json').write_text(json.dumps(arm.starting_hold_readback, indent=2)+'\n')
        else:
            arm.check_health()
            q = arm.positions(); check_start(q, path[0])
            initial_goal = arm.goals()
        floor_z = float(plan['minimum_model_tcp_z_m'])-.012
        servo, current_guard = HoverServo(q, floor_z, initial_goal), CurrentGuard()
        state['status'] = 'holding check'
        save_status()
        def move():
            last = start = time.monotonic(); fault_time = 0.; elapsed = 0.; stable = None
            lost_since = None
            try:
                with (root/'motion.jsonl').open('w', buffering=1) as log:
                    while not stop.is_set():
                        now = time.monotonic(); dt = now-last
                        if dt < .005: stop.wait(.005); continue
                        last = now
                        actual, currents, raw = arm.telemetry()
                        state.update(currents_a=currents.tolist(), actual_q_rad=actual.tolist())
                        current_guard.check(currents, dt)
                        geometry.check(actual)
                        if now-fault_time > .25:
                            arm.check_health(); fault_time = now
                        if now-observation['host_s'] > .35:
                            raise RuntimeError('Camera preview stale')
                        point = observation['point']
                        valid = point is not None
                        if valid and np.linalg.norm(point-plan['object_position_m']) > .015:
                            raise RuntimeError('Tape moved from the checked static target; stop and replan')
                        if valid: lost_since = None
                        elif lost_since is None: lost_since = now
                        elif now-lost_since > 2:
                            raise RuntimeError('Tape lost for over two seconds')
                        if now-start < 2:
                            # Supported start: validate current before leaving the rest.
                            nominal = path[0]
                        elif valid:
                            elapsed = min(duration, elapsed+dt)
                            index = elapsed/duration*(len(path)-1)
                            i = min(int(index), len(path)-2); u = index-i
                            nominal = path[i]*(1-u)+path[i+1]*u
                            state['status'] = 'approaching' if elapsed < duration else 'settling'
                        goal = servo.step(nominal, actual, dt) if valid else servo.goal
                        geometry.check(goal, .025)
                        arm.goal(goal); state['motion_sent'] = True
                        tcp = forward(actual, [.16, 0, 0])[:3, 3]
                        err = float(np.max(np.abs(actual-path[-1])))
                        state.update(fraction=elapsed/duration, currents_a=currents.tolist(),
                                     model_tcp_m=tcp.tolist(), max_joint_error_deg=float(np.rad2deg(err)))
                        if point is not None:
                            state['observed_tcp_minus_object_m'] = (tcp-point).tolist()
                        log.write(json.dumps({'host_s': now, 'actual_q_rad': actual.tolist(),
                                              'nominal_q_rad': nominal.tolist(), 'goal_q_rad': goal.tolist(),
                                              'currents_a': currents.tolist(), 'phase': state['status'],
                                              'model_tcp_m': tcp.tolist()})+'\n')
                        if elapsed >= duration:
                            close = (err < np.deg2rad(.7)
                                     and np.linalg.norm(tcp-np.array(plan['target_T_base_tcp'])[:3, 3]) < .008
                                     and valid and abs((tcp-point)[2]-.05) < .012)
                            if close:
                                if stable is None: stable = now
                                if now-stable > .75:
                                    state['status'] = 'hover reached / monitored hold'
                                    if not state['reached']:
                                        state['reached'] = True
                                        (root/'reached.json').write_text(json.dumps(dict(state), indent=2)+'\n')
                                        print('Hover reached; maintaining monitored hold.', flush=True)
                            else: stable = None
                            if now-start > duration+18 and not state['reached']:
                                raise RuntimeError('Hover did not settle within bounded correction')
                        if now-start > duration+300:
                            state['status'] = 'finished monitored hold'; break
                        stop.wait(max(0., 1/30-(time.monotonic()-now)))
            except BaseException as exc:
                state['error'] = str(exc); state['status'] = 'stopped on error'
                print('STOPPED: '+str(exc), flush=True)
            finally:
                state['hold_behavior'] = 'No further goals; last bounded motor goals retained. Hardware faults can independently disable torque.'
                done.set()
        worker = threading.Thread(target=move, daemon=True); worker.start()
        while not done.is_set() and not stop.is_set():
            observe(); save_status()
            if last_view is not None:
                cv2.imwrite(str(root/'live_frame.tmp.jpg'), last_view)
                (root/'live_frame.tmp.jpg').replace(root/'live_frame.jpg')
        if stop.is_set() and not state['error']: state['status'] = 'operator stop'
    except BaseException as exc:
        state['error'] = str(exc); state['status'] = 'stopped before or during activation'
        raise
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=10)
            if worker.is_alive(): raise RuntimeError('Worker still active; do not open another serial owner')
        if arm is not None: arm.close()
        if camera is not None: camera.close()
        observation_log.close()
        cv2.destroyAllWindows()
        if last_view is not None: cv2.imwrite(str(root/'last_frame.jpg'), last_view)
        (root/'summary.json').write_text(json.dumps(dict(state), indent=2)+'\n')
        save_status()


if __name__ == '__main__':
    main()
