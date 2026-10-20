import json
import hashlib
import time
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from real_robot.aloha_dynamic_follow import DynamicStep, observation, validate_plan, approach_duration
from real_robot.vx300s_kinematics import forward
from real_robot.velocity_report import velocity_report
from real_robot.tracking_config import validate_control, load_config
from real_robot.aloha_tape_hover import HoverServo


class DynamicTests(unittest.TestCase):
    def test_warm_handoff_retains_load_compensation_with_encoder_quantization(self):
        reference=np.deg2rad([0,3,10,0,76,0]);actual=reference.copy();actual[1]+=np.deg2rad(.4)
        held=reference.copy();held[1]-=np.deg2rad(1.5)+.7*2*np.pi/4096
        servo=HoverServo(reference,.07,held,max_goal_rate_deg_s=3)
        np.testing.assert_array_equal(servo.goal,held)
        next_goal=servo.step(reference,actual,1/30)
        self.assertLessEqual(np.max(np.abs(next_goal-held)),np.deg2rad(3)/30+1e-10)
        with self.assertRaises(ValueError):HoverServo(reference,.07,reference+np.deg2rad(2))

    def test_approach_duration_obeys_configured_joint_and_tcp_rates(self):
        q=np.deg2rad([0,20,-10,0,80,0]); end=q+np.deg2rad([5,2,0,0,-2,0])
        u=np.linspace(0,1,101)[:,None];path=q+(10*u**3-15*u**4+6*u**5)*(end-q)
        c=validate_control({'max_joint_speed_deg_s':.5,'max_tcp_speed_mps':.002})
        duration=approach_duration(path,c);dt=duration/(len(path)-1)
        self.assertLessEqual(np.abs(np.diff(path,axis=0)).max()/dt,np.deg2rad(.5)+1e-10)
        xyz=np.array([forward(qq,[.16,0,0])[:3,3] for qq in path])
        self.assertLessEqual(np.linalg.norm(np.diff(xyz,axis=0),axis=1).max()/dt,.002+1e-10)

    def test_plan_rejects_changed_boot_old_scene_and_inconsistent_rotation(self):
        q=np.deg2rad([0,20,-10,0,80,0]); target=forward(q,[.16,0,0]); anchor=target[:3,3]-[0,0,.05]
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'model.xml';model.write_text('offline validation fixture')
            plan=dict(schema='right_aloha_dynamic_pilot_v1',execution_blocked=False,scene_check_passed=True,
                arm='right',tcp_offset_m=[.16,0,0],hover_height_m=.05,base_calibration_validated=False,
                max_nominal_tcp_speed_mps=.015,max_object_speed_mps=.05,
                plan_boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                scene_observation_host_monotonic_s=time.monotonic(),approach_q_rad=[q.tolist(),q.tolist()],
                initial_q_rad=q.tolist(),object_workspace_min_m=(anchor-.01).tolist(),
                object_workspace_max_m=(anchor+.01).tolist(),collision_model=str(model),
                collision_model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                target_T_base_tcp=target.tolist(),hover_rotation=target[:3,:3].tolist(),
                surface_top_offset_m=.03,initial_object_top_anchor_m=anchor.tolist(),minimum_tcp_z_m=target[2,3])
            validate_plan(plan)
            for bad in ({'plan_boot_id':'another boot'},
                        {'scene_observation_host_monotonic_s':time.monotonic()-301},
                        {'hover_rotation':np.eye(3).tolist()}, {'execution_blocked':True}):
                with self.assertRaises(ValueError): validate_plan({**plan,**bad})

    def test_configuration_rejects_unsafe_or_misspelled_values(self):
        for values in ({'max_tcp_speed_mps':.2},{'max_object_speed_mps':.8},
                       {'kp':float('nan')},{'ki':-1},{'damping':0},{'max_speed':.01}):
            with self.assertRaises(ValueError): validate_control(values)
        config=load_config(Path(__file__).parents[1]/'config.yaml')
        self.assertEqual(config['control']['max_tcp_speed_mps'],.015)

    def test_pid_speed_bound_and_workspace_guard(self):
        q=np.deg2rad([0,20,-10,0,80,0]); pose=forward(q,[.16,0,0]); p=pose[:3,3]
        control={'ki':.2,'kd':.1,'max_tcp_speed_mps':.005}
        stepper=DynamicStep(q,pose[:3,:3],.03,control,(p-.02,p+.02))
        previous=p.copy()
        for _ in range(60):
            candidate=stepper.step(p+[.01,0,0],1/30); actual=forward(candidate,[.16,0,0])[:3,3]
            self.assertLessEqual(np.linalg.norm(actual-previous)*30,.00501)
            previous=actual
        blocked=DynamicStep(q,pose[:3,:3],.03,control,(p-1e-10,p+1e-10))
        with self.assertRaises(RuntimeError): blocked.step(p+[.01,0,0],1/30)

    def test_local_tracking_retains_joint_slew_and_reduces_error(self):
        q = np.deg2rad([0, 20, -10, 0, 80, 0]); pose = forward(q, [.16, 0, 0])
        target = pose[:3, 3]+[.008, .008, 0]
        stepper = DynamicStep(q, pose[:3, :3], .03); previous = q.copy()
        initial = np.linalg.norm(target-pose[:3, 3])
        for _ in range(120):
            candidate = stepper.step(target, 1/30)
            self.assertLessEqual(np.max(np.abs(candidate-previous)), np.deg2rad(3)/30+1e-9)
            previous = candidate
        self.assertLess(np.linalg.norm(target-forward(previous, [.16, 0, 0])[:3, 3]), initial/3)
        with self.assertRaises(RuntimeError): stepper.step(target, .2)
        with self.assertRaises(RuntimeError): stepper.step(target+[.2, 0, 0], 1/30)

    def test_reader_rejects_stale_wrong_frame_high_speed_and_workspace_escape(self):
        plan = {'rig_sha256': 'test', 'surface_top_offset_m': .03,
                'object_workspace_min_m': [.4, -.03, .05], 'object_workspace_max_m': [.46, .03, .08],
                'max_object_speed_mps': .05}
        row = {'schema': 'front_side_observation_v1', 'running': True, 'stale': False,
               'rig_sha256': 'test', 'output_frame': 'right_base_estimate',
               'timing': {'oldest_host_arrival_monotonic_s': 10.},
               'track': {'accepted': True, 'ready': True, 'frame': 'right_base_estimate', 'position_m': [.43, 0, .04]},
               'velocity_report': {'speed_mps': .02}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'obs.json'; path.write_text(json.dumps(row))
            valid, _ = observation(path, plan, 10.1); np.testing.assert_allclose(valid['point'], [.43, 0, .07])
            self.assertIsNone(observation(path, plan, 10.2)[0])
            for changed in ({**row, 'rig_sha256': 'other'}, {**row, 'stale': True},
                            {**row, 'running': False}, {**row, 'velocity_report': {'speed_mps': .2}},
                            {**row, 'track': {**row['track'], 'position_m': [.6, 0, .04]}}):
                path.write_text(json.dumps(changed)); self.assertIsNone(observation(path, plan, 10.1)[0])

    def test_velocity_units_uncertainty_and_no_report_for_stale_or_unready(self):
        track = SimpleNamespace(ready=True, accepted=True, velocity_mps=np.array([.3, .4, 0]),
                                covariance=np.eye(6)*.0001, status='tracking')
        report = velocity_report(track)
        self.assertAlmostEqual(report['speed_mps'], .5); self.assertAlmostEqual(report['speed_cm_s'], 50)
        self.assertEqual(report['range_assessment'], 'inside')
        self.assertIsNone(velocity_report(track, True)['velocity_mps'])
        track.ready = False; self.assertIsNone(velocity_report(track)['speed_mps'])
        track.ready = True; track.velocity_mps[:] = 0
        self.assertEqual(velocity_report(track)['range_assessment'], 'below')


if __name__ == '__main__': unittest.main()
