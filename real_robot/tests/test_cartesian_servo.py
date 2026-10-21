import json,tempfile,unittest
from pathlib import Path
import numpy as np
from real_robot.cartesian_servo import CartesianIKServo,validate_servo
from real_robot.vx300s_kinematics import inverse_downward,forward,LOWER,UPPER
from real_robot.aloha_object_servo import read_target
from real_robot.visible_extent import visible_top

class CartesianServoTests(unittest.TestCase):

    def test_visible_top_rejects_missing_depth_and_background_outliers(self):
        yy,xx=np.mgrid[:100,:100];mask=(((xx-50)**2+(yy-50)**2)<30**2).astype(np.uint8)*255
        depth=np.full((100,100),.4);depth[25:27,45:55]=1.2
        transform=np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,.07],[0,0,0,1.]])
        deproject=lambda xy,z: [(xy[0]-50)*.001,(xy[1]-50)*.001,z]
        quality={'reason':None,'depth_m':.4,'depth_mad_m':.001}
        top=visible_top(mask,depth,deproject,transform,quality)
        self.assertTrue(.094<top<.105)
        self.assertIsNone(visible_top(mask,depth*0,deproject,transform,quality))
        self.assertIsNone(visible_top(None,depth,deproject,transform,quality))
    def test_analytic_ik_matches_full_fk_across_tabletop(self):
        seed=np.deg2rad([0,3,10,0,77,0]);rotation=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        for x in [.25,.35,.45,.49]:
            for y in [-.12,0,.12]:
                for z in [.10,.16,.20]:
                    q=inverse_downward([x,y,z],seed)
                    pose=forward(q,[.16,0,0]);np.testing.assert_allclose(pose[:3,3],[x,y,z],atol=1e-9)
                    np.testing.assert_allclose(pose[:3,:3],rotation,atol=1e-9)
                    self.assertTrue(np.all(q>=LOWER)&np.all(q<=UPPER))
        with self.assertRaises(ValueError):inverse_downward([1,0,.1],seed)

    def test_sixteen_cm_move_converges_with_tcp_and_joint_rate_caps(self):
        initial=np.array([.36,0,.125]);target=initial+[.16,.03,.005]
        q=inverse_downward(initial,np.zeros(6));servo=CartesianIKServo(q);dt=1/30
        for _ in range(600):
            previous=servo.q.copy();candidate=servo.step(target,q,dt)
            self.assertLessEqual(np.max(np.abs(candidate-previous)),np.deg2rad(6)*dt+1e-9)
            self.assertLessEqual(np.linalg.norm(forward(candidate,[.16,0,0])[:3,3]-forward(previous,[.16,0,0])[:3,3]),.03*dt+1e-8)
            servo.accept(candidate);q=candidate.copy()
        self.assertLess(np.linalg.norm(forward(q,[.16,0,0])[:3,3]-target),.001)

    def test_retained_goal_is_not_replaced_by_measured_gravity_sag(self):
        held=np.deg2rad([-.8,1.5,9.8,0,76.2,-.8]);actual=held.copy();actual[1]+=np.deg2rad(1.9)
        servo=CartesianIKServo(held);np.testing.assert_array_equal(servo.q,held)
        target=forward(actual,[.16,0,0])[:3,3]
        candidate=servo.step(target,actual,1/30)
        self.assertLessEqual(np.max(np.abs(candidate-held)),np.deg2rad(6)/30+1e-9)
        np.testing.assert_array_equal(servo.hold(),held)

    def test_limits_unreachable_targets_and_bad_settings(self):
        q=inverse_downward([.36,0,.12],np.zeros(6));servo=CartesianIKServo(q)
        with self.assertRaises(ValueError):servo.step([.7,0,.12],q,1/30)
        with self.assertRaises(ValueError):servo.step([.54,.18,.30],q,1/30)
        with self.assertRaises(ValueError):servo.step([.36,0,.12],q,.2)
        for config in [{'max_joint_speed_deg_s':1000},{'max_tcp_speed_mps':.8},{'kp':float('nan')},{'unknown':1}]:
            with self.assertRaises(ValueError):validate_servo(config)

    def test_half_meter_per_second_profile_removes_pilot_joint_and_acceleration_caps(self):
        from real_robot.vx300s_kinematics import inverse_tool_pitch
        settings=dict(max_tcp_speed_mps=.5,max_joint_speed_deg_s=None,max_tcp_acceleration_mps2=None,
            max_command_lead_m=None,max_following_error_deg=12,tool_pitch_deg=60,experimental_current_guard=False)
        q=inverse_tool_pitch([.30,0,.15],np.zeros(6),60)
        with self.assertRaises(ValueError):CartesianIKServo(q,settings)
        servo=CartesianIKServo(q,settings,[180]*6);speeds=[]
        for _ in range(120):
            previous=servo.q.copy();candidate=servo.step([.53,.08,.15],q,1/30)
            speed=np.linalg.norm(forward(candidate,[.16,0,0])[:3,3]-forward(previous,[.16,0,0])[:3,3])*30
            self.assertLessEqual(speed,.5+1e-8);speeds.append(speed)
            self.assertLessEqual(np.max(np.abs(candidate-previous)),np.deg2rad(180)/30+1e-9)
            servo.accept(candidate);q=candidate
        self.assertGreater(max(speeds),.1)
        self.assertLess(np.linalg.norm(forward(q,[.16,0,0])[:3,3]-[.53,.08,.15]),.002)

    def test_fast_object_is_accepted_but_stale_and_wrong_frame_are_held(self):
        c=validate_servo();row=dict(schema='front_side_observation_v1',running=True,stale=False,
            rig_sha256='test',output_frame='right_base_estimate',timing={'oldest_host_arrival_monotonic_s':10.},
            track={'accepted':True,'ready':True,'frame':'right_base_estimate','position_m':[.4,0,.03]},
            velocity_report={'speed_mps':.4})
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'observation.json';p.write_text(json.dumps(row))
            target,reason=read_target(p,'test',c,10.1);self.assertIsNone(reason)
            np.testing.assert_allclose(target['target_m'],[.4,0,.122])
            self.assertIsNone(read_target(p,'other',c,10.1)[0]);self.assertIsNone(read_target(p,'test',c,10.3)[0])
            row['track']['ready']=False
            row['fusion']={'identity_recent':True,'stale':False,'output_frame':'right_base_estimate',
                'sources':['side'],'position_output_m':[.4,.01,.03],'visible_top_z_base_m':.075}
            p.write_text(json.dumps(row));target,reason=read_target(p,'test',c,10.1)
            self.assertIsNone(reason);np.testing.assert_allclose(target['target_m'],[.4,.01,.125])
            p.write_text('{');self.assertIsNone(read_target(p,'test',c,10.1)[0])

if __name__=='__main__':unittest.main()
