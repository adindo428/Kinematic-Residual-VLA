"""Offline image-servo, state-machine, and contact-guard tests. No devices."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from real_robot.stationary_pick import StationaryPick, PoseServo, PickGeometry, inside_polygon, pose_error
from real_robot.vx300s_kinematics import forward, inverse
from real_robot.wrist_alignment import blue_seed, tape_feature, line_error, corrections, roll_step


def fixture():
    qs = {'pregrasp':[-.17027187,.20401944,.02914563,-.10124273,1.13821374,.24083498],
        'before_close':[-.20401944,.25924275,.28992237,-.06749515,.71483505,.24390295],
        'lifted':[-.10891264,-.08283496,.03528156,-.14112623,1.2532623,.30833014],
        'above_box':[-.51081560,.00766990,.06596117,-.30986412,1.33763125,.47246608],
        'release':[-.47706803,-.03067962,.12732041,-.29299033,1.24866036,.47246608]}
    template={'tcp_offset_m':[.16,0,0],'object_reference_m':[.4417,-.0994,.03],
        'stages':{k:{'T_base_tcp':forward(q,[.16,0,0]).tolist(),'q_rad':q} for k,q in qs.items()},
        'wrist_reference':{'axis_rad':np.deg2rad(93.5),'center_px':[358.,267.],
            'axis_reliable':True,'depth_m':.182},
        'T_gripper_wrist_camera':[[.04109096,-.44785687,.89316054,.07263760],
            [-.99799696,-.06143059,.01511099,.01176350],
            [.04809982,-.89199244,-.44948404,.08309034],[0,0,0,1]],
        'gripper':{'open_raw':3104,'empty_closed_raw':2475,'observed_contact_raw':2645}}
    box={'interior_polygon_xy_m':[[.27,-.30],[.39,-.30],[.39,-.18],[.27,-.18]],
         'release_xy_m':[.3275,-.2401],'rim_z_base_m':.055}
    return template,box,np.array(qs['pregrasp'])


class WristTests(unittest.TestCase):
    def test_axis_is_undirected_and_roll_correction_has_verified_sign(self):
        self.assertAlmostEqual(line_error(np.deg2rad(89),np.deg2rad(-89)),np.deg2rad(-2))
        self.assertLess(roll_step(np.deg2rad(5),.89),0)
        with self.assertRaises(ValueError):roll_step(.1,.1)

    def test_blue_reflection_does_not_change_tape_axis(self):
        image=np.full((480,640,3),100,np.uint8)
        image[80:420,50:270]=[220,180,150]  # reflected illumination, low saturation
        rectangle=cv2.boxPoints(((350,260),(35,110),20)).astype(int)
        cv2.fillConvexPoly(image,rectangle,(180,30,10))
        _,mask=blue_seed(image)
        f=tape_feature(image,mask,np.full((480,640),.18))
        self.assertTrue(f['axis_reliable'])
        self.assertLess(np.linalg.norm(np.array(f['center_px'])-[350,260]),3)
        self.assertAlmostEqual(f['depth_m'],.18)
        self.assertLess(abs(line_error(f['axis_rad'],np.deg2rad(110))),np.deg2rad(2))

    def test_round_or_clipped_mask_cannot_command_alignment(self):
        image=np.zeros((480,640,3),np.uint8);mask=np.zeros((480,640),np.uint8)
        cv2.circle(mask,(350,260),40,255,-1);image[mask>0]=[180,30,10]
        f=tape_feature(image,mask,np.full((480,640),.2))
        self.assertFalse(f['axis_reliable'])
        with self.assertRaises(ValueError):corrections(f,f,{},np.eye(4))

    def test_image_translation_correction_reduces_error(self):
        p,_,q=fixture();camera=forward(q)@np.array(p['T_gripper_wrist_camera'])
        intr={'fx':393.,'fy':393.,'ppx':323.,'ppy':240.}
        feature={'center_px':[375.,275.],'axis_rad':1.6,'axis_reliable':True,'depth_m':.18}
        ref={'center_px':[358.,267.],'axis_rad':1.6}
        delta,_=corrections(feature,ref,intr,camera)
        self.assertLessEqual(np.linalg.norm(delta),.00300001)
        z=feature['depth_m'];u,v=feature['center_px']
        pc=np.array([(u-intr['ppx'])*z/intr['fx'],(v-intr['ppy'])*z/intr['fy'],z,1])
        world=camera@pc;moved=camera.copy();moved[:3,3]+=delta
        new=np.linalg.inv(moved)@world
        pixel=np.array([intr['fx']*new[0]/new[2]+intr['ppx'],intr['fy']*new[1]/new[2]+intr['ppy']])
        self.assertLess(np.linalg.norm(pixel-ref['center_px']),np.linalg.norm(np.array(feature['center_px'])-ref['center_px']))


class ServoTests(unittest.TestCase):
    def test_retained_holding_effort_and_command_rate(self):
        _,_,q=fixture();held=q+np.deg2rad([.2,1,-.5,0,.5,0])
        servo=PoseServo(q,held)
        np.testing.assert_allclose(servo.step(q,q,1/30),held)
        old=servo.goal.copy();command=servo.step(q+.1,q,1/30)
        self.assertLessEqual(np.max(np.abs(command-old)),np.deg2rad(3)/30+1e-10)
        with self.assertRaises(RuntimeError):servo.step(q,q-.3,1/30)


class StateMachineTests(unittest.TestCase):
    def run_attempt(self,drop=False,placement_visible=True,probe_gain=-.9):
        p,box,q0=fixture();q=q0.copy();machine=StationaryPick(p,box,p['object_reference_m'],q,0.)
        seen=[]
        for k in range(2000):
            now=k*.1
            q=machine.target_q.copy() if machine.target_q is not None else inverse(machine.target_pose,q,p['tcp_offset_m'])
            f=dict(p['wrist_reference']);f['axis_rad']+=probe_gain*(q[5]-q0[5]);raw=3104;current=.05
            if machine.stage in ('CLOSE','LIFT','VERIFY_LIFT','TRANSPORT','ABOVE_BOX') or (machine.stage=='RELEASE' and machine.gripper_target!=machine.open_raw):
                raw=2645;current=-.52;f.update(depth_m=.095,center_px=[380,390])
            if drop and machine.stage=='VERIFY_LIFT':f.update(depth_m=.24)
            intr={'fx':393.,'fy':393.,'ppx':323.,'ppy':240.}
            if machine.stage in ('VERIFY_RELEASE','RETREAT'):
                camera=forward(q)@np.array(p['T_gripper_wrist_camera'])
                point=np.linalg.inv(camera)@np.array([*box['release_xy_m'],.025,1])
                f.update(depth_m=float(point[2]),center_px=[intr['fx']*point[0]/point[2]+intr['ppx'],intr['fy']*point[1]/point[2]+intr['ppy']])
                if not placement_visible:f=None
            machine.update(now,q,raw,current,{'feature':f,'timing':{'sensor_timestamp_s':now,'intrinsics':intr}})
            if not seen or seen[-1]!=machine.stage:seen.append(machine.stage)
            if machine.done:return machine,seen
        raise AssertionError('State machine did not finish or fail')

    def test_full_feedback_driven_attempt(self):
        m,seen=self.run_attempt()
        self.assertTrue(m.done)
        self.assertEqual(seen,['OPEN','PREGRASP','PROBE','ALIGN','DESCEND','CLOSE','LIFT',
            'VERIFY_LIFT','TRANSPORT','ABOVE_BOX','RELEASE','VERIFY_RELEASE','RETREAT','COMPLETE'])
        self.assertAlmostEqual(m.roll_gain,-.9)

    def test_drop_prevents_transport(self):
        with self.assertRaisesRegex(RuntimeError,'Lift verification failed'):self.run_attempt(drop=True)

    def test_release_timer_cannot_fabricate_placement_success(self):
        with self.assertRaisesRegex(RuntimeError,'placement could not be visually verified'):
            self.run_attempt(placement_visible=False)

    def test_wrong_wrist_response_aborts_before_descent(self):
        with self.assertRaisesRegex(RuntimeError,'response not verified'):self.run_attempt(probe_gain=.9)

    def test_empty_closed_gripper_is_not_object_contact(self):
        p,box,q=fixture();m=StationaryPick(p,box,p['object_reference_m'],q,0.)
        m.stage='CLOSE';f=p['wrist_reference']
        with self.assertRaisesRegex(RuntimeError,'No supported gripper contact'):
            for k in range(90):m.update(k*.1,q,2475,-.53,{'feature':f,'timing':{'sensor_timestamp_s':k}})

    def test_moving_object_and_unreliable_alignment_abort(self):
        p,box,q=fixture();m=StationaryPick(p,box,p['object_reference_m'],q,0.)
        with self.assertRaisesRegex(RuntimeError,'Tape moved'):m.update(.1,q,3104,.05,{},False)
        m.stage='ALIGN'
        with self.assertRaisesRegex(RuntimeError,'axis/depth lost'):m.update(.2,q,3104,.05,{'feature':None})


class ContactGeometryTests(unittest.TestCase):
    def test_contact_relaxes_only_fingers_and_keeps_camera_clearance(self):
        def xml(camera_low=False):
            start='<mujoco><worldbody><geom name="table" type="box" size="2 2 .01" pos="0 0 -.01"/>'
            for i in range(6):
                start+=f'<body pos=".03 0 {".2" if i==0 else "0"}"><joint type="hinge" axis="0 0 1"/><geom type="box" size=".005 .005 .005" mass=".1"/>'
            for side,sign in [('left',1),('right',-1)]:
                start+=f'<body pos="0 0 -.185"><joint type="slide" axis="0 1 0"/><geom name="{side}_gripper_finger" type="box" size=".002 .002 .002" mass=".01"/></body>'
            if camera_low:start+='<geom name="wrist_camera_bound" type="box" pos=".02 0 -.185" size=".002 .002 .002" mass=".01"/>'
            return start+'</body>'*6+'</worldbody></mujoco>'
        _,box,_=fixture();box['interior_polygon_xy_m']=[[1,1],[1.2,1],[1.2,1.2],[1,1.2]]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'robot.xml';path.write_text(xml())
            g=PickGeometry(str(path),np.zeros(6),box)
            with self.assertRaisesRegex(RuntimeError,'Table clearance'):g.check(np.zeros(6))
            g.check(np.zeros(6),contact=True)
            path.write_text(xml(True));g=PickGeometry(str(path),np.zeros(6),box)
            with self.assertRaisesRegex(RuntimeError,'wrist_camera_bound'):g.check(np.zeros(6),contact=True)


if __name__=='__main__':
    unittest.main()
