import unittest
import numpy as np
from real_robot.aloha_hover_check import validate_path,check_start,duration_for,RightArm,BoundedSettler
from real_robot.vx300s_kinematics import MOTOR_IDS


class FakeWriter:
    def __init__(self):self.parameters={};self.sent=False
    def clearParam(self):self.parameters={}
    def addParam(self,motor,data):self.parameters[motor]=data;return True
    def txPacket(self):self.sent=True;return 0


class HoverCheckTests(unittest.TestCase):
    def test_bounded_settling_cancels_static_error_without_large_steps(self):
        target=np.zeros(6);bias=np.deg2rad([0,1.2,1.,0,0,0])
        control=BoundedSettler(target,np.zeros(6));actual=bias.copy();last=np.zeros(6)
        for _ in range(450):
            goal=control.step(actual,1/30)
            self.assertLessEqual(np.max(np.abs(goal-last)),np.deg2rad(2)/30+1e-10)
            self.assertLessEqual(np.max(np.abs(goal-target)),np.deg2rad(3)+1e-10)
            actual+=(goal+bias-actual)*.15
            last=goal
        self.assertLess(np.max(np.abs(actual-target)),np.deg2rad(.1))
        with self.assertRaises(RuntimeError):control.step(np.ones(6),1/30)
        with self.assertRaises(ValueError):control.step(actual,.2)

    def test_parked_limit_overrun_only_recovers(self):
        p=np.zeros((20,6));p[:,2]=np.deg2rad(np.linspace(93,90,20))
        validate_path(p)
        p[:,2]=np.deg2rad(np.linspace(93,94,20))
        with self.assertRaises(ValueError):validate_path(p)
        p[:,2]=np.deg2rad(np.linspace(95,90,20))
        with self.assertRaises(ValueError):validate_path(p)

    def test_changed_start_rejected(self):
        with self.assertRaises(RuntimeError):check_start(np.ones(6)*.02,np.zeros(6))

    def test_path_rate_bound(self):
        u=np.linspace(0,1,301);s=10*u**3-15*u**4+6*u**5
        p=s[:,None]*np.deg2rad([100,20,10,0,0,0])[None,:]
        validate_path(p);duration=duration_for(p)
        speed=np.abs(np.diff(p,axis=0))/(duration/300)
        self.assertLessEqual(speed.max(),np.deg2rad(3)+1e-10)

    def test_only_six_master_goal_positions_written(self):
        arm=RightArm.__new__(RightArm);arm.writer=FakeWriter()
        arm.dxl=type('SDK',(),{'COMM_SUCCESS':0})
        q=np.deg2rad([1,-20,30,40,50,-60]);arm.goal(q)
        self.assertTrue(arm.writer.sent)
        self.assertEqual(set(arm.writer.parameters),set(MOTOR_IDS))
        self.assertNotIn(9,arm.writer.parameters)
        for motor,value in zip(MOTOR_IDS,q):
            raw=int.from_bytes(bytes(arm.writer.parameters[motor]),'little',signed=True)
            self.assertLessEqual(abs((raw-2048)*2*np.pi/4096-value),np.pi/4096+1e-9)
        arm.writer.sent=False
        with self.assertRaises(ValueError):arm.goal([np.nan]*6)
        self.assertFalse(arm.writer.sent)
