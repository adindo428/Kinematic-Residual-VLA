import unittest
import numpy as np
from real_robot.aloha_arm import RightArm, signed
from real_robot.vx300s_kinematics import MOTOR_IDS


class FakeWriter:
    def __init__(self):self.parameters={};self.sent=False
    def clearParam(self):self.parameters={}
    def addParam(self,motor,data):self.parameters[motor]=data;return True
    def txPacket(self):self.sent=True;return 0


class AlohaArmTests(unittest.TestCase):
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

    def test_signed_motor_units(self):
        self.assertEqual(signed(65535, 16), -1)
        self.assertEqual(signed(0xffffffff, 32), -1)
        self.assertEqual(signed(2048, 32), 2048)
