import unittest
import numpy as np
from real_robot.aloha_grid_check import GridFollower


class GridFollowerTests(unittest.TestCase):
    def test_bias_compensation_tracks_moving_target_with_slew_bound(self):
        board=np.diag([1.,-1.,-1.,1.]);q=np.zeros(6);bias=np.deg2rad([0,1.1,.8,0,0,0])
        follower=GridFollower(board,q,q);actual=bias.copy();last=q.copy()
        for i in range(600):
            nominal=np.zeros(6);nominal[0]=np.deg2rad(min(i/60,3))
            goal=follower.step(nominal,actual,1/30)
            self.assertLessEqual(np.max(np.abs(goal-last)),np.deg2rad(3)/30+1e-9)
            self.assertLessEqual(np.max(np.abs(follower.correction)),np.deg2rad(3)+1e-9)
            actual+=(goal+bias-actual)*.15;last=goal
        self.assertLess(np.max(np.abs(actual-nominal)),np.deg2rad(.1))

    def test_height_and_timing_guards_stop_before_goal(self):
        board=np.diag([1.,-1.,-1.,1.]);board[2,3]=.5
        follower=GridFollower(board,np.zeros(6),np.zeros(6))
        with self.assertRaises(RuntimeError):follower.step(np.zeros(6),np.zeros(6),1/30)
        with self.assertRaises(ValueError):follower.step(np.zeros(6),np.zeros(6),.2)

    def test_large_joint_error_is_rejected(self):
        follower=GridFollower(np.diag([1.,-1.,-1.,1.]),np.zeros(6),np.zeros(6))
        with self.assertRaises(RuntimeError):follower.step(np.zeros(6),np.ones(6),1/30)
