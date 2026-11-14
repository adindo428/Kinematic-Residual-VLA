import argparse
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock,patch

import numpy as np

from real_robot.relative_target import normalize_relative_target,parse_relative_target
from real_robot.aloha_charuco_capture import following_target


class RelativeTargetTests(unittest.TestCase):
    def test_collect_forwards_nine_limits_and_bottle_profile_to_recording(self):
        from real_robot import collect
        plan=(collect.PROJECT/'real_robot/collection_plan.json').read_text()
        limits=[20]*8+[35]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'real_robot').mkdir()
            (root/'real_robot/collection_plan.json').write_text(plan)
            with patch.object(collect,'PROJECT',root), \
                 patch.object(collect,'load_config',return_value={'perception':{'rig':'rig.json','max_skew_s':.035}}), \
                 patch.object(collect,'has_prompt',return_value=True), \
                 patch.object(collect,'LeaseOwner'),patch.object(collect,'stop_child'), \
                 patch.object(collect,'wait_for_perception'), \
                 patch.object(collect.subprocess,'Popen',return_value=Mock()) as popen, \
                 patch('real_robot.dynamic_capture.main',return_value=0) as capture, \
                 patch('real_robot.experiment_tracker.refresh'):
                code=collect.main(['--task','blue_bottle','--no-export','--episodes','100',
                                   '--robot.max_relative_target',json.dumps(limits),'--output',str(root/'session')])
            self.assertEqual(code,0)
            args=capture.call_args.args[0]
            self.assertEqual(json.loads(args[args.index('--robot-max-relative-target')+1]),limits)
            perception=popen.call_args.args[0]
            self.assertTrue(perception[perception.index('--object-profile')+1].endswith('blue_bottle/on_ramp'))
            self.assertIn('--front-mask-only',perception)
            self.assertNotIn('--front-camera-only',perception)

    def test_nine_motor_clipping_matches_installed_lerobot(self):
        import torch
        from lerobot.common.robot_devices.robots.manipulator import ensure_safe_goal_position
        limits=[20]*8+[35]
        actual=np.array([10,-12,4,7,2,-9,0,6,12],dtype=np.float32)
        for delta in [np.full(9,100,dtype=np.float32),np.full(9,-100,dtype=np.float32),np.arange(9,dtype=np.float32)]:
            desired=actual+delta
            expected=ensure_safe_goal_position(torch.tensor(desired),torch.tensor(actual),limits).numpy()
            np.testing.assert_allclose(following_target(desired,actual,limits),expected)

    def test_bad_bounds_are_rejected_before_hardware(self):
        for value in ['[20,35]','[20,20,20,20,20,20,20,20,-35]',
                      '[20,20,20,20,20,20,20,20,NaN]','true','[20,20,20,20,20,20,20,20,"35"]','0','Infinity']:
            with self.subTest(value=value),self.assertRaises(argparse.ArgumentTypeError):parse_relative_target(value)
        self.assertEqual(normalize_relative_target('20'),20)
        self.assertEqual(parse_relative_target('[20,20,20,20,20,20,20,20,35]'),[20]*8+[35])
