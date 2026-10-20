import unittest
import numpy as np

from real_robot.efficienttam_tracker import EfficientTAMTracker, prune_state, copy_buffers, project_native_pixel


class StreamingMemoryTests(unittest.TestCase):
    def test_native_projection_inverts_distorted_deprojection(self):
        intr = {'width': 640, 'height': 480, 'fx': 300., 'fy': 300., 'ppx': 320., 'ppy': 240.}
        def deproject(pixel, z):
            x, y = (np.array(pixel)-[320, 240])/300
            return [z*(x+.05*x**3), z*(y+.03*y**3), z]
        frame = {'timing': {'intrinsics': intr}, 'deproject': deproject}
        np.testing.assert_allclose(project_native_pixel(deproject([100, 70], .4), frame), [100, 70], atol=.001)
        self.assertIsNone(project_native_pixel([0, 0, -.4], frame))
        self.assertIsNone(project_native_pixel([5, 0, .4], frame))

    def test_recovery_accepts_only_blue_depth_consistent_masks(self):
        for blue, depth, expected in [(True, .3, True), (False, .3, False), (True, .45, False)]:
            with self.subTest(blue=blue, depth=depth):
                tracker = EfficientTAMTracker.__new__(EfficientTAMTracker)
                original = {'original': True}; proposal = {'proposal': True}
                tracker.states = {'front': original, 'side': {}}
                tracker.indices = {'front': 10, 'side': 10}; tracker.seed_depths = {'front': .3, 'side': .3}
                tracker.seed_boxes = {'front': [40, 40, 20, 20]}; tracker.last_recovery = {}
                mask = np.zeros((100, 100), np.uint8); mask[40:60, 40:60] = 255
                calls = []
                def propose(role, box):
                    calls.append(box); return mask, {'object_probability': .99}, proposal
                tracker._propose = propose
                frame = {'timing': {'sensor_timestamp_s': 1., 'intrinsics':
                    {'width': 100, 'height': 100, 'fx': 100., 'fy': 100., 'ppx': 50., 'ppy': 50.}},
                    'bgr': np.full((100, 100, 3), [200, 0, 0] if blue else [0, 0, 200], np.uint8),
                    'depth': np.full((100, 100), depth),
                    'deproject': lambda px, z: [(px[0]-50)*z/100, (px[1]-50)*z/100, z]}
                candidates = {'front': [], 'side': [{'point': np.array([0, 0, .3])}]}
                infos = {'front': {'object_probability': .1}, 'side': {'object_probability': .99}}
                tracker.recover_pair({'front': frame, 'side': frame}, candidates, infos, np.eye(4))
                self.assertEqual(infos['front']['auto_recovery']['accepted'], expected)
                self.assertIs(tracker.states['front'], proposal if expected else original)
                self.assertEqual(bool(candidates['front']), expected)
                if not expected:
                    tracker.recover_pair({'front': frame, 'side': frame}, candidates, infos, np.eye(4))
                    self.assertEqual(len(calls), 1)

    def test_trial_containers_are_isolated_while_buffers_are_shared(self):
        array = np.ones((3, 3)); state = {'nested': [{'buffer': array, 'metadata': {'old': True}}]}
        trial = copy_buffers(state); trial['nested'][0]['metadata'].clear()
        self.assertIs(trial['nested'][0]['buffer'], array)
        self.assertEqual(state['nested'][0]['metadata'], {'old': True})

    def test_pruning_preserves_prompt_and_recent_causal_memory(self):
        prompt = object()
        state = {'output_dict_per_obj': {0: {'cond_frame_outputs': {0: prompt},
                  'non_cond_frame_outputs': {i: object() for i in range(1, 101)}}},
                 'frames_tracked_per_obj': {0: {i: {'reverse': False} for i in range(101)}},
                 'images': {100: object()}}
        prune_state(state, 100)
        output = state['output_dict_per_obj'][0]
        self.assertIs(output['cond_frame_outputs'][0], prompt)
        self.assertEqual(set(output['non_cond_frame_outputs']), set(range(69, 101)))
        self.assertEqual(set(state['frames_tracked_per_obj'][0]), {0, *range(69, 101)})
        self.assertEqual(list(state['images']), [100])


if __name__ == '__main__': unittest.main()
