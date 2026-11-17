"""CPU checks for noise pairing, geometry, causality and simulator hook placement."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

from velocity_sensor_math import CausalSlope, error_metrics, gaussian_velocity, surface_position


class Measurements(unittest.TestCase):
    def test_noise_pairing_and_requested_scale(self):
        truth = np.array([.2, -.1, .04])
        a = gaussian_velocity(truth, .01, 'scene', 4, 31)
        np.testing.assert_array_equal(a, gaussian_velocity(truth, .01, 'scene', 4, 31))
        b = gaussian_velocity(truth, .05, 'scene', 4, 31)
        np.testing.assert_allclose(b-truth, 5*(a-truth), rtol=1e-12)
        self.assertFalse(np.array_equal(a, gaussian_velocity(truth, .01, 'scene', 5, 31)))
        np.testing.assert_array_equal(truth, [.2, -.1, .04])
        errors = np.stack([gaussian_velocity(np.zeros(3), .05, 'scene', 1, i) for i in range(6000)])
        np.testing.assert_allclose(errors.mean(0), 0, atol=.002)
        np.testing.assert_allclose(errors.std(0), .05, atol=.002)

    def test_camera_robot_geometry(self):
        # A symmetric patch at optical z=1 transforms through rotated camera/base.
        mask = np.ones((5, 5), bool)
        k = np.array([[100., 0, 2], [0, 100., 2], [0, 0, 1]])
        q = np.array([np.sqrt(.5), 0, 0, np.sqrt(.5)])
        p = surface_position(mask, np.ones((5, 5)), k, np.array([1., 2., 0]), q,
                             np.array([1., 1., 0]), q)
        np.testing.assert_allclose(p, [1, 0, 1], atol=1e-12)
        self.assertIsNone(surface_position(mask, np.zeros((5, 5)), k, np.zeros(3), q, np.zeros(3), q))

    def test_slope_and_missing_observation(self):
        fit = CausalSlope()
        v = np.array([.12, -.04, .08])
        for i in range(8):
            estimate, ready, lag = fit.update(i*.04, np.ones(3)+i*.04*v)
        self.assertTrue(ready)
        np.testing.assert_allclose(estimate, v, atol=1e-12)
        self.assertAlmostEqual(lag, .08)
        estimate, ready, _ = fit.update(.32, None)
        self.assertFalse(ready);np.testing.assert_array_equal(estimate, np.zeros(3))
        self.assertEqual(len(fit.points), 0)

    def test_equivalent_noise_is_rmse_not_centered_std(self):
        m = error_metrics([[.1, .1, .1]]*20)
        self.assertAlmostEqual(m['sigma_equivalent'], .1)
        np.testing.assert_allclose(m['std_xyz'], 0, atol=1e-12)

    def test_first_frame_instance_mapping(self):
        from velocity_uncertainty_sensor import initial_target_mask
        image = np.ones((1, 5, 5, 1), dtype=np.int32)*7
        for label in ['/World/envs/env_0/Object/mesh', {'primPath': '/World/envs/env_0/Object'}, {'class': 'OBJECT'}]:
            sensor = SimpleNamespace(data=SimpleNamespace(output={'instance_segmentation_fast': image},
                info=[{'instance_segmentation_fast': {'idToLabels': {'7': label}}}]))
            self.assertEqual(initial_target_mask(sensor).sum(), 25)

    def test_diagnostics_do_not_mutate_truth(self):
        import resident_table1_server as resident
        from velocity_uncertainty_sensor import VelocitySensor
        with tempfile.TemporaryDirectory() as temp:
            resident.CURRENT = dict(scene='scene.json', folder=temp, velocity_sensor='gaussian', sigma_v_mps=.05)
            sensor = VelocitySensor()
            env = SimpleNamespace(env=SimpleNamespace(step_dt=.04))
            truth = np.array([[.2, .1, .0]], dtype=np.float32)
            original = truth.copy()
            for i in range(20):
                estimate = sensor.measure(env, np.zeros((1, 3)), truth, i)
                self.assertEqual(estimate.shape, (1, 3))
            sensor.finish()
            np.testing.assert_array_equal(truth, original)
            summary = json.loads((Path(temp)/'velocity_diagnostics/repeat01.json').read_text())
            self.assertFalse(summary['privileged_state_fallback'])
            self.assertEqual(summary['velocity_error_mps']['samples'], 20)

    def test_missing_reset_mask_refreshes_without_physics_step(self):
        from velocity_uncertainty_sensor import VelocitySensor
        class Camera:
            def __init__(self):
                self._is_outdated = np.array([False])
                self.data = SimpleNamespace(output={'instance_segmentation_fast': np.ones((1, 5, 5, 1), np.int32)},
                    info=[{'instance_segmentation_fast': {'idToLabels': {'1': 'BACKGROUND'}}}])
            def update(self, dt, force_recompute=False):
                assert self._is_outdated.all()
                self.data.info[0]['instance_segmentation_fast']['idToLabels'] = {'1': '/World/envs/env_0/Object'}
        camera = Camera()
        class Sim:
            current_time = 0.
            renders = 0
            def render(self):self.renders += 1
        sim = Sim()
        sensor = object.__new__(VelocitySensor)
        sensor.prepared = False;sensor.mode = 'efficienttam';sensor.render_refreshes = 0
        env = SimpleNamespace(unwrapped=SimpleNamespace(sim=sim, scene=SimpleNamespace(sensors={'opst_cam': camera})))
        sensor.prepare_frame(env)
        self.assertTrue(sensor.prepared)
        self.assertEqual(sim.renders, 2)
        self.assertEqual(sim.current_time, 0.)
        sensor.prepare_frame(env)
        self.assertEqual(sim.renders, 2)

    def test_unavailable_initial_mask_never_uses_future_privileged_seed(self):
        import velocity_uncertainty_sensor as module
        sensor = object.__new__(module.VelocitySensor)
        sensor.initialized = False;sensor.initialization_failure = None;sensor.fit = CausalSlope()
        camera = SimpleNamespace(data=SimpleNamespace(output={
            'rgb': np.zeros((1, 5, 5, 3), np.uint8), 'instance_segmentation_fast': np.zeros((1, 5, 5, 1), np.int32)},
            info=[{'instance_segmentation_fast': {'idToLabels': {'0': 'BACKGROUND'}}}]))
        env = SimpleNamespace(unwrapped=SimpleNamespace(scene=SimpleNamespace(sensors={'opst_cam': camera})))
        velocity, position, ready, _, _ = sensor.estimate_from_images(env, 0)
        self.assertFalse(ready);self.assertIsNone(position);np.testing.assert_array_equal(velocity, np.zeros(3))
        self.assertIsNotNone(sensor.initialization_failure)
        camera.data.output['instance_segmentation_fast'][:] = 1
        camera.data.info[0]['instance_segmentation_fast']['idToLabels']['1'] = '/World/envs/env_0/Object'
        velocity, position, ready, _, _ = sensor.estimate_from_images(env, 1)
        self.assertFalse(ready);self.assertIsNone(position)

    def test_pinned_simulator_patch(self):
        from velocity_uncertainty_server import patch_source
        source = Path(__file__).resolve().parents[1]/'scratch/dom45/official/source/simulations/evaluate.py'
        text = patch_source(source.read_text())
        ast.parse(text)
        # Sensor modifies a copied observation dictionary; no physical root-state write.
        self.assertIn("curr_state['object'] = dict(curr_state['object'], velocity=", text)
        self.assertNotIn('write_root_velocity', text)
        self.assertLess(text.index('_velocity_sensor.measure('), text.index('"dt_scale":'))


if __name__ == '__main__':
    unittest.main()
