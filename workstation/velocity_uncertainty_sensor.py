"""Simulator-side velocity sensor. Privileged states are isolated to diagnostics."""
import atexit
import gzip
import json
from pathlib import Path
import pickle
import re
import subprocess
import time
import numpy as np
from velocity_sensor_math import CausalSlope, error_metrics, gaussian_velocity, surface_position

REPO = Path(__file__).resolve().parents[1]
WORK = REPO/'scratch/dom45'
PROCESS = None
LOG = None


def close():
    global PROCESS
    if PROCESS is not None:
        PROCESS.terminate()
        try:
            PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            PROCESS.kill();PROCESS.wait()
        PROCESS = None
    if LOG is not None:
        LOG.close()


atexit.register(close)


def rpc(request):
    global PROCESS, LOG
    if PROCESS is None:
        import resident_table1_server as resident
        path = Path(resident.CURRENT['folder']).parents[2]/'efficienttam-worker.log'
        path.parent.mkdir(exist_ok=True, parents=True)
        LOG = path.open('a')
        PROCESS = subprocess.Popen([
            str(WORK/'envs/efficienttam/bin/python'), '-u', str(REPO/'workstation/efficienttam_stream_worker.py'),
            str(WORK/'perception/EfficientTAM'), str(WORK/'perception/checkpoints/efficienttam_ti_512x512.pt')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=LOG)
        receive(180)
    pickle.dump(request, PROCESS.stdin, protocol=4);PROCESS.stdin.flush()
    return receive(120)


def receive(timeout):
    import select
    if not select.select([PROCESS.stdout], [], [], timeout)[0]:
        raise TimeoutError('EfficientTAM response timed out')
    return pickle.load(PROCESS.stdout)


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def initial_target_mask(sensor):
    image = array(sensor.data.output['instance_segmentation_fast'])[0].squeeze()
    labels = sensor.data.info[0]['instance_segmentation_fast']['idToLabels']
    ids = []
    for key, value in labels.items():
        text = str(value)
        if re.search(r"/Object(?:/|['\"},\s]|$)", text) or (isinstance(value, dict) and str(value.get('class', '')).lower() == 'object'):
            ids.append(int(key))
    mask = np.isin(image, ids)
    if mask.shape != image.shape or mask.sum() < 20:
        raise RuntimeError(f'First-frame target mask unavailable: {labels}; never substitute object pose/velocity')
    return mask


class VelocitySensor:
    def __init__(self):
        from resident_table1_server import CURRENT, COUNTS
        self.job = dict(CURRENT)
        self.repeat = COUNTS[Path(self.job['scene']).stem]+1+self.job.get('repeat_offset', 0)
        self.mode = self.job['velocity_sensor']
        self.fit = CausalSlope()
        self.rows = []
        self.ready = []
        self.position_errors = []
        self.velocity_errors = []
        self.ready_errors = []
        self.initialized = False
        self.initialization_failure = None
        self.seed_mask_used = False
        self.latencies = []
        self.prepared = False
        self.render_refreshes = 0
        if self.mode == 'efficienttam':
            rpc({'reset': True})

    def prepare_frame(self, env):
        """Refresh every arm's first observation without advancing physics."""
        if self.prepared:
            return
        before = float(env.unwrapped.sim.current_time)
        # Resetting PhysX state does not guarantee cached RGB/depth/instance
        # buffers have rendered that state. Two render-only updates flush both
        # transforms and annotators, including when an old target mask exists.
        for _ in range(2):
            env.unwrapped.sim.render()
            for camera in env.unwrapped.scene.sensors.values():
                if type(camera).__name__ == 'Camera':
                    # IsaacLab 2.2.1's force_recompute still respects this flag.
                    camera._is_outdated[:] = True
                    camera.update(0., force_recompute=True)
            self.render_refreshes += 1
            assert float(env.unwrapped.sim.current_time) == before, 'Camera refresh advanced physics'
        self.prepared = True

    def estimate_from_images(self, env, index):
        """No object pose/velocity enters this function or the segmentation worker."""
        scene = env.unwrapped.scene
        sensor = scene.sensors['opst_cam']
        rgb = array(sensor.data.output['rgb'])[0, ..., :3]
        if self.initialization_failure is not None:
            return np.zeros(3), None, False, 0., 0
        try:
            first_mask = initial_target_mask(sensor) if not self.initialized else None
        except RuntimeError as exc:
            # A genuinely unavailable first-frame seed is a perception failure,
            # not a reason to skip the scene or inject later privileged identity.
            self.initialization_failure = str(exc)
            self.fit.points.clear()
            return np.zeros(3), None, False, 0., 0
        mask = rpc({'rgb': rgb, 'initial_mask': first_mask})['mask']
        self.initialized = True
        self.seed_mask_used = True
        robot = scene['robot'].data
        position = surface_position(
            mask, array(sensor.data.output['distance_to_image_plane'])[0],
            array(sensor.data.intrinsic_matrices)[0], array(sensor.data.pos_w)[0],
            array(sensor.data.quat_w_ros)[0], array(robot.root_pos_w)[0], array(robot.root_quat_w)[0])
        timestamp = index*float(env.env.step_dt)
        velocity, ready, lag = self.fit.update(timestamp, position)
        return velocity, position, ready, lag, int(np.sum(mask))

    def measure(self, env, truth_position, truth_velocity, index):
        truth_position = array(truth_position).reshape(-1, 3)[0].copy()
        truth_velocity = array(truth_velocity).reshape(-1, 3)[0].copy()
        tick = time.perf_counter()
        if self.mode == 'gaussian':
            velocity = gaussian_velocity(truth_velocity, float(self.job['sigma_v_mps']),
                                         self.job['scene'], self.repeat, index)
            position, ready, lag, pixels = None, True, 0., None
        else:
            velocity, position, ready, lag, pixels = self.estimate_from_images(env, index)
        elapsed = time.perf_counter()-tick
        self.latencies.append(elapsed)
        error = velocity-truth_velocity
        self.velocity_errors.append(error)
        self.ready.append(ready)
        if ready:
            self.ready_errors.append(error)
        if position is not None:
            self.position_errors.append(position-truth_position)
        self.rows.append(dict(index=index, timestamp_s=index*float(env.env.step_dt),
            truth_position_m=truth_position.tolist(), truth_velocity_mps=truth_velocity.tolist(),
            measured_velocity_mps=velocity.tolist(), estimated_surface_position_m=None if position is None else position.tolist(),
            velocity_ready=bool(ready), temporal_fit_lag_s=lag, mask_pixels=pixels, sensor_wall_s=elapsed))
        return velocity.astype(np.float32)[None]

    def finish(self):
        dest = Path(self.job['folder'])/'velocity_diagnostics'
        dest.mkdir(exist_ok=True, parents=True)
        stem = f'repeat{self.repeat:02d}'
        with gzip.open(dest/(stem+'.jsonl.gz'), 'wt') as f:
            for row in self.rows:
                f.write(json.dumps(row)+'\n')
        summary = dict(mode=self.mode, sigma_requested_mps=self.job.get('sigma_v_mps'),
            observations=len(self.rows), ready_observations=sum(self.ready),
            unavailable_fraction=float(1-np.mean(self.ready)) if self.ready else None,
            velocity_error_mps=error_metrics(self.velocity_errors),
            velocity_error_ready_mps=error_metrics(self.ready_errors),
            visible_surface_vs_COM_error_m=error_metrics(self.position_errors),
            sensor_p50_ms=float(np.median(self.latencies)*1000) if self.latencies else None,
            sensor_p95_ms=float(np.percentile(self.latencies, 95)*1000) if self.latencies else None,
            first_frame_privileged_mask_only=self.seed_mask_used,
            initialization_failure=self.initialization_failure,
            post_reset_render_refreshes=self.render_refreshes,
            future_frames_used=False, privileged_state_fallback=False,
            missing_velocity_policy='zero until three observations; zero and clear history on missing depth/mask')
        (dest/(stem+'.json')).write_text(json.dumps(summary, indent=2)+'\n')
