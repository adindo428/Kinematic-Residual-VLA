"""Frozen tracking adaptations of the nine DynamicVLA Table 1 dimensions.

These are controlled tracking stress tests, not the DOM manipulation benchmark.
Only current RGB/state/velocity are sent to the unchanged policy worker.
"""
import numpy as np

DIMENSIONS = {
    'CR': 'Lateral direction reversals (one cycle per 4 seconds).',
    'DA': 'Smooth time-varying speed, alternating 0.5x and 1.5x.',
    'LS': 'Continuous 30-second tracking with repeated three-axis motion.',
    'VU': 'Exposure/contrast variation and a fixed peripheral visual occluder.',
    'SR': 'Tracking at ten different workspace/depth locations.',
    'MP': 'RGB freezes of 0.24 seconds at t=2 and t=5; state and velocity remain current.',
    'VG': 'Five apple meshes absent from the single-apple training data; same room.',
    'MG': 'Held-out motion speeds from 1.5x to 2x.',
    'DR': 'Two scripted 2.5 cm target displacements over 0.2 seconds (kinematic push proxy).',
}
MODELS = ('baseline', 'velocity', 'coordinate', 'coordinate_velocity')
SCORING_VERSION = 4
SUCCESS_DEFINITION = ('For each episode, calculate 100 times the duration with 3D EE-to-hover-target '
                      'distance <=0.03 m divided by the episode duration; average episode percentages '
                      'with equal weight. The entire episode, including the initial second, counts. '
                      'No binary pass/fail threshold or orientation gate. A policy termination earns '
                      'no tracking credit for the unexecuted remainder of its requested horizon; '
                      'completion rate and observed-only coverage are reported separately.')
TIME_DEFINITION = ('Time to first entry within 3 cm, capped at the requested horizon if never acquired. '
                   'This is acquisition time, not task completion time. Simulation time excludes '
                   'expert warmup and inference pauses.')


class BenchmarkPath:
    def __init__(self, trajectory, condition):
        self.data = {k: np.asarray(v, dtype=float) for k, v in trajectory.items()}
        self.condition = condition

    def sample(self, t):
        t = np.asarray(t, dtype=float)
        dimension = self.condition['dimension']
        speed = self.condition.get('speed_multiplier', 1.)
        clock, rate, acceleration = t*speed, t*0+speed, t*0
        if dimension == 'DA':
            w = 2*np.pi/4
            clock = t + .5*(1-np.cos(w*t))/w
            rate, acceleration = 1+.5*np.sin(w*t), .5*w*np.cos(w*t)
        d = self.data
        a = clock[..., None]*d['omega'] + d['phase']
        p = d['center'] + d['amplitude']*np.sin(a)
        v0 = d['amplitude']*d['omega']*np.cos(a)
        p0 = -d['amplitude']*d['omega']**2*np.sin(a)
        v, acc = v0*rate[..., None], p0*rate[..., None]**2+v0*acceleration[..., None]
        if dimension == 'DR':
            # C1 continuous, bounded target offsets. No artificial velocity inconsistency.
            direction = np.asarray(self.condition['push_direction'])
            for when, sign in ((2., 1.), (5., -1.)):
                u = np.clip((t-when)/.2, 0, 1)
                active = (t>when) & (t<when+.2)
                p += (.025*sign*u*u*(3-2*u))[..., None]*direction
                v += (.025*sign*6*u*(1-u)/.2*active)[..., None]*direction
                acc += (.025*sign*(6-12*u)/.2**2*active)[..., None]*direction
        return p, v, acc

    def serialize(self):
        return {'base_trajectory': {k: v.tolist() for k, v in self.data.items()},
                'condition': self.condition}


class ObservationCondition:
    def __init__(self, condition):
        self.condition = condition
        self.previous = None

    def apply(self, images, time):
        d = self.condition['dimension']
        if d == 'MP' and (2 <= time < 2.24 or 5 <= time < 5.24) and self.previous is not None:
            return {k: v.copy() for k, v in self.previous.items()}
        result = {k: v.copy() for k, v in images.items()}
        if d == 'VU':
            exposure = self.condition['exposure']
            contrast = self.condition['contrast']
            for k, rgb in result.items():
                rgb = np.clip(((rgb.astype(float)-127.5)*contrast+127.5)*exposure, 0, 255).astype('uint8')
                # Fixed image-space mask: never uses the true target position.
                rgb[:54, :96] = self.condition['occluder_rgb']
                result[k] = rgb
        self.previous = {k: v.copy() for k, v in result.items()}
        return result


def mean_episode_success_percent(metrics):
    """Equal episode weights, even when requested horizons differ."""
    return float(100*np.mean([m['success_fraction'] for m in metrics]))


def tracking_score(rows, completed, seconds, physics_hz, acquisition_seconds=1., tolerance_m=.03,
                   require_terminal_state=True):
    """Continuous coverage score; left-endpoint samples each represent one dt.

    The state at T validates completion but contributes zero duration. A terminated
    trial keeps the requested horizon denominator, giving no credit after stopping.
    """
    if seconds<=0 or physics_hz<=0 or tolerance_m<0:
        raise ValueError('Invalid scoring configuration')
    times = np.array([r['time'] for r in rows], dtype=float)
    errors = np.array([r['hover_error_m'] for r in rows], dtype=float)
    n_intervals = round(seconds*physics_hz)
    expected = np.arange(n_intervals+int(require_terminal_state))/physics_hz
    prefix_valid = bool(len(times)<=len(expected) and np.isfinite(errors).all()
                        and (errors>=0).all() and np.isfinite(times).all()
                        and np.allclose(times, expected[:len(times)], rtol=0, atol=1e-8))
    valid = prefix_valid and len(times)==len(expected)
    if not prefix_valid or (completed and not valid):
        raise ValueError('Invalid or incomplete telemetry for a completed episode')
    inside = errors<=tolerance_m
    interval_mask = times<seconds-1e-9
    inside_indices = np.flatnonzero(inside & interval_mask)
    fraction = len(inside_indices)/n_intervals
    observed_count = int(interval_mask.sum())
    post_mask = (times>=acquisition_seconds-1e-9) & interval_mask
    post_intervals = max(0,round((seconds-acquisition_seconds)*physics_hz))
    first_time = float(times[inside_indices[0]]) if len(inside_indices) else None
    stable_time = None
    if completed and valid and len(errors) and errors[-1]<=tolerance_m:
        outside = np.flatnonzero(errors>tolerance_m)
        stable_time = float(times[outside[-1]+1]) if len(outside) else 0.
    return {'success': fraction, 'success_fraction': fraction, 'success_percent': 100*fraction,
            'completed': bool(completed and valid), 'scoring_version': SCORING_VERSION,
            'telemetry_complete': bool(valid), 'telemetry_prefix_valid': prefix_valid,
            'position_tolerance_m': tolerance_m,
            'whole_episode_within_3cm_fraction': fraction,
            'observed_within_3cm_fraction': len(inside_indices)/observed_count if observed_count else None,
            'whole_episode_within_2cm_fraction': float(np.sum((errors<=.02) & interval_mask)/n_intervals),
            'within_tolerance_seconds': len(inside_indices)/physics_hz,
            'requested_duration_seconds': seconds, 'observed_duration_seconds': observed_count/physics_hz,
            'initial_second_counts_for_success': True,
            'terminal_sample_required': bool(require_terminal_state),
            'acquisition_seconds': acquisition_seconds,
            'post_acquisition_max_error_m': float(errors[post_mask].max()) if post_mask.any() else None,
            'post_acquisition_within_3cm_fraction': float(np.sum(inside & post_mask)/post_intervals) if post_intervals else None,
            'sustained_acquisition_time_s': stable_time,
            'time_to_first_within_3cm_s': first_time,
            'acquisition_time_capped_s': first_time if first_time is not None else seconds}
