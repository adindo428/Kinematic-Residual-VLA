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


def tracking_score(rows, completed, seconds, physics_hz, acquisition_seconds=1., tolerance_m=.02):
    """Success requires every 200 Hz sample in [1,T], including final state.

    Time is sustained acquisition time, capped at the trial horizon for failures.
    Successful-only time is provided separately. Missing/infrastructure runs do
    not produce a published success table; policy terminations count as failures.
    """
    times = np.array([r['time'] for r in rows], dtype=float)
    errors = np.array([r['hover_error_m'] for r in rows], dtype=float)
    expected = np.arange(round(seconds*physics_hz)+1)/physics_hz
    valid = bool(len(times)==len(expected) and np.isfinite(errors).all()
                 and np.isfinite(times).all() and np.allclose(times, expected, rtol=0, atol=1e-8))
    mask = times>=acquisition_seconds-1e-9
    success = bool(completed and valid and mask.any() and (errors[mask]<=tolerance_m).all())
    stable_time = None
    if completed and valid and errors[-1]<=tolerance_m:
        outside = np.flatnonzero(errors>tolerance_m)
        stable_time = float(times[outside[-1]+1]) if len(outside) else 0.
    return {'success': success, 'completed': bool(completed and valid),
            'telemetry_complete': valid, 'position_tolerance_m': tolerance_m,
            'acquisition_seconds': acquisition_seconds,
            'post_acquisition_max_error_m': float(errors[mask].max()) if mask.any() else None,
            'post_acquisition_within_2cm_fraction': float(np.sum(errors[mask]<=tolerance_m)/np.sum(expected>=acquisition_seconds-1e-9)),
            'observed_post_acquisition_within_2cm_fraction': float(np.mean(errors[mask]<=tolerance_m)) if mask.any() else None,
            'sustained_acquisition_time_s': stable_time,
            'successful_acquisition_time_s': stable_time if success else None,
            'failure_capped_time_s': stable_time if success else seconds}
