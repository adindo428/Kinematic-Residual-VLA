from dataclasses import dataclass

import numpy as np

from .controller import ModeSelector

MODES = (-45, 0, 45)


@dataclass
class Trajectory:
    center: np.ndarray
    amplitude: np.ndarray
    omega: np.ndarray
    phase: np.ndarray

    def sample(self, t):
        a = np.asarray(t)[..., None] * self.omega + self.phase
        return (self.center + self.amplitude*np.sin(a),
                self.amplitude*self.omega*np.cos(a),
                -self.amplitude*self.omega**2*np.sin(a))

    def serialize(self):
        return {k: getattr(self, k).tolist() for k in ("center", "amplitude", "omega", "phase")}


def generate(cfg, rng, seconds):
    half = np.asarray(cfg["half_extents"])-cfg["workspace_margin"]
    amplitude = half*rng.uniform(0.55, 0.95, 3)
    omega = 2*np.pi*rng.uniform(*cfg["frequency_range"], 3)
    # Analytic conservative norm bounds hold between sampled timesteps, too.
    scale = min(1, cfg["max_speed"]/np.linalg.norm(amplitude*omega),
                cfg["max_acceleration"]/np.linalg.norm(amplitude*omega**2))
    amplitude *= scale
    center = np.asarray(cfg["center"]) + rng.uniform(-1, 1, 3)*(half-amplitude)*0.5
    phase = rng.uniform(-np.pi, np.pi, 3)
    kind = cfg["kind"]
    if kind == "stationary":
        amplitude[:] = 0
    elif kind in ("positive", "negative"):
        amplitude[[0, 2]] = 0
        omega[:] = np.pi/(seconds*1.25)
        amplitude[1] = half[1]*0.95
        phase[:] = -omega*seconds/2
        if kind == "negative":
            phase += np.pi
    elif kind == "neutral":
        amplitude[1] = 0
    elif kind == "switch":
        amplitude[[0, 2]] = 0
        omega[1] = 2*np.pi/seconds
        amplitude[1] = half[1]*0.95
        phase[1] = 0
    amplitude *= min(1, cfg["max_speed"]/max(np.linalg.norm(amplitude*omega), 1e-12),
                     cfg["max_acceleration"]/max(np.linalg.norm(amplitude*omega**2), 1e-12))
    center = np.asarray(cfg["center"])+np.clip(center-cfg["center"], -half+amplitude, half-amplitude)
    return Trajectory(center, amplitude, omega, phase)


def predict_modes(path, cfg):
    c, k = cfg["collection"], cfg["controller"]
    selector = ModeSelector(k["enter_speed"], k["exit_speed"])
    stride = c["physics_hz"]//c["fps"]
    counts = np.zeros(3, dtype=int)
    vy = path.sample(np.arange(round(c["seconds"]*c["physics_hz"]))/c["physics_hz"])[1][:, 1]
    for step, velocity in enumerate(vy):
        mode = selector.update(velocity)
        if step % stride == 0:
            counts[MODES.index(mode)] += 1
    return counts


def choose_trajectory(cfg, attempt, counts):
    rng = np.random.default_rng(np.random.SeedSequence([cfg["collection"]["seed"], attempt]))
    duration = cfg["collection"]["seconds"]+cfg["collection"]["future_frames"]/cfg["collection"]["fps"]
    best, best_score = None, float("inf")
    for _ in range(cfg["collection"]["candidates"]):
        path = generate(cfg["trajectory"], rng, duration)
        predicted = predict_modes(path, cfg)
        total = np.asarray(counts)+predicted
        score = np.sum((total/total.sum()-1/3)**2)
        if score < best_score:
            best, best_score = path, score
    return best
