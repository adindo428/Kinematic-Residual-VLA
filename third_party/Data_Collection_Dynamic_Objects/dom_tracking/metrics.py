import numpy as np

from .trajectory import MODES


def episode_metrics(controls, frames, cfg):
    if not controls or not frames:
        raise ValueError("Empty episode")
    errors = np.array([np.linalg.norm(c["command"]["position_error"]) for c in controls])
    modes = np.array([c["command"]["mode"] for c in controls])
    angles = np.array([c["command"]["orientation_error"] for c in controls])
    times = np.array([c["timestamp"] for c in controls])
    if not all(np.isfinite(x).all() for x in (errors, angles, times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid/nonfinite control timeline or tracking errors")
    switches = np.r_[True, modes[1:] != modes[:-1]]
    last_switch = np.maximum.accumulate(np.where(switches, times, -np.inf))
    settled = times-last_switch >= cfg["quality"]["transition_seconds"]
    maximum_angle = float(np.rad2deg(angles[settled].max())) if settled.any() else None
    frame_modes = [f["mode"] for f in frames]
    per_mode = {str(m): float(np.sqrt(np.mean(errors[modes == m]**2))) if np.any(modes == m) else None for m in MODES}
    velocities = np.array([f["object_velocity"] for f in frames])
    visible = {name: float(np.mean([f["cameras"][name]["target_in_frame"] for f in frames])) for name in frames[0]["cameras"]}
    result = dict(rmse=float(np.sqrt(np.mean(errors**2))), max_error=float(errors.max()),
                  per_mode_rmse=per_mode, max_settled_orientation_degrees=maximum_angle,
                  per_mode_control_samples={str(m): int(np.sum(modes == m)) for m in MODES},
                  settled_samples=int(settled.sum()), switches=int(switches.sum()-1),
                  mode_counts=[frame_modes.count(m) for m in MODES],
                  saturation_fraction=float(np.mean([c["command"]["saturated"] for c in controls])),
                  max_contact_force=max(c["contact_force"] for c in controls),
                  joint_limit_violation=any(c["joint_limit_violation"] for c in controls),
                  workspace_violation=any(c.get("workspace_violation", False) for c in controls),
                  visible_fraction=visible,
                  velocity_histograms={name: histogram(values, cfg["trajectory"]["max_speed"], name == "speed")
                                       for name, values in zip(("vx", "vy", "vz", "speed"), [*velocities.T, np.linalg.norm(velocities, axis=1)])})
    q = cfg["quality"]
    reasons = []
    for key in ("rmse", "max_error"):
        if result[key] > q[key]:
            reasons.append(key)
    if maximum_angle is None or maximum_angle > q["orientation_degrees"]:
        reasons.append("orientation")
    if result["max_contact_force"] > q["contact_force"]:
        reasons.append("collision")
    if result["joint_limit_violation"]:
        reasons.append("joint_limit")
    if result["workspace_violation"]:
        reasons.append("workspace")
    if min(visible.values()) < q["min_visible_fraction"]:
        reasons.append("target_out_of_frame")
    result["rejection_reasons"] = reasons
    return result


def histogram(values, bound, nonnegative=False):
    counts, edges = np.histogram(values, bins=np.linspace(0 if nonnegative else -bound, bound, 13))
    return dict(counts=counts.tolist(), edges=edges.tolist())


def run_report(store, cfg):
    import json
    from collections import Counter
    episodes = [json.loads((p/"episode.json").read_text()) for p in store.completed()]
    counts = np.sum([e["metrics"]["mode_counts"] for e in episodes], axis=0) if episodes else np.zeros(3, int)
    freq = counts/max(1, counts.sum())
    histograms = {}
    for e in episodes:
        for key, hist in e["metrics"]["velocity_histograms"].items():
            if key not in histograms:
                histograms[key] = {"counts": np.zeros(len(hist["counts"]), int), "edges": hist["edges"]}
            histograms[key]["counts"] += hist["counts"]
    per_mode = {}
    for mode in MODES:
        key = str(mode)
        weights = [e["metrics"]["per_mode_control_samples"][key] for e in episodes]
        denominator = sum(weights)
        per_mode[key] = float(np.sqrt(sum(w*(e["metrics"]["per_mode_rmse"][key] or 0)**2
                                         for w, e in zip(weights, episodes))/denominator)) if denominator else None
    rejected = [json.loads(p.read_text()) for p in (store.root/"rejected").glob("*/episode.json")]
    reasons = Counter(reason for e in rejected for reason in e["metrics"]["rejection_reasons"])
    return dict(episodes=len(episodes), requested=cfg["collection"]["episodes"], mode_order=list(MODES),
                mode_counts=counts.tolist(), mode_frequency=freq.tolist(),
                balanced=bool(len(episodes) and np.max(abs(freq-1/3)) <= cfg["collection"]["balance_tolerance"]),
                complete=len(episodes) == cfg["collection"]["episodes"],
                rmse=float(np.sqrt(np.mean([e["metrics"]["rmse"]**2 for e in episodes]))) if episodes else None,
                per_mode_rmse=per_mode,
                max_error=max((e["metrics"]["max_error"] for e in episodes), default=None),
                saturation_fraction=float(np.mean([e["metrics"]["saturation_fraction"] for e in episodes])) if episodes else None,
                rejection_reasons=dict(reasons),
                velocity_histograms=histograms,
                rejected=len(list((store.root/"rejected").iterdir())),
                interrupted=len(list((store.root/"interrupted").iterdir())))
