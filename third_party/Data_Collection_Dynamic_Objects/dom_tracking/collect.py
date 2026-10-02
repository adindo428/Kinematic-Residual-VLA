"""Simulation-independent orchestration; the backend owns all simulator interaction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import run_fingerprint
from .controller import Expert
from .metrics import episode_metrics, run_report
from .storage import RunStore, VideoWriter, atomic_json, write_records
from .trajectory import choose_trajectory

CAMERAS = ("wrist_cam", "opst_cam", "side_cam")


def smoke_suite(cfg, backend, layout, software):
    import copy
    root = Path(cfg["collection"]["output"])
    results = {}
    required = {"stationary": {0}, "positive": {-45}, "negative": {45}, "neutral": {0}, "switch": {-45, 0, 45}}
    for kind, expected_modes in required.items():
        test_cfg = copy.deepcopy(cfg)
        test_cfg["collection"].update(purpose="smoke", episodes=1, seconds=10 if kind == "switch" else 2,
                                      max_attempts=3, output=str(root/kind))
        test_cfg["trajectory"]["kind"] = kind
        backend.cfg = test_cfg
        try:
            report = collect(test_cfg, backend, layout, software)
            observed = {m for m, n in zip(report["mode_order"], report["mode_counts"]) if n}
            if not expected_modes.issubset(observed):
                raise ValueError(f"Missing expected modes {expected_modes-observed}")
            results[kind] = dict(passed=True, report=report)
        except ValueError as exc:
            results[kind] = dict(passed=False, error=str(exc))
        atomic_json(root/"smoke_suite.json", results)
    if not all(r["passed"] for r in results.values()):
        raise ValueError(f"Smoke suite failed; inspect {root/'smoke_suite.json'}")
    return results


def collect(cfg, backend, layout, software):
    c = cfg["collection"]
    provenance = dict(config=cfg, config_sha256=run_fingerprint(cfg), layout=layout, software=software,
                      schema_version=1, joint_names=backend.robot.joint_names,
                      action_semantics="future_observed_pose_base_frame", camera_names=list(CAMERAS))
    store = RunStore(c["output"], provenance)
    try:
        audit = json.loads(Path(cfg["assets"]["manifest"]).read_text())
        if audit["audit_sha256"] != layout["audit_sha256"]:
            raise ValueError("Asset manifest changed since scene inspection")
        atomic_json(store.root/"asset_manifest.json", audit)
        count = int(c["seconds"]*c["fps"])
        stride = c["physics_hz"]//c["fps"]
        while len(store.completed()) < c["episodes"] and store.state["next_attempt"] < c["max_attempts"]:
            totals = np.zeros(3, int)
            for p in store.completed():
                totals += json.loads((p/"episode.json").read_text())["metrics"]["mode_counts"]
            attempt, tmp = store.begin()
            path = choose_trajectory(cfg, attempt, totals)
            metadata = dict(attempt=attempt, trajectory=path.serialize(), seed=c["seed"], purpose=c["purpose"])
            frames, controls, writers = [], [], {}
            try:
                backend.reset(path)
                expert = Expert(cfg["controller"], 1/c["physics_hz"])
                # Hold the initial target position while settling, with zero velocity feedback.
                for step in range(round(c["warmup_seconds"]*c["physics_hz"])):
                    state = backend.state()
                    target = backend.target()
                    target["velocity"] = np.zeros(3)
                    cmd = expert.command(state, target, backend.jacobian(), backend.limits, backend.velocity_limits)
                    backend.apply(cmd)
                    backend.step(0, render=step % stride == 0)
                expert = Expert(cfg["controller"], 1/c["physics_hz"])
                backend.capture(-1/c["fps"])  # discard renderer warm-up image
                for name in CAMERAS:
                    writers[name] = VideoWriter(tmp/f"{name}.mp4", c["fps"], cfg["camera"]["width"], cfg["camera"]["height"], cfg["camera"]["codec"])
                total_steps = (count+c["future_frames"])*stride
                for step in range(total_steps):
                    timestamp = step/c["physics_hz"]
                    state = backend.state()
                    target = backend.target()
                    command = expert.command(state, target, backend.jacobian(), backend.limits, backend.velocity_limits)
                    contact = backend.contact_force()
                    joints = state["joint_position"]
                    violation = bool(np.any(joints < backend.limits[:, 0]) or np.any(joints > backend.limits[:, 1]))
                    controls.append(dict(timestamp=timestamp, control_step=step, state=state, command=command,
                                         target=target, object_position=backend.target_p.copy(),
                                         object_quaternion=[1., 0., 0., 0.],
                                         object_velocity=backend.target_v.copy(), object_acceleration=backend.target_a.copy(),
                                         workspace_violation=bool(np.any(state["ee_position"] < target["workspace_lower"]-1e-4) or
                                                                  np.any(state["ee_position"] > target["workspace_upper"]+1e-4)),
                                         contact_force=contact, joint_limit_violation=violation))
                    if step % stride == 0:
                        frame_index = step//stride
                        images, cameras = backend.capture(timestamp)
                        frame = dict(timestamp=timestamp, frame_index=frame_index, control_step=step,
                                     ee_position=state["ee_position"], ee_quaternion=state["ee_quaternion"],
                                     object_position=backend.target_p.copy(), object_quaternion=[1., 0., 0., 0.],
                                     object_velocity=backend.target_v.copy(), object_acceleration=backend.target_a.copy(),
                                     mode=command["mode"], cameras=cameras)
                        frames.append(frame)
                        if frame_index < count:
                            for name, image in images.items():
                                writers[name].add(image)
                    backend.apply(command)
                    backend.step((step+1)/c["physics_hz"])
                metrics = episode_metrics(controls[:count*stride], frames[:count], cfg)
                # Continuation must also be physically valid, because it supplies action labels.
                if any(x["contact_force"] > cfg["quality"]["contact_force"] or x["joint_limit_violation"] or x["workspace_violation"] for x in controls[count*stride:]):
                    metrics["rejection_reasons"].append("invalid_continuation")
                metadata.update(metrics=metrics, frame_count=count, continuation_frames=c["future_frames"])
                accepted = not metrics["rejection_reasons"]
            except (ValueError, RuntimeError) as exc:
                metadata.update(error=f"{type(exc).__name__}: {exc}", metrics={"rejection_reasons": [str(exc)]})
                accepted = False
            finally:
                for writer in writers.values():
                    writer.close()
            write_records(tmp/"controls", controls, c["raw_format"])
            write_records(tmp/"frames", frames, c["raw_format"])
            store.commit(tmp, metadata, accepted)
            print(json.dumps(dict(attempt=attempt, accepted=accepted, episodes=len(store.completed()),
                                  reasons=metadata["metrics"]["rejection_reasons"])), flush=True)
            atomic_json(store.root/"report.json", run_report(store, cfg))
        report = run_report(store, cfg)
        atomic_json(store.root/"report.json", report)
        if not report["complete"] or (c["purpose"] == "training" and not report["balanced"]):
            raise ValueError("Collection did not meet episode/balance targets; inspect report.json. Gates were not relaxed.")
        return report
    finally:
        store.close()
