"""Paired fixed-orientation collection, independent of the legacy velocity-to-tilt rule."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from .config import run_fingerprint, fingerprint
from .controller import Expert
from .metrics import episode_metrics
from .storage import RunStore, VideoWriter, atomic_json, write_records
from .trajectory import generate

CAMERAS = ('wrist_cam', 'opst_cam', 'side_cam')
MODES = (-45, 45)


def record_rollout(cfg, backend, path, mode, tmp, metadata):
    c = cfg['collection']
    count = int(c['seconds'] * c['fps'])
    stride = c['physics_hz'] // c['fps']
    tmp.mkdir(parents=True)
    frames, controls, writers = [], [], {}
    try:
        backend.reset(path)
        expert = Expert(cfg["controller"], 1/c["physics_hz"], fixed_mode=mode)
        # Hold the initial target position while settling, with zero velocity feedback.
        for step in range(round(c["warmup_seconds"]*c["physics_hz"])):
            state = backend.state()
            target = backend.target()
            target["velocity"] = np.zeros(3)
            cmd = expert.command(state, target, backend.jacobian(), backend.limits, backend.velocity_limits)
            backend.apply(cmd)
            backend.step(0, render=step % stride == 0)
        expert = Expert(cfg["controller"], 1/c["physics_hz"], fixed_mode=mode)
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
    metadata['fixed_mode'] = mode
    atomic_json(tmp/'episode.json', metadata)
    return metadata


def choose_paired_path(cfg, attempt):
    """Reject paths with inadequate movement in either direction, not based on preferred tilt."""
    rng = np.random.default_rng(np.random.SeedSequence([cfg['collection']['seed'], attempt]))
    duration = cfg['collection']['seconds'] + cfg['collection']['future_frames']/cfg['collection']['fps']
    for _ in range(1000):
        path = generate(cfg['trajectory'], rng, duration)
        velocities = path.sample(np.arange(round(cfg['collection']['seconds']*cfg['collection']['fps']))/cfg['collection']['fps'])[1][:,1]
        if np.mean(velocities > .01) >= .2 and np.mean(velocities < -.01) >= .2:
            return path
    raise ValueError('Could not generate a path with both lateral directions represented')


def paired_report(store, cfg):
    pairs = [json.loads((p/'episode.json').read_text()) for p in store.completed()]
    rejected = [json.loads(p.read_text()) for p in (store.root/'rejected').glob('*/episode.json')]
    return {'complete': len(pairs) == cfg['collection']['episodes']//2,
            'balanced': bool(pairs), 'episodes': 2*len(pairs), 'pairs': len(pairs),
            'requested': cfg['collection']['episodes'], 'mode_order': [-45,45],
            'mode_frequency': [.5,.5] if pairs else [0.,0.],
            'mode_counts': [sum(p['frame_count'] for p in pairs)]*2,
            'rejected_pairs': len(rejected),
            'rejection_reasons': [p['metrics']['rejection_reasons'] for p in rejected],
            'orientation_policy': 'paired fixed modes, independent of position and velocity',
            'split_unit': 'whole trajectory pair',
            'sampling_note': 'Both modes attempted for every candidate. Only pairs where both pass quality gates enter training.'}


def collect_paired(cfg, backend, layout, software):
    c = cfg['collection']
    if c['episodes'] % 2:
        raise ValueError('Paired collection requires an even episode count')
    provenance = dict(config=cfg, config_sha256=run_fingerprint(cfg), layout=layout, software=software,
                      schema_version=2, joint_names=backend.robot.joint_names,
                      action_semantics='future_observed_pose_base_frame', camera_names=list(CAMERAS),
                      orientation_policy='paired_fixed_modes')
    store = RunStore(c['output'], provenance)
    try:
        audit = json.loads(Path(cfg['assets']['manifest']).read_text())
        assert audit['audit_sha256'] == layout['audit_sha256']
        atomic_json(store.root/'asset_manifest.json',audit)
        while 2*len(store.completed()) < c['episodes'] and store.state['next_attempt'] < c['max_attempts']:
            attempt, tmp = store.begin()
            path = choose_paired_path(cfg, attempt)
            pair_group = fingerprint(path.serialize())
            records = []
            # Alternating order avoids always executing one mode first.
            order = MODES if attempt % 2 == 0 else MODES[::-1]
            for mode in order:
                name = 'minus45' if mode < 0 else 'plus45'
                metadata = dict(attempt=attempt, trajectory=path.serialize(), seed=c['seed'],
                                purpose=c['purpose'], pair_group=pair_group,
                                object_usd=layout['object_usd'])
                records.append(record_rollout(cfg,backend,path,mode,tmp/name,metadata))
            reasons = [f"{r['fixed_mode']}:{reason}" for r in records for reason in r['metrics']['rejection_reasons']]
            metadata = dict(attempt=attempt, pair_group=pair_group, trajectory=path.serialize(),
                            object_usd=layout['object_usd'], frame_count=int(c['seconds']*c['fps']),
                            metrics={'rejection_reasons':reasons}, modes={str(r['fixed_mode']):r['metrics'] for r in records})
            store.commit(tmp,metadata,not reasons)
            report = paired_report(store,cfg)
            atomic_json(store.root/'report.json',report)
            print(json.dumps(dict(attempt=attempt,accepted=not reasons,episodes=report['episodes'],reasons=reasons)),flush=True)
        report = paired_report(store,cfg)
        atomic_json(store.root/'report.json',report)
        if not report['complete'] or not report['balanced']:
            raise ValueError('Paired collection incomplete; quality gates have not been relaxed')
        return report
    finally:
        store.close()
