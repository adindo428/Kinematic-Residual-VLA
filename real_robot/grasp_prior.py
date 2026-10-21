"""Offline blue-tape grasp prior fitting. No camera or actuator access."""
import argparse
import json
from pathlib import Path

import numpy as np

from .geometry import validate_transform
from .vx300s_kinematics import forward, MOTOR_IDS

EVENTS = ('pregrasp', 'before_close', 'closed', 'lifted', 'above_box', 'release', 'placed')
MASTER_INDICES = np.array(MOTOR_IDS)-1


def tape_frame(position, heading_deg):
    """X is known rolling direction, Y the axle, Z gravity-up; no spin phase."""
    p = np.asarray(position, float)
    if p.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(heading_deg):
        raise ValueError('Need a finite tape position and heading')
    a = np.deg2rad(heading_deg)
    t = np.eye(4)
    t[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    t[:3, 3] = p
    return t


def follower_pose(raw_ticks, tcp_offset):
    """Same encoder/model convention as the current right-arm tracker.

    This deliberately does not use LeRobot's calibrated angle conventions.
    """
    raw = np.asarray(raw_ticks, dtype=np.int64)
    if raw.shape != (9,):
        raise ValueError('Expected all nine follower encoder values in motor-ID order')
    raw = (raw + 2**31) % 2**32 - 2**31
    q = (raw[MASTER_INDICES]-2048)*2*np.pi/4096
    return q, forward(q, tcp_offset)


def fresh_observation(record, rig_sha256, now, max_age=.20):
    """Accept only the tracker's current, identity-associated base-frame proxy."""
    if record.get('schema') != 'front_side_observation_v1':
        raise ValueError('Perception has no valid observation yet')
    if not record.get('running'):
        raise ValueError('Perception stopped; start perception with --record-rgbd and leave it running')
    if record.get('rig_sha256') != rig_sha256:
        raise ValueError('Perception rig mismatch; use the same configuration for both commands')
    if record.get('output_frame') != 'right_base_estimate':
        raise ValueError('Perception output frame is not right_base_estimate')
    oldest = (record.get('timing') or {}).get('oldest_host_arrival_monotonic_s')
    if oldest is None or not np.isfinite(oldest) or not 0 <= now-oldest <= max_age:
        age = None if oldest is None else now-oldest
        raise ValueError(f'Perception frame is stale or has an invalid timestamp (age {age!r} s; limit {max_age:.2f} s)')
    if record.get('stale', True):
        raise ValueError(f'Perception marked stale (frame age {now-oldest:.3f} s)')
    fusion = record.get('fusion') or {}
    if (fusion.get('stale', True) or not fusion.get('identity_recent')
            or fusion.get('output_frame') != 'right_base_estimate' or not fusion.get('sources')):
        raise ValueError('No current associated tape proxy')
    p = np.asarray(fusion.get('position_output_m'), float)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError('Invalid tape position')
    return p


def pose_summary(poses):
    """Select a demonstrated SE(3) medoid, retaining spread and every sample."""
    matrices = np.array([validate_transform(p) for p in poses])
    if not len(matrices):
        raise ValueError('No poses to summarize')
    translations = matrices[:, :3, 3]
    rotations = matrices[:, :3, :3]
    dots = np.einsum('aij,bij->ab', rotations, rotations)
    angles = np.arccos(np.clip((dots-1)/2, -1, 1))
    distances = np.linalg.norm(translations[:, None]-translations[None, :], axis=-1)
    index = int(np.argmin(np.sum(distances + .05*angles, axis=1)))
    return {'T': matrices[index].tolist(), 'representative_index': index,
            'sample_count': len(matrices),
            'translation_deviation_m': distances[index].tolist(),
            'rotation_deviation_deg': np.rad2deg(angles[index]).tolist(),
            'samples': matrices.tolist()}


def fit_session(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema') != 'blue_tape_grasp_capture_v1':
        raise ValueError('Not a blue-tape grasp recording')
    episodes = [json.loads(p.read_text()) for p in sorted((root/'episodes').glob('*.json'))]
    accepted = [e for e in episodes if e.get('outcome') == 'success'
                and all(k in e['events'] for k in EVENTS)]
    if not accepted:
        raise ValueError('No complete, operator-confirmed successful episodes')
    # Guard against mixing different taping/heading/TCP definitions across files.
    for e in accepted:
        validate_transform(e['T_base_tape_reference'])
        if e['box_id'] not in ('box_1', 'box_2'):
            raise ValueError('Unknown box location')
    stages = {}
    for name in ('pregrasp', 'before_close', 'closed', 'lifted'):
        poses = [np.linalg.inv(np.asarray(e['T_base_tape_reference'])) @
                 np.asarray(e['events'][name]['sample']['T_base_tcp_candidate']) for e in accepted]
        stages[name] = pose_summary(poses)
        stages[name]['gripper_percent_samples'] = [
            e['events'][name]['sample']['follower_calibrated'][-1] for e in accepted]
        stages[name]['gripper_raw_tick_samples'] = [
            e['events'][name]['sample']['follower_raw_ticks'][-1] for e in accepted]
        stages[name]['gripper_current_a_samples'] = [
            e['events'][name]['sample']['currents_a'][-1] for e in accepted]
    boxes = {}
    for box in ('box_1', 'box_2'):
        selected = [e for e in accepted if e['box_id'] == box]
        boxes[box] = {'sample_count': len(selected), 'geometry_registered': False,
                      'interior_bounds_m': None, 'rim_height_m': None}
        for name in ('above_box', 'release'):
            boxes[box][name] = None if not selected else pose_summary([
                e['events'][name]['sample']['T_base_tcp_candidate'] for e in selected])
    return {'schema': 'blue_tape_grasp_prior_candidate_v1', 'object': 'blue_tape',
            'posture': 'upright_on_curved_edge', 'source_session': str(root.resolve()),
            'source_episode_ids': [e['episode_id'] for e in accepted],
            'successful_episodes': len(accepted), 'excluded_episodes': len(episodes)-len(accepted),
            'rig_sha256': manifest['rig_sha256'], 'tcp_offset_m': manifest['tcp_offset_m'],
            'frame_convention': 'X rolling heading, Y axle, Z up; origin frozen visible-surface proxy',
            'grasp_stages_in_tape_reference': stages, 'box_locations_in_base': boxes,
            'validated_for_execution': False, 'success_labels': 'operator confirmed',
            'notes': ['TCP, kinematics and absolute camera-to-base calibration remain provisional.',
                      'Gripper percentages are software-calibrated values, not opening widths.',
                      'Samples are stationary teaching snapshots, not dynamic interception validation.',
                      'Box geometry and held-object collision clearance require registration.',
                      'Fitted spread is descriptive; it is not an execution tolerance.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', help='New output file; default: SESSION/grasp-prior-candidate.json')
    args = parser.parse_args(argv)
    result = fit_session(args.session)
    output = Path(args.output) if args.output else Path(args.session)/'grasp-prior-candidate.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"Candidate from {result['successful_episodes']} successful episodes: {output}")
    for stage, summary in result['grasp_stages_in_tape_reference'].items():
        print(f"{stage}: maximum translation deviation {max(summary['translation_deviation_m'])*1000:.1f} mm; "
              f"rotation {max(summary['rotation_deviation_deg']):.1f} deg")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
