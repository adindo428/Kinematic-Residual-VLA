"""Fit and apply an RGB-referenced, camera-specific software depth correction.

No hardware access. RGB board poses are the assumed metric reference. The
correction is empirical over the recorded workspace, not a factory calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sample_board(reference, archive):
    detection = reference['detection']
    if not detection.get('valid'):
        raise ValueError('Require a valid RGB board pose')
    timing = detection['timing']
    if timing.get('depth_correction'):
        raise ValueError('Fit requires uncorrected depth observations')
    with np.load(archive) as data:
        depth = data['depth_m']
    pose = np.asarray(detection['T_camera_board'], float)
    normal = pose[:3, 2]
    rays = np.c_[detection['normalized_pixels'], np.ones(len(detection['ids']))]
    expected = (normal @ pose[:3, 3]) / (rays @ normal)
    samples = []; missing = []
    for i, corner in enumerate(detection['ids']):
        x, y = np.rint(detection['pixels'][i]).astype(int)
        patch = depth[max(0,y-2):y+3, max(0,x-2):x+3]
        good = patch[np.isfinite(patch) & (patch > 0)]
        if len(good) < 13:
            missing.append(corner)
            continue
        samples.append(dict(corner_id=corner, pixel=detection['pixels'][i],
                            raw_z_m=float(np.median(good)), rgb_reference_z_m=float(expected[i]),
                            ray_norm=float(np.linalg.norm(rays[i]))))
    if len(samples) < 20:
        raise ValueError('Need at least 20 corners with usable depth')
    return dict(samples=samples, missing_depth_corner_ids=missing, timing=timing,
                archive=str(Path(archive).resolve()),
                archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),
                reference_sha256=digest(reference))


def check_profile(model, timing):
    if model.get('schema') != 'krvla_rgb_depth_correction_v1':
        raise ValueError('Unknown depth-correction schema')
    if timing.get('depth_correction'):
        raise ValueError('Depth correction already applied; refusing double correction')
    if timing['camera_serial'] != model['camera_serial']:
        raise ValueError('Depth correction belongs to another camera')
    expected = model['profile']
    if timing.get('distortion_model') != expected['distortion_model']:
        raise ValueError('Depth correction camera distortion model differs')
    for key, value in expected['intrinsics'].items():
        actual = timing['intrinsics'].get(key)
        matches = (np.isclose(actual, value, rtol=.005, atol=0) if key in ('fx', 'fy')
                   else actual == value)
        if not matches:
            raise ValueError(f'Depth correction camera intrinsics differ: {key}')
    b = model['inverse_depth_offset_per_m']
    lo, hi = model['raw_depth_domain_m']
    if not np.isfinite([b, lo, hi]).all() or not 0 < lo < hi or min(1/lo+b, 1/hi+b) <= 0:
        raise ValueError('Invalid depth-correction parameters')
    a, c = model.get('application_depth_range_m', [lo, hi])
    if not np.isfinite([a, c]).all() or not 0 < a <= lo < hi <= c or min(1/a+b, 1/c+b) <= 0:
        raise ValueError('Invalid depth-correction application range')


def apply_depth(depth, model, timing):
    """Correct aligned Z, distinguishing fitted support from explicit extrapolation."""
    check_profile(model, timing)
    raw = np.asarray(depth, float)
    lo, hi = model['raw_depth_domain_m']
    application_lo, application_hi = model.get('application_depth_range_m', [lo, hi])
    valid = np.isfinite(raw) & (raw > 0)
    supported = valid & (raw >= application_lo) & (raw <= application_hi)
    extrapolated = supported & ((raw < lo) | (raw > hi))
    result = np.zeros_like(raw)
    result[supported] = 1 / (1/raw[supported] + model['inverse_depth_offset_per_m'])
    metadata = dict(model_sha256=digest(model), schema=model['schema'],
                    inverse_depth_offset_per_m=model['inverse_depth_offset_per_m'],
                    raw_depth_domain_m=[lo, hi], applied_pixels=int(supported.sum()),
                    application_depth_range_m=[application_lo, application_hi],
                    extrapolated_pixels=int(extrapolated.sum()),
                    out_of_domain_pixels=int((valid & ~supported).sum()),
                    base_calibration_validated=False)
    return result, {**timing, 'depth_correction': metadata}


def metrics(samples, offset):
    raw = np.array([s['raw_z_m'] for s in samples])
    expected = np.array([s['rgb_reference_z_m'] for s in samples])
    weights = np.array([s['ray_norm'] for s in samples])
    error = (1/(1/raw+offset)-expected)*weights*1000
    return dict(count=len(samples), rms_xyz_mm=float(np.sqrt(np.mean(error**2))),
                median_abs_xyz_mm=float(np.median(np.abs(error))),
                max_abs_xyz_mm=float(np.max(np.abs(error))))


def main(argv=None):
    from scipy.optimize import least_squares
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--camera-check', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads((args.session/'manifest.json').read_text())
    if manifest.get('board_type') != 'chessboard':
        parser.error('Requires saved chessboard RGB-D references')
    references = [json.loads((args.session/f'board_{i}.json').read_text()) for i in (0, 1)]
    diagnostic = json.loads((args.camera_check/'camera_check.json').read_text())
    references.append(diagnostic['frames'][0]['comparison']['native_color_rays']['reference'])
    paths = [args.session/'board_0.npz', args.session/'board_1.npz', args.camera_check/'board.npz']
    groups = [sample_board(ref, path) for ref, path in zip(references, paths)]
    first = groups[0]['samples']
    raw = np.array([s['raw_z_m'] for s in first])
    target = np.array([s['rgb_reference_z_m'] for s in first])
    weights = np.array([s['ray_norm'] for s in first])
    # One parameter. No spatial polynomial or per-corner correction that can
    # simply memorize the board. Fit only the first saved placement.
    fit = least_squares(lambda p: (1/(1/raw+p[0])-target)*weights, [0.], bounds=([-.5], [.5]))
    if not fit.success or abs(fit.x[0]) >= .499:
        raise ValueError('Depth correction did not converge within its bounds')
    offset = float(fit.x[0])
    timing = groups[0]['timing']
    depths = [s['raw_z_m'] for g in groups for s in g['samples']]
    model = dict(schema='krvla_rgb_depth_correction_v1', camera_serial=timing['camera_serial'],
                 formula='corrected_z_m = 1 / (1 / raw_z_m + inverse_depth_offset_per_m)',
                 inverse_depth_offset_per_m=offset, raw_depth_domain_m=[min(depths), max(depths)],
                 profile={k:timing[k] for k in ('intrinsics', 'distortion_model')},
                 reference='User-confirmed 310 x 310 mm, 10 x 10 square board; recorded RGB PnP poses assumed correct',
                 status='empirical_rgb_referenced_depth_correction', base_calibration_validated=False,
                 scene_generalization_validated=False,
                 outside_domain_policy='Return zero/invalid depth; never extrapolate or mix corrected and raw valid measurements',
                 scope='Recorded board workspace and depth range; target-independent accuracy not established',
                 source_fit_archive_sha256=groups[0]['archive_sha256'])
    for group in groups:
        check_profile(model, group['timing'])
        group['before'] = metrics(group['samples'], 0.)
        group['after'] = metrics(group['samples'], offset)
    report = dict(model=model, groups=groups, fit_group=0, evaluation_groups=[1, 2],
                  evaluation_disclosure='Other captures were inspected during exploratory model comparison; these are not blind validation results.',
                  original_archives_modified=False, camera_firmware_modified=False,
                  meets_5mm_maximum_error=all(g['after']['max_abs_xyz_mm'] <= 5 for g in groups),
                  note='Missing depth excluded by availability only; no residual-based outlier removal. No contact or base-frame validation is implied.')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    for name, value in [('depth_correction.json', model), ('depth_fit_report.json', report)]:
        (root/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for i, (path, group) in enumerate(zip(paths, groups)):
        with np.load(path) as archive:
            corrected, metadata = apply_depth(archive['depth_m'], model, group['timing'])
            np.savez(root/f'corrected_board_{i}.npz', bgr=archive['bgr'], depth_m=corrected)
        (root/f'corrected_board_{i}_timing.json').write_text(json.dumps(metadata, indent=2)+'\n')
    # Separate opt-in rig/config: the base transform retains its original
    # unvalidated status. No collected episode or default config is rewritten.
    rig = dict(manifest['rig'])
    if rig.get('base_calibration_validated'):
        raise ValueError('Changing depth requires separate review of existing validated base evidence')
    rig['depth_corrections'] = {**rig.get('depth_corrections', {}), 'front': model}
    rig_path = root/'rig-depth-corrected.json'
    rig_path.write_text(json.dumps(rig, indent=2)+'\n')
    if manifest.get('config_yaml'):
        import yaml
        config = yaml.safe_load(manifest['config_yaml'])
        config['perception']['rig'] = str(rig_path)
        (root/'config-depth-corrected.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(json.dumps(dict(output=str(root), inverse_depth_offset_per_m=offset,
                          captures=[dict(before=g['before'], after=g['after']) for g in groups],
                          base_calibration_validated=False), indent=2))


if __name__ == '__main__':
    main()
