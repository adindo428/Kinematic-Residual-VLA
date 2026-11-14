"""Record explicit acceptance of an existing transform without relabeling validation."""
import hashlib
import json
from pathlib import Path


TRANSFORM_KEYS = ('front_serial', 'side_serial', 'output_frame', 'T_front_side',
                  'T_base_front_candidate', 'T_base_side_candidate')
ACCEPTANCE_PATH = Path(__file__).with_name('base_transform_acceptance.json')


def transform_digest(rig):
    values = {key: rig[key] for key in TRANSFORM_KEYS}
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def accepted_transform(rig):
    if not ACCEPTANCE_PATH.exists():
        return None
    record = json.loads(ACCEPTANCE_PATH.read_text())
    if (record.get('schema') != 'krvla_operator_transform_acceptance_v1'
            or record.get('operator_accepted') is not True
            or rig.get('operator_confirmed_front_and_base_unchanged') is not True):
        return None
    try:
        return record if transform_digest(rig) == record['transform_sha256'] else None
    except (KeyError, TypeError, ValueError):
        return None


def dataset_acceptance(root, cameras):
    """Bind acceptance to every fixed-camera record and the exact metadata file."""
    accepted = None
    if not cameras:
        return None
    for row in cameras:
        fixed = [c for c in row['cameras'] if c['source_role'] != 'wrist']
        if not fixed:
            return None
        for camera in fixed:
            record = accepted_transform(camera['calibration'])
            if record is None:
                return None
            accepted = record
    return {**accepted, 'episodes': len(cameras),
            'camera_metadata_sha256': hashlib.sha256((Path(root)/'meta/camera.jsonl').read_bytes()).hexdigest()}
