#!/usr/bin/env python3
"""Derive the 20-action study from frozen data without rescanning DOM."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.dom_training import load_family
from utils.dom_experiment import (atomic_json, digest, file_digest, source_digest,
                                 PREDICTION_HORIZON, EXECUTION_HORIZON)


def derive_family(parent_root, output_root):
    parent_root, output_root = Path(parent_root).resolve(), Path(output_root).resolve()
    parent = load_family(parent_root, check_source=False)
    initialization = Path(parent['initialization']).resolve()
    if file_digest(initialization/'model.safetensors') != parent['weights_sha256']:
        raise ValueError('Pinned initialization changed')
    if not (parent_root/'phase_cache').is_dir():
        raise FileNotFoundError('The frozen phase-prefix cache is required')
    output_root.mkdir(parents=True, exist_ok=False)
    # Prefix gate states depend on past observations/commands, not chunk length.
    # Keep the parent's frozen cache and initialization intact and reuse them.
    (output_root/'phase_cache').symlink_to(parent_root/'phase_cache', target_is_directory=True)
    (output_root/'initialization').symlink_to(initialization, target_is_directory=True)
    # split.json has its own top-level checksum in older manifests and may
    # therefore be absent from the general artifact list.
    for name in sorted(set(parent['artifacts']) | {'split.json'}):
        if name.startswith('initialization/'):
            continue
        dest = output_root/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent_root/name, dest)
    plan = parent_root/'evaluation_plan.json'
    if plan.exists():
        shutil.copy2(plan, output_root/plan.name)
    provenance = {
        'parent_family': str(parent_root), 'parent_fingerprint': parent['fingerprint'],
        'parent_source_sha256': parent['source_sha256'],
        'old_prediction_horizon': parent.get('prediction_horizon', 50),
        'prediction_horizon': PREDICTION_HORIZON, 'execution_horizon': EXECUTION_HORIZON,
        'reason': 'User requested matching the paper with 20 predicted and executed actions.',
        'reused_artifacts': parent['artifacts'],
    }
    atomic_json(output_root/'horizon_change.json', provenance)
    manifest = {k: v for k, v in parent.items() if k != 'fingerprint'}
    manifest.update(prediction_horizon=PREDICTION_HORIZON, execution_horizon=EXECUTION_HORIZON,
                    source_sha256=source_digest(Path(__file__).resolve().parents[1]),
                    derived_from=provenance)
    manifest['artifacts'] = {**parent['artifacts'], 'split.json': parent['split_sha256'],
                             'horizon_change.json': file_digest(output_root/'horizon_change.json')}
    manifest['fingerprint'] = digest(manifest)
    atomic_json(output_root/'manifest.json', manifest)
    # Verify copied metadata and the new source before any GPU job can start.
    load_family(output_root)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    result = derive_family(args.parent_root, args.output_root)
    print(json.dumps({'fingerprint': result['fingerprint'],
                      'prediction_horizon': result['prediction_horizon'],
                      'execution_horizon': result['execution_horizon'],
                      'counts': result['counts']}, indent=2))
