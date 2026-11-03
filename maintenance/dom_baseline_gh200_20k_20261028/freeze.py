"""Freeze a fresh no-velocity DOM baseline from the preserved initialization."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
FAMILY = ROOT / 'family'
sys.path.insert(0, str(SOURCE))
from utils.dom_experiment import atomic_json, digest, file_digest, source_digest

def main():
    assert not (FAMILY / 'manifest.json').exists(), 'Do not edit a frozen attempt'
    parent = json.loads((FAMILY / 'parent-manifest.json').read_text())
    assert digest({k: v for k, v in parent.items() if k != 'fingerprint'}) == parent['fingerprint']
    assert source_digest(SOURCE) == parent['source_sha256'] == 'f991015a229476400e0fc772177af33759d1bc3f23901381e2f83c7faaeab921'
    assert file_digest(FAMILY / 'initialization/model.safetensors') == parent['weights_sha256']
    assert file_digest(FAMILY / 'initialization/config.json') == parent['artifacts']['initialization/config.json']
    split = json.loads((FAMILY / 'split.json').read_text())['splits']
    report = json.loads((FAMILY / 'subset-report.json').read_text())
    provenance = json.loads((FAMILY / 'subset-data-provenance.json').read_text())
    records = json.loads((FAMILY / 'episodes.json').read_text())
    lengths = {int(r['episode_index']): int(r['length']) for r in records}
    assert len(split['train']) == 20000 and len(split['validation']) == 2015 and len(split['test']) == 2011
    assert len(set(sum(split.values(), []))) == 24026
    assert sum(lengths[i] for i in split['train']) == report['train_anchors'] == 2406773
    verified = json.loads((ROOT / 'verified-data.json').read_text())
    assert verified['complete'] and not verified['errors'] and verified['completed_files'] == 66045
    assert verified['phase_files_verified'] == 22015
    manifest = {k: parent[k] for k in ['dataset', 'dataset_revision', 'model_revision', 'normalization',
        'phase_config', 'prediction_horizon', 'execution_horizon', 'fps', 'weights_sha256', 'velocity_std']}
    manifest.update(version=2, variant='dom100', kr=False, velocity_conditioning=False,
        dataset_root=verified['dataset_root'], initialization=str(FAMILY / 'initialization'),
        source_sha256=source_digest(SOURCE), split_sha256=file_digest(FAMILY / 'split.json'),
        verified_data_sha256=file_digest(ROOT / 'verified-data.json'),
        counts={k: len(v) for k, v in split.items()}, episodes=24026,
        frames=sum(lengths.values()), train_frames=report['train_anchors'],
        velocity_mean=provenance['velocity_mean'],
        initialization_provenance={'kind': 'same released DOM initialization as the historical matched family',
            'parent_manifest': parent['fingerprint'], 'training_checkpoint_resumed': False,
            'prior_DOM_pretraining': True, 'claim': '20k fine-tuning episodes; not 20k-only lifetime exposure'},
        subset_provenance={'source': 'cross-vla-krfm-20k-20261020', 'parent_split_sha256': report['parent_split_sha256'],
            'task_quotas': report['task_quotas'], 'held_out_membership_preserved': True},
        training_authorization='User requested missing plain DOM baseline on two-node GH200 setup, 2026-10-28; ten epochs follows current 20k campaign',
        profile_reference={'job_id': '3381859', 'variant': 'domv100', 'world_size': 8,
            'microbatch': 44, 'global_batch': 352, 'samples_per_second': 373.45165815190313,
            'use': 'candidate runtime only; fresh dom100 readiness required'})
    artifacts = [p for p in FAMILY.iterdir() if p.is_file() and p.suffix in ('.json', '.py', '.sh')]
    artifacts += [FAMILY / 'initialization/config.json']
    manifest['artifacts'] = {str(p.relative_to(FAMILY)): file_digest(p) for p in sorted(artifacts)}
    manifest['fingerprint'] = digest(manifest)
    atomic_json(FAMILY / 'manifest.json', manifest)
    runtime = {'microbatch': 44, 'workers': 4, 'prefetch': 2, 'pin_memory': False,
               'fused': False, 'sampling': 'frames', 'precision': 'bf16'}
    selected = {'manifest': manifest['fingerprint'], 'runtime': runtime, 'world_size': 8,
                'node_count': 2, 'gpus_per_node': 4, 'global_effective_batch': 352,
                'approval': 'candidate; full same-attempt dom100 readiness required'}
    selected['fingerprint'] = digest(selected)
    atomic_json(FAMILY / 'selected-runtime.json', selected)
    recipe = {'manifest': manifest['fingerprint'], 'epochs': 10, 'effective_batch': 352,
        'lr': 3e-5, 'precision': 'bf16', 'sampling': 'frames', 'scheduler': 'cosine',
        'warmup_updates': 1000, 'seed': 42, 'early_stopping': {'enabled': False},
        'runtime_fingerprint': digest(runtime), 'supported_world_sizes': [8],
        'simulator_evaluation_mode': 'disabled_on_deltaai', 'velocity_conditioning': False, 'kr': False,
        'scheduler_decay_steps': 10 * ((report['train_anchors'] + 351) // 352),
        'initialization': 'fresh from preserved released DOM initialization; no domv100 run state'}
    recipe['fingerprint'] = digest(recipe)
    atomic_json(FAMILY / 'recipe.json', recipe)
    implementation = [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py', '.sh', '.sbatch')]
    lock = {'files': {p.name: file_digest(p) for p in sorted(implementation)},
            'manifest': manifest['fingerprint'], 'recipe': recipe['fingerprint'],
            'runtime': selected['fingerprint']}
    atomic_json(ROOT / 'implementation-lock.json', lock)
    print(json.dumps({'manifest': manifest['fingerprint'], 'recipe': recipe['fingerprint'],
        'train_episodes': 20000, 'epochs': 10, 'updates': recipe['scheduler_decay_steps'],
        'global_batch': 352, 'variant': 'dom100', 'velocity': False, 'kr': False}), flush=True)

if __name__ == '__main__':
    main()
