"""Read-only source/initialization/split checks; this is not GPU approval."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
FAMILY = ROOT / 'family'
sys.path.insert(0, str(ROOT / 'source'))
from utils.dom_experiment import digest, file_digest, source_digest, VARIANTS

parent = json.loads((FAMILY / 'parent-manifest.json').read_text())
assert digest({k: v for k, v in parent.items() if k != 'fingerprint'}) == parent['fingerprint']
assert source_digest(ROOT / 'source') == parent['source_sha256'] == 'f991015a229476400e0fc772177af33759d1bc3f23901381e2f83c7faaeab921'
assert file_digest(FAMILY / 'initialization/model.safetensors') == parent['weights_sha256']
assert file_digest(FAMILY / 'initialization/config.json') == parent['artifacts']['initialization/config.json']
assert VARIANTS['dom100'] == (False, False, 'constant')
split = json.loads((FAMILY / 'split.json').read_text())['splits']
records = json.loads((FAMILY / 'episodes.json').read_text())
lengths = {int(r['episode_index']): int(r['length']) for r in records}
assert {k: len(v) for k, v in split.items()} == {'train': 20000, 'validation': 2015, 'test': 2011}
assert len(set(sum(split.values(), []))) == 24026
assert sum(lengths[i] for i in split['train']) == 2406773
assert set(json.loads((FAMILY / 'pilot_episodes.json').read_text())).issubset(split['train'])
anchors = json.loads((FAMILY / 'validation_anchors.json').read_text())
assert 0 <= min(anchors) <= max(anchors) < sum(lengths[i] for i in split['validation'])
print(json.dumps({'preparation_checks_passed': True, 'gpu_readiness_approved': False,
    'training_episodes': 20000, 'training_frames': 2406773, 'initialization_sha256': parent['weights_sha256'],
    'source_sha256': parent['source_sha256'], 'velocity_conditioning': False, 'kr': False}), flush=True)
