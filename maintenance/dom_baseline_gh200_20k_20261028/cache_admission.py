"""Reuse this attempt's completed byte verification; preserve per-access checks."""
import json
from pathlib import Path

def install(root):
    root = Path(root)
    from utils.dom_experiment import file_digest
    manifest = json.loads((root / 'manifest.json').read_text())
    receipt = root.parent / 'verified-data.json'
    assert file_digest(receipt) == manifest['verified_data_sha256']
    verified = json.loads(receipt.read_text())
    assert verified['complete'] and verified['state'] == 'complete' and not verified['errors']
    assert verified['completed_files'] == verified['total_files'] == 66045
    assert verified['dataset_root'] == manifest['dataset_root']
    assert verified['dataset_revision'] == manifest['dataset_revision']
    split = json.loads((root / 'split.json').read_text())['splits']
    allowed = set(split['train'] + split['validation'])
    import utils.datasets as datasets
    def admit(repo_id, data_root, episodes, required_features, revision):
        assert repo_id == 'hzxie/DOM' and Path(data_root).resolve() == Path(manifest['dataset_root']).resolve()
        assert revision == manifest['dataset_revision']
        assert episodes is not None and set(episodes).issubset(allowed)
        assert set(required_features).issubset({'observation.images.opst_cam', 'observation.images.wrist_cam',
            'observation.state', 'observation.environment_state', 'action'})
    datasets._ensure_lerobot_subset_cached = admit
