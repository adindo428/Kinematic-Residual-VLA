"""The dataset, gate, prediction and validation must use the same action horizon."""
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from core import dom_training
from utils.dom_experiment import experiment_horizons, digest, file_digest, atomic_json


def test_dataset_requests_twenty_action_targets_and_twenty_gate_values(tmp_path, monkeypatch):
    captured = {}

    class Base:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            self.episodes = kwargs['episodes']
            self.meta = SimpleNamespace(fps=25)
            self.episode_data_index = {'length': torch.tensor([10, 12])}
        def _get_episode(self, episode):
            return {}

    adapter = ModuleType('utils.datasets')
    adapter.LeRobotDataset = Base
    adapter.ImageTransforms = lambda *args: None
    monkeypatch.setitem(sys.modules, 'utils.datasets', adapter)
    manifest = {'prediction_horizon': 20, 'execution_horizon': 20, 'dataset_root': str(tmp_path),
                'dataset_revision': 'pinned', 'phase_config': {}}
    monkeypatch.setattr(dom_training, 'load_family', lambda *args, **kwargs: manifest)
    for name in ['parquet_hashes.json', 'video_hashes.json', 'phase_hashes.json']:
        (tmp_path/name).write_text(json.dumps({}))
    data = dom_training.build_dataset(tmp_path, [8, 15], training=True)
    assert captured['delta_timestamps']['action'] == list(range(20))
    assert data.horizon == 20 and data.episodes == [8, 15]
    assert sum('images' in key for key in captured['required_features']) == 2


def test_horizon_configuration_rejects_execution_beyond_prediction():
    assert experiment_horizons({}) == (20, 20)
    assert experiment_horizons({'prediction_horizon': 20, 'execution_horizon': 20}) == (20, 20)
    for prediction, execution in [(0, 20), (20, 0), (20, 50)]:
        with pytest.raises(ValueError):
            experiment_horizons({'prediction_horizon': prediction, 'execution_horizon': execution})


def test_twenty_step_validation_uses_valid_targets_and_rejects_stale_fifty_step_batches(monkeypatch):
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(chunk_size=20, n_action_steps=20, max_action_dim=7)
        def reset(self):
            pass
        def predict_action_chunk(self, batch, noise=None):
            return torch.zeros_like(noise)

    monkeypatch.setattr(dom_training, 'fixed_noise', lambda batch, policy, device: torch.zeros(2, 20, 7))
    targets = torch.zeros(2, 20, 7)
    targets[1] = 2
    padding = torch.zeros(2, 20, dtype=torch.bool)
    padding[0, 10:] = True
    batch = {'action': targets, 'action_is_pad': padding, 'dynamic_object.phase_id': torch.tensor([0, 1])}
    metrics = dom_training.evaluate(Policy(), [batch], 'fp32', device='cpu')
    assert metrics['h20/action_mse'] == pytest.approx(4*20/30)
    assert metrics['phase0/h20/action_mse'] == 0
    assert metrics['phase1/h20/action_mse'] == 4
    assert not any('h50/' in key for key in metrics)
    stale = {**batch, 'action': torch.zeros(2, 50, 7), 'action_is_pad': torch.zeros(2, 50, dtype=torch.bool)}
    with pytest.raises(ValueError, match='match the prediction horizon'):
        dom_training.evaluate(Policy(), [stale], 'fp32', device='cpu')


def test_horizon_change_preserves_frozen_data_and_invalidates_old_profiles(tmp_path):
    from scripts.dom100_horizon20 import derive_family
    parent = tmp_path/'parent'
    parent.mkdir()
    (parent/'phase_cache').mkdir()
    (parent/'phase_cache/000008.npz').write_bytes(b'cached prefix state')
    (parent/'initialization').mkdir()
    (parent/'initialization/model.safetensors').write_bytes(b'pinned weights')
    atomic_json(parent/'initialization/config.json', {'chunk_size': 50})
    atomic_json(parent/'split.json', {'splits': {'train': [8], 'validation': [15], 'test': [23]}})
    atomic_json(parent/'dataset_selection.json', {'episode_ids': [8, 15, 23]})
    atomic_json(parent/'phase_hashes.json', {'8': file_digest(parent/'phase_cache/000008.npz')})
    atomic_json(parent/'evaluation_plan.json', {'mode': 'after_training'})
    artifacts = {name: file_digest(parent/name) for name in ['dataset_selection.json',
                 'phase_hashes.json', 'initialization/config.json']}
    manifest = {'initialization': str(parent/'initialization'),
                'weights_sha256': file_digest(parent/'initialization/model.safetensors'),
                'split_sha256': file_digest(parent/'split.json'), 'source_sha256': 'previous 50-step source',
                'artifacts': artifacts, 'counts': {'train': 1, 'validation': 1, 'test': 1},
                'train_frames': 99, 'velocity_mean': [1, 2, 3], 'velocity_std': [4, 5, 6]}
    manifest['fingerprint'] = digest(manifest)
    atomic_json(parent/'manifest.json', manifest)
    before = (parent/'manifest.json').read_bytes()
    result = derive_family(parent, tmp_path/'h20')
    assert experiment_horizons(result) == (20, 20)
    assert result['fingerprint'] != manifest['fingerprint']
    assert result['artifacts']['split.json'] == manifest['split_sha256']
    assert result['artifacts']['dataset_selection.json'] == manifest['artifacts']['dataset_selection.json']
    assert result['velocity_mean'] == manifest['velocity_mean'] and result['train_frames'] == 99
    assert (tmp_path/'h20/phase_cache/000008.npz').read_bytes() == b'cached prefix state'
    assert (parent/'manifest.json').read_bytes() == before
    with pytest.raises(FileExistsError):
        derive_family(parent, tmp_path/'h20')
