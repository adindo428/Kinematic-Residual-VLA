"""Native two-time DOM+velocity inputs and trained normalization are retained."""
import json
from pathlib import Path
import torch
from torch import nn


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, split='train', training=False, seed=42):
        root = Path(root)
        site = json.loads((root/'site.json').read_text())
        from core.dom_training import build_dataset
        self.episodes = json.loads((root/'split.json').read_text())['splits'][split]
        self.data = build_dataset(Path(site['dom_family']), self.episodes, training=training)
        self.seed = seed
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]


class Policy(nn.Module):
    def __init__(self, root, recipe, device):
        super().__init__()
        from utils.helpers import get_policy_cfg
        from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
        parent = Path(root)/'parents'/recipe['name']
        config = get_policy_cfg(cfg_file=parent/'config.json')
        self.policy = DynamicVLAPolicy(config).to(device)
        self.recipe = recipe
        if not config.condition_on_object_velocity or config.enable_residual_flow:
            raise ValueError('The DOM member must be the saved velocity-conditioned baseline')
    def forward(self, batch):
        device = next(self.parameters()).device
        batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        return self.policy(batch)[0]
    @torch.no_grad()
    def predict(self, batch):
        device = next(self.parameters()).device
        batch = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
        self.policy.reset()
        return self.policy.predict_action_chunk(batch)
