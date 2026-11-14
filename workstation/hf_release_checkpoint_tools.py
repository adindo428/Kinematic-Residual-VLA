"""Portable loaders for adi2440/kinematic-residual-vla-checkpoints.

Load one model family per Python process so its frozen module names remain isolated.
"""
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import patch

REPO_ID = 'adi2440/kinematic-residual-vla-checkpoints'


def index(root): return json.loads((Path(root)/'model-index.json').read_text())


def resolve(root, checkpoint):
    catalog = index(root); key = catalog['aliases'].get(checkpoint, checkpoint)
    return next(c for c in catalog['checkpoints'] if c['id'] == key)


def download(checkpoint, local_dir, training_state=False, token=None):
    from huggingface_hub import hf_hub_download, snapshot_download
    path = hf_hub_download(REPO_ID, 'model-index.json', token=token)
    catalog = json.loads(Path(path).read_text()); key = catalog['aliases'].get(checkpoint, checkpoint)
    item = next(c for c in catalog['checkpoints'] if c['id'] == key)
    patterns = ['model-index.json', 'files.json', 'SHA256SUMS', 'environment_versions.json', 'tools/*', 'licenses/*', 'assets/**','evaluation_configs/**',
                'environment-*.json','requirements-*.txt',item['code']+'/**', 'code/vlash/**', item['path']+'/**',
                'code/runtime-native/**' if item['format']=='native_safetensors' else 'code/runtime-adapters/**']
    if item.get('family'): patterns += [item['family']+'/**']
    ignored = None if training_state or item['format']=='adapter_torch' else ['**/training_state.pt']
    return Path(snapshot_download(REPO_ID, allow_patterns=patterns, ignore_patterns=ignored,
                                  local_dir=local_dir, token=token))


@contextmanager
def local_assets(root):
    from transformers import AutoConfig, AutoTokenizer
    root = Path(root)
    with ExitStack() as stack:
        for cls in [AutoConfig, AutoTokenizer]:
            original = cls.from_pretrained
            def load(name, *args, _original=original, **kw):
                local = root/'assets'/str(name)
                if local.is_dir():
                    name = str(local); kw['local_files_only'] = True
                return _original(name, *args, **kw)
            stack.enter_context(patch.object(cls, 'from_pretrained', side_effect=load))
        yield


def load_model(root, checkpoint, device='cuda', train=False, inference_config=None):
    root = Path(root).resolve(); item = resolve(root, checkpoint); directory = root/item['path']
    flavor='native' if item['format']=='native_safetensors' else 'adapters'
    sys.path.insert(0,str(root/f'code/runtime-{flavor}'))
    import torch,transformers
    expected=json.loads((root/f'environment-{flavor}.json').read_text())
    if transformers.__version__!=expected['transformers']:
        raise RuntimeError(f"Use requirements-{flavor}.txt: expected transformers {expected['transformers']}, found {transformers.__version__}")
    sys.path.insert(0, str(root/item['code']))
    if item['format'] == 'native_safetensors':
        from utils.helpers import get_policy_cfg
        from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
        from safetensors.torch import load_model as load_weights
        config_path=root/inference_config if inference_config else directory/'config.json'
        cfg = get_policy_cfg(cfg_file=config_path); cfg.enable_streaming = False; cfg.device = str(device)
        with local_assets(root): model = DynamicVLAPolicy.from_pretrained(str(directory), config=cfg)
        load_weights(model, str(directory/'model.safetensors'), strict=True)
        model.to(device)
    else:
        if inference_config: raise ValueError('Evaluation config variants are provided for native DOM/KRV checkpoints')
        sys.path.insert(1, str(root/'code/vlash'))
        from policy_inference import Policy
        code = root/item['code']; recipe = json.loads((code/'recipes'/(item['model']+'.json')).read_text())
        refs = json.loads((code/'checkpoints.local.json').read_text())
        for repo, ref in refs.items():
            location = root/'assets'/repo
            if not location.is_dir(): raise FileNotFoundError('Missing asset directory: '+str(location))
            ref['path'] = str(location)
        state = torch.load(directory/'checkpoint.pt', map_location='cpu', mmap=True, weights_only=False)
        if state['identity'] != item['identity'] or state['progress'] != item['progress']:
            raise ValueError('Checkpoint identity/progress differs from the release manifest')
        original_read=Path.read_text
        def read_refs(path,*args,**kwargs):
            if path.resolve()==(code/'checkpoints.local.json').resolve(): return json.dumps(refs)
            return original_read(path,*args,**kwargs)
        # Resolve machine-specific asset paths in memory, preserving downloaded checksums.
        with patch.object(Path,'read_text',read_refs), local_assets(root): model = Policy(str(code), recipe, device)
        model.load_state_dict(state['model'], strict=True)
    if train: model.train()
    else: model.eval().requires_grad_(False)
    return model, item


def training_state(root, checkpoint):
    """Return optimizer/scheduler/progress/RNG state without allocating GPU memory."""
    import torch
    root = Path(root); item = resolve(root, checkpoint)
    return torch.load(root/item['training_state'], map_location='cpu', mmap=True, weights_only=False)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('--local-dir', required=True)
    p.add_argument('--training-state', action='store_true'); a = p.parse_args()
    print(download(a.checkpoint, a.local_dir, a.training_state))
