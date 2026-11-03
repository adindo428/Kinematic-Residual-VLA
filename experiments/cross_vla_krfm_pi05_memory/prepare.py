"""Freeze a derived campaign without editing the audited parent or training data."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download
from common import atomic, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    args = parser.parse_args()
    root, parent = args.root, args.parent
    manifest = json.loads((parent / 'manifest.json').read_text())
    split = json.loads((parent / 'split.json').read_text())['splits']
    assert not (set(split['train']) & set(split['validation']))
    assert not (set(split['train']) & set(split['test']))
    for name in ['split.json', 'parquet_hashes.json', 'video_hashes.json', 'dataset_selection.json']:
        shutil.copyfile(parent / name, root / name)
    atomic(root / 'data-provenance.json', {
        'parent': str(parent), 'manifest': manifest['fingerprint'],
        'split_sha256': sha(root / 'split.json'), 'dataset_root': manifest['dataset_root'],
        'dataset_revision': manifest['dataset_revision'],
        'velocity_source': 'privileged current simulator velocity, analytic KR only',
        'velocity_mean': manifest['velocity_mean'], 'fps': 25,
        'audit_reused': True, 'per_access_byte_checks': True,
        'normalization': 'train-only deterministic uniform-anchor sample, original action coordinate; shared by paired baseline/KR',
    })
    # Uniform frame sampling avoids giving short episodes disproportionate weight.
    # Stats include valid chunk actions and VLASH proxy states for all delays0..3.
    # No validation/test data contribute to normalization or drift centering.
    if not (root / 'statistics.json').exists():
        lengths = {}
        for line in (Path(manifest['dataset_root']) / 'meta/episodes.jsonl').open():
            row = json.loads(line)
            lengths[row['episode_index']] = row['length']
        eps = split['train']
        ends = np.cumsum([lengths[ep] for ep in eps])
        rng = np.random.default_rng(42)
        chosen = rng.choice(int(ends[-1]), size=min(200_000, int(ends[-1])), replace=False)
        groups = {}
        for idx in chosen:
            local = int(np.searchsorted(ends, idx, side='right'))
            groups.setdefault(eps[local], []).append(int(idx - (ends[local - 1] if local else 0)))
        hashes = json.loads((root / 'parquet_hashes.json').read_text())

        def read(entry):
            ep, anchors = entry
            file = Path(manifest['dataset_root']) / f'data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet'
            if sha(file) != hashes[str(ep)]:
                raise ValueError(f'Parquet changed: {ep}')
            t = pq.read_table(file, columns=['action', 'observation.state'])
            a, s = [np.array(t[k].to_pylist(), dtype=np.float32) for k in ['action', 'observation.state']]
            actions, states = [], []
            for frame in anchors:
                # Uniform single offset to keep the estimator bounded; all
                # methods share this same frozen physical-coordinate statistic.
                d = (ep + frame) % 4
                state = a[min(frame + d - 1, len(a)-1), :6].copy() if d else s[frame].copy()
                chunk = a[frame+d:min(frame+d+20, len(a))].copy()
                chunk[:, :6] -= state[:6]
                actions.append(chunk)
                states.append(state)
            return np.concatenate(actions), np.stack(states)

        actions, states = [], []
        with ThreadPoolExecutor(max_workers=8) as pool:
            for i, (a, s) in enumerate(pool.map(read, groups.items())):
                actions.append(a); states.append(s)
                if i % 5000 == 0:
                    print('normalization episodes', i, flush=True)
        stats = {}
        for key, values in [('action', actions), ('observation.state', states)]:
            x = np.concatenate(values)
            stats[key] = {'mean': x.mean(axis=0).tolist(), 'std': x.std(axis=0).clip(1e-6).tolist(),
                          'q01': np.quantile(x, .01, axis=0).tolist(), 'q99': np.quantile(x, .99, axis=0).tolist()}
        atomic(root / 'statistics.json', stats)
    refs = {}
    for name in ['lerobot/pi05_base', 'lerobot/smolvla_base']:
        revision = HfApi().model_info(name).sha
        path = snapshot_download(name, revision=revision)
        files = {str(p.relative_to(path)): sha(p) for p in Path(path).rglob('*') if p.is_file()}
        refs[name] = {'revision': revision, 'path': path, 'files': files}
    for name in ['google/paligemma-3b-pt-224', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct']:
        # Tokenizers only. A gated-tokenizer access failure stops preparation.
        revision = HfApi().model_info(name).sha
        path = snapshot_download(name, revision=revision,
                                 allow_patterns=['*token*', '*vocab*', '*merges*', '*special*', '*processor*', 'config.json'])
        refs[name] = {'revision': revision, 'path': path}
    atomic(root / 'checkpoints.json', refs)
    import importlib.metadata
    import importlib.util
    import subprocess
    files={}
    for module in ['lerobot.policies.pi05.modeling_pi05', 'lerobot.policies.smolvla.modeling_smolvla',
                   'transformers.models.gemma.modeling_gemma']:
        path=importlib.util.find_spec(module).origin
        files[path]=sha(path)
    for path in (root/'upstream-vlash'/'vlash').rglob('*.py'):
        files[str(path)]=sha(path)
    atomic(root/'implementation-lock.json',{'versions':{k:importlib.metadata.version(k) for k in
        ['torch','torchcodec','transformers','lerobot','numpy']},'files':files,
        'vlash_commit':subprocess.check_output(['git','-C',str(root/'upstream-vlash'),'rev-parse','HEAD'],text=True).strip()})
    atomic(root / 'prepared.json', {'complete': True, 'statistics_sha256': sha(root / 'statistics.json'),
                                  'split_sha256': sha(root / 'split.json'), 'checkpoints_sha256': sha(root / 'checkpoints.json'),
                                  'implementation_sha256':sha(root/'implementation-lock.json')})


if __name__ == '__main__':
    main()
