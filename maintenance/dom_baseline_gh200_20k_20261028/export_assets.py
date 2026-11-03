"""Build a scoped export of the preserved DOM implementation and 20k metadata."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

work = Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
parent = work / 'domv-scale-ddp-20261014'
out = work / 'export-dom100-20k-20261028'
request = json.load(sys.stdin)
assert not out.exists(), 'Inspect existing export before retrying'
out.mkdir()
family = out / 'family'
family.mkdir()
split = request['split']
selected = set(sum(split['splits'].values(), []))
needed = set(split['splits']['train'] + split['splits']['validation'])
assert len(split['splits']['train']) == 20000

def write(name, value):
    (family / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')

shutil.copytree(work / 'training-sources/delta-domv-f991015a', out / 'source',
                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
write('parent-manifest.json', json.loads((parent / 'manifest.json').read_text()))
write('split.json', split)
write('subset-report.json', request['report'])
write('subset-data-provenance.json', request['provenance'])
for name in ['parquet_hashes.json', 'phase_hashes.json']:
    records = json.loads((parent / name).read_text())
    write(name, {str(i): records[str(i)] for i in sorted(selected)})
records = json.loads((parent / 'video_hashes.json').read_text())
videos = {k: v for k, v in records.items()
          if int(Path(k).stem.split('_')[-1]) in selected
          and ('observation.images.opst_cam' in k or 'observation.images.wrist_cam' in k)}
write('video_hashes.json', videos)
records = json.loads((parent / 'episodes.json').read_text())
write('episodes.json', [r for r in records if int(r['episode_index']) in selected])
for name in ['validation_anchors.json', 'validation_cases.json', 'test_cases.json']:
    shutil.copy2(parent / name, family / name)
write('pilot_episodes.json', split['splits']['train'][:8])
(family / 'initialization').mkdir()
for name in ['config.json', 'model.safetensors']:
    (family / 'initialization' / name).symlink_to((parent / 'initialization' / name).resolve())
(family / 'phase_cache').mkdir()
for i in sorted(needed):
    (family / 'phase_cache' / f'{i:06d}.npz').symlink_to((parent / 'phase_cache' / f'{i:06d}.npz').resolve())
shutil.copytree(work / 'native-krvla-dom', out / 'native', symlinks=True)
print(json.dumps({'export': str(out), 'training_episodes': len(split['splits']['train']),
                  'phase_files': len(needed), 'video_hashes': len(videos)}))
