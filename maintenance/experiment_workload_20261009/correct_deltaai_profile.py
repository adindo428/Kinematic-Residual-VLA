"""Freeze the confirmed sampler correction without changing the old family."""
import json
from pathlib import Path
import shutil
import sys

work = Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
old_source = Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261008T201101Z/dom100')
source = old_source.parent.parent / 'training-20261009T150839Z' / 'dom100'
old_family = work / 'dom100-palmetto-h20'
family = work / 'dom100-palmetto-h20-r2'
assert not source.exists() and not family.exists()
shutil.copytree(old_source, source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
path = source / 'utils/dom_data.py'
text = path.read_text()
old = 'self.lengths = np.asarray(lengths, dtype=np.int64)'
new = 'self.lengths = np.array(lengths, dtype=np.int64, copy=True)'
assert text.count(old) == 1
text = text.replace(old, new)
old = 'self.lengths = base.episode_data_index["length"].numpy()'
new = 'self.lengths = base.episode_data_index["length"].numpy().copy()'
assert text.count(old) == 1
path.write_text(text.replace(old, new))
sys.path.insert(0, str(source))
from utils.dom_experiment import digest, file_digest, source_digest
original = json.loads((old_family / 'manifest.json').read_text())
old_fingerprint = original.pop('fingerprint')
assert digest(original) == old_fingerprint
assert source_digest(old_source) == original['source_sha256']
family.mkdir()
for item in old_family.iterdir():
    if item.is_file() and item.name != 'manifest.json':
        shutil.copy2(item, family / item.name)
for name in ('phase_cache', 'initialization'):
    (family / name).symlink_to((old_family / name).resolve(), target_is_directory=True)
for name, expected in original.get('artifacts', {}).items():
    assert file_digest(family / name) == expected, name
assert file_digest(family / 'split.json') == original['split_sha256']
manifest = dict(original)
manifest['source_sha256'] = source_digest(source)
manifest['sampler_fix'] = {
    'parent_manifest': old_fingerprint,
    'parent_source_sha256': original['source_sha256'],
    'reason': 'Own numpy length buffers before multiprocessing workers relocate torch storage',
    'changed_file': 'utils/dom_data.py',
    'split_and_targets_unchanged': True,
    'old_profile_for_reference_only': str(old_family / 'profiling/gh200-3347563'),
}
manifest['fingerprint'] = digest(manifest)
(family / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
(family / 'provenance').mkdir()
(family / 'provenance/parent-manifest.json').write_text((old_family / 'manifest.json').read_text())
report = {'family': str(family), 'source': str(source), 'fingerprint': manifest['fingerprint'],
          'source_sha256': manifest['source_sha256'], 'split_sha256': manifest['split_sha256'],
          'training_launched': False, 'old_family_preserved': True}
(family / 'SAMPLER_FIX_REPORT.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
