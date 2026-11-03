"""Relocate immutable preparation artifacts without recomputing their split."""
import hashlib, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

work=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
family=work/'dom100-palmetto-h20'
source=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261008T201101Z/dom100')
sys.path.insert(0,str(source))
from utils.dom_experiment import digest, file_digest, source_digest

original=json.loads((family/'manifest.json').read_text())
fingerprint=original.pop('fingerprint')
assert digest(original)==fingerprint, 'Palmetto manifest fingerprint mismatch'
original['fingerprint']=fingerprint
assert original['prediction_horizon']==original['execution_horizon']==20
assert file_digest(family/'split.json')==original['split_sha256']
for name, expected in original['artifacts'].items():
    if name=='initialization/config.json':
        path=work/'dom100/initialization/config.json'
    else: path=family/name
    assert file_digest(path)==expected, 'Artifact checksum mismatch: '+name
assert file_digest(work/'dom100/initialization/model.safetensors')==original['weights_sha256']
link=family/'initialization'
if not link.exists(): link.symlink_to(work/'dom100/initialization',target_is_directory=True)

# Cached Palmetto bytes must describe the same immutable objects that the
# DeltaAI downloader verifies; otherwise importing its cache would be invalid.
inventory=json.loads((work/'setup-20261008/download-inventory.json').read_text())
entry=next(r for r in inventory if r['repo']=='hzxie/DOM')
assert entry['revision']==original['dataset_revision']
expected={r['path']:r['sha256'] for r in entry['files'] if r['sha256']}
parquets=json.loads((family/'parquet_hashes.json').read_text())
assert digest(parquets)==original['parquets_sha256']
for ep,value in parquets.items():
    ep=int(ep); path=f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
    assert expected.get(path)==value, 'Palmetto parquet differs from pinned dataset: '+path
videos=json.loads((family/'video_hashes.json').read_text())
for path,value in videos.items():
    assert expected.get(path)==value['sha256'], 'Palmetto video differs from pinned dataset: '+path
if '--metadata-only' in sys.argv:
    print(json.dumps({'metadata_verified':True,'pinned_parquets_matched':len(parquets),'pinned_videos_matched':len(videos),'split_sha256':original['split_sha256']}),flush=True)
    sys.exit(0)
del expected, inventory, videos

phases=json.loads((family/'phase_hashes.json').read_text())
def check_phase(item):
    ep,value=item
    assert file_digest(family/'phase_cache'/f'{int(ep):06d}.npz')==value, 'Phase cache checksum mismatch: '+ep
items=list(phases.items())
with ThreadPoolExecutor(max_workers=8) as pool:
    for start in range(0,len(items),512):
        list(pool.map(check_phase,items[start:start+512]))
        if start%10000<512: print(json.dumps({'phase_cache_verified':min(start+512,len(items)),'phase_cache_total':len(phases)}),flush=True)

provenance=family/'provenance'; provenance.mkdir(exist_ok=True)
(provenance/'palmetto-manifest.json').write_text(json.dumps(original,indent=2))
manifest=dict(original)
manifest.pop('fingerprint')
manifest['dataset_root']=str(work/'huggingface/lerobot/hzxie/DOM')
manifest['initialization']=str(work/'dom100/initialization')
manifest['source_sha256']=source_digest(source)
manifest['relocated_from']={'family':'/scratch/aparame/Dynamic_VLA/dom100-h20','fingerprint':fingerprint,'source_sha256':original['source_sha256'],'split_unchanged':True,'phase_cache_unchanged':True,'verified_against_pinned_dataset_objects':True}
manifest['fingerprint']=digest(manifest)
temp=family/'manifest.json.tmp'; temp.write_text(json.dumps(manifest,indent=2)); temp.replace(family/'manifest.json')
report={'family':str(family),'source':str(source),'split_sha256':manifest['split_sha256'],'episode_counts':manifest['counts'],'train_frames':manifest['train_frames'],'verified_phase_cache_files':len(phases),'prediction_horizon':20,'execution_horizon':20,'fingerprint':manifest['fingerprint'],'portable_preparation_ready':True,'dataset_download_still_required':True,'gh200_profiling_still_required':True}
(family/'REUSE_REPORT.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)
