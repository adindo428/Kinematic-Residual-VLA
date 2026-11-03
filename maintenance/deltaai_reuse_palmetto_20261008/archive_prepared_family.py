import hashlib, io, json, sys, tarfile, gzip
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

root=Path('/scratch/aparame/Dynamic_VLA/dom100-h20')
names=['manifest.json','dataset_selection.json','dataset_coverage.json','episodes.json','split.json','parquet_hashes.json','phase_hashes.json','pilot_episodes.json','validation_anchors.json','validation_cases.json','test_cases.json','video_hashes.json','horizon_change.json']
phases=json.loads((root/'phase_hashes.json').read_text())
def read_phase(ep):
    raw=(root/'phase_cache'/f'{int(ep):06d}.npz').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==phases[ep], 'Phase hash mismatch '+ep
    return ep,raw
def add(tar,name,raw):
    info=tarfile.TarInfo(name); info.size=len(raw); info.mode=0o644
    tar.addfile(info,io.BytesIO(raw))
with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1) as compressed, tarfile.open(fileobj=compressed,mode='w|') as tar:
    for name in names: add(tar,name,(root/name).read_bytes())
    ids=list(phases)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(ids),256):
            for ep,raw in pool.map(read_phase,ids[start:start+256]):
                add(tar,f'phase_cache/{int(ep):06d}.npz',raw)
            if start%10000<256: print('phase_cache_archived',min(start+256,len(ids)),file=sys.stderr,flush=True)
