"""Plan or verify the old-account dataset transfer. No Hub downloads occur."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent / 'family'
DATA = Path('/work/nvme/bicn/anayak2/dynamicvla/huggingface/lerobot/hzxie/DOM')
REVISION = 'bbf6baa15f7253746d271a6e958e037afa537b1e'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    split = json.loads((ROOT / 'split.json').read_text())['splits']
    required = set(split['train'] + split['validation'])
    parquet = json.loads((ROOT / 'parquet_hashes.json').read_text())
    videos = json.loads((ROOT / 'video_hashes.json').read_text())
    files = {f'data/chunk-{i//1000:03d}/episode_{i:06d}.parquet': parquet[str(i)] for i in required}
    files.update({p: v['sha256'] for p, v in videos.items()
                  if int(Path(p).stem.split('_')[-1]) in required})
    dataset_files = len(files)
    metadata = {k: v for k, v in json.loads((ROOT / 'metadata_hashes.json').read_text()).items()
                if k in ['meta/info.json', 'meta/tasks.jsonl', 'meta/episodes.jsonl', 'meta/episodes_stats.jsonl']}
    files.update(metadata)
    needed = []

    def check(item):
        rel, expected = item
        path = DATA / rel
        return rel if not path.is_file() or sha(path) != expected else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for idx, missing in enumerate(pool.map(check, sorted(files.items())), 1):
            if missing is not None:
                needed.append(missing)
            if idx % 2000 == 0 or idx == len(files):
                state = {'complete': False, 'mode': 'transfer_plan' if args.plan else 'transfer_verification',
                         'checked': idx, 'total_files': len(files), 'needs_transfer': len(needed),
                         'seconds': time.monotonic()-start}
                (ROOT.parent / 'prefetch-progress.json').write_text(json.dumps(state, indent=2))
                print(json.dumps(state), flush=True)
    (ROOT.parent / 'transfer-files.nul').write_bytes(b''.join(p.encode()+b'\0' for p in needed))
    summary = {'needs_transfer': len(needed), 'already_verified': len(files)-len(needed),
               'total_files': len(files), 'dataset_files': dataset_files,
               'metadata_files': len(metadata), 'source': 'old Delta account; no Hub downloads'}
    (ROOT.parent / 'transfer-plan.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.plan:
        print(json.dumps(summary), flush=True)
        return
    assert not needed, f'{len(needed)} files need transfer; see transfer-files.nul'
    phase = json.loads((ROOT / 'phase_hashes.json').read_text())
    for i in sorted(required):
        assert sha(ROOT / 'phase_cache' / f'{i:06d}.npz') == phase[str(i)], i
    receipt = {'complete': True, 'state': 'complete', 'errors': [], 'dataset_root': str(DATA),
               'dataset_revision': REVISION, 'training_episodes': len(split['train']),
               'validation_episodes': len(split['validation']), 'test_downloaded': False,
               'completed_files': dataset_files, 'total_files': dataset_files,
               'phase_files_verified': len(required), 'metadata_files_verified': len(metadata),
               'metadata_hashes': metadata, 'seconds': time.monotonic()-start,
               'data_acquisition': 'reused verified cache plus transfer from old Delta account; Hub downloads stopped at user request'}
    path = ROOT.parent / 'verified-data.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(receipt, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
