#!/usr/bin/env python3
"""Snapshot and verify the requested DOMV/KRV100 Table 1 evaluation archive."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'scratch/dom45'
REPO_ID = 'adi2440/krvla-evaluation-results'
PREFIX = 'table1-domv-krv100-20261111'
STAGE = WORK / 'releases' / PREFIX
LOW = WORK / 'evaluations/low-clutter-full-table1-20261102'
FULL = WORK / 'evaluations/krv-full-original-dom-20261106'


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def prepare():
    STAGE.mkdir(parents=True, exist_ok=True)
    groups = {f'low-clutter-{key}': [(LOW / 'results' / key, Path('low-clutter/results') / key)]
              for key in ['dom100_best', 'dom100_latest', 'krv100_best', 'krv100_latest']}
    groups['full-scale-shared-domv-krv100'] = [
        (FULL, Path('full-scale/krv-full-original-dom-20261106')),
        (WORK / 'evaluations/domv-full-original-dom-20261106', Path('full-scale/domv-full-original-dom-20261106'))]
    groups['protocols-and-comparisons'] = [
        (LOW / name, Path('low-clutter') / name) for name in [
            'README.md', 'table1_comparison.md', 'table1_comparison.csv', 'comparison.json',
            'protocol.json', 'selection.json', 'catalog.json', 'configs', 'test_lists',
            'paired_analysis', 'checkpoint_naming_audit.json']]
    # Archive evaluation artifacts only; weights are already released separately.
    excluded = {'.pt', '.pth', '.safetensors', '.bin', '.pyc'}
    records = []
    archives = []
    for label, sources in groups.items():
        target = STAGE / (label + '.tar')
        temporary = target.with_suffix('.tar.tmp')
        group_records = []
        with tarfile.open(temporary, 'w', dereference=True) as tar:
            for source, destination in sources:
                assert source.exists(), source
                paths = sorted(source.rglob('*')) if source.is_dir() else [source]
                for path in paths:
                    if not path.is_file() or path.suffix in excluded or '__pycache__' in path.parts:
                        continue
                    relative = path.relative_to(source) if source.is_dir() else None
                    name = str(destination / relative) if relative is not None else str(destination)
                    before = path.stat()
                    tar.add(path, arcname=name, recursive=False)
                    after = path.stat()
                    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f'Changed during snapshot: {path}'
                    group_records.append({'archive': target.name, 'path': name, 'bytes': before.st_size,
                                          'video': path.suffix.lower() in {'.mp4', '.webm', '.mov', '.avi'}})
        temporary.replace(target)
        with tarfile.open(target) as tar:
            members = tar.getmembers()
            assert [(m.name, m.size) for m in members] == [(r['path'], r['bytes']) for r in group_records]
        entry = {'path': target.name, 'bytes': target.stat().st_size, 'sha256': sha256(target),
                 'files': len(group_records), 'videos': sum(r['video'] for r in group_records)}
        archives.append(entry)
        records.extend(group_records)
        print(json.dumps(entry), flush=True)
    for name, rows in [('file-index.csv', records), ('video-index.csv', [r for r in records if r['video']])]:
        with (STAGE / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['archive', 'path', 'bytes', 'video'])
            writer.writeheader()
            writer.writerows(rows)
    manifest = {'repo_id': REPO_ID, 'prefix': PREFIX, 'snapshot_utc': datetime.now(timezone.utc).isoformat(),
                'archives': archives, 'files': len(records), 'videos': sum(r['video'] for r in records),
                'full_scale_status': 'Paused/partial; preserve per-condition denominators. Upload complete does not mean evaluation complete.'}
    write(STAGE / 'manifest.json', manifest)
    (STAGE / 'SHA256SUMS').write_text(''.join(f"{a['sha256']}  {a['path']}\n" for a in archives))
    (STAGE / 'README.md').write_text('''# DOMV and KRV100 Table 1 evaluation runs and videos

This snapshot contains recorded videos, trial scores, trajectories/actions, logs,
verification records, checkpoint identity metadata, protocols and comparison tables.
Download the tar files and extract with `tar -xf ARCHIVE.tar`.
`video-index.csv` lists every video and its exact path inside its archive;
`file-index.csv` lists all archived files. `SHA256SUMS` verifies the tar downloads.

- Low clutter: DOM (`dom100`, the fresh 20k baseline) and KRV100, both validation-best
  and latest epoch 10. Four completed suites of 1,800 scored trials each. Pilots are
  retained in distinct `pilot/` folders and are excluded from the scores.
- Full scale: original DOM configurations, DOMV_Full validation-best and available
  KRV100_Full best/latest constant/causal-gate runs. Shared scene-first execution
  artifacts, including DOMV videos, reside in the shared KRV resident schedule tree.
  These evaluations are paused or partial. Consult the archived comparison table
  and per-condition counts; provisional results are not complete 1,800-trial suites.
- Protocols and comparisons: low-clutter configurations, seeds, checkpoint catalog,
  original table snapshots and paired DOM/KRV analyses. Comparison tables retain
  contextual rows for other methods; their run/video artifacts are outside this upload.

All available videos in the selected run trees are retained, including separately
labelled pilot/qualification and historical blocks. Recording policies differ by
phase, so some scored trials have no video. No new evaluations were run. Model
weight files are excluded; original relative artifact paths are preserved.
This upload retains the existing dataset repository visibility.
''')


def upload():
    from huggingface_hub import HfApi
    api = HfApi(token=(WORK / 'cache/huggingface/token').read_text().strip())
    assert api.whoami()['name'] == 'adi2440'
    info = api.repo_info(REPO_ID, repo_type='dataset')
    print('Destination:', REPO_ID, 'private:', info.private, flush=True)
    manifest = json.loads((STAGE / 'manifest.json').read_text())
    paths = [STAGE / a['path'] for a in manifest['archives']]
    paths += [STAGE / name for name in ['README.md', 'manifest.json', 'SHA256SUMS', 'file-index.csv', 'video-index.csv']]
    # A commit per archive makes retries resumable and keeps upload progress reviewable.
    for path in paths:
        print('Uploading', path.name, flush=True)
        api.upload_file(repo_id=REPO_ID, repo_type='dataset', path_or_fileobj=path,
                        path_in_repo=PREFIX + '/' + path.name,
                        commit_message='Archive Table 1 DOMV/KRV100 evaluations: ' + path.name)
    remote = {f.path: f for f in api.list_repo_tree(REPO_ID, repo_type='dataset', path_in_repo=PREFIX, recursive=True)}
    for path in paths:
        entry = remote[PREFIX + '/' + path.name]
        assert entry.size == path.stat().st_size, path.name
        if entry.lfs:
            assert entry.lfs.sha256 == sha256(path), path.name
        else:
            data = path.read_bytes()
            assert entry.blob_id == hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest(), path.name
    receipt = {'state': 'complete', 'repo_id': REPO_ID, 'prefix': PREFIX,
               'verified_files': len(paths), 'verified_bytes': sum(p.stat().st_size for p in paths),
               'videos': manifest['videos'], 'archived_files': manifest['files'],
               'private': info.private, 'completed_utc': datetime.now(timezone.utc).isoformat()}
    write(STAGE / 'upload-receipt.json', receipt)
    api.upload_file(repo_id=REPO_ID, repo_type='dataset', path_or_fileobj=STAGE / 'upload-receipt.json',
                    path_in_repo=PREFIX + '/upload-receipt.json', commit_message='Verify Table 1 evaluation archive checksums')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    if args.upload:
        upload()
    else:
        prepare()
