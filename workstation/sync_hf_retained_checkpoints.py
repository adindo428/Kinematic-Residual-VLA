#!/usr/bin/env python3
"""Incrementally back up retained local checkpoints and sanitized metric snapshots."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'scratch/dom45'
OUT = WORK / 'releases/incremental-20261117'
REPO = 'adi2440/kinematic-residual-vla-checkpoints'
SECRET = re.compile(r'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN .*PRIVATE KEY-----')


def clean(text):
    if SECRET.search(text):
        raise ValueError('Credential pattern in metadata')
    text = text.replace(str(ROOT), '<project>').replace(str(WORK), '<work>')
    text = re.sub(r'/(?:home|scratch|projects|work|media|mnt|tmp)/[^\s\"\x27,;)]+', '<local-path>', text)
    text = re.sub(r'[\w.-]+@(?:[\w.-]+)', '<redacted-account>', text)
    return text


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def training_metrics():
    """Snapshot training loss, validation and generation pointers, excluding logs."""
    stage = OUT / 'training-metrics'
    for p in (WORK / 'training').rglob('*.json'):
        if not (p.name.startswith(('epoch-', 'validation-')) or p.name in
                {'best.json', 'latest.json', 'final.json', 'training_complete.json'}):
            continue
        if not p.is_file() or p.stat().st_size > 10 * 1024**2:
            continue
        target = stage / p.relative_to(WORK / 'training')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(clean(p.read_text()))
    api = HfApi(token=(WORK / 'cache/huggingface/token').read_text().strip())
    api.upload_folder(repo_id=REPO, folder_path=stage, path_in_repo='updates/20261117/training-metrics',
                      commit_message='Back up training and validation metric snapshots')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=(WORK / 'cache/huggingface/token').read_text().strip())
    assert api.whoami()['name'] == 'adi2440'
    remote = {}
    for f in api.list_repo_tree(REPO, recursive=True):
        lfs = getattr(f, 'lfs', None)
        if lfs:
            remote[lfs['sha256'] if isinstance(lfs, dict) else lfs.sha256] = f.path
    cachepath = OUT / 'hashes.private.json'
    cache = json.loads(cachepath.read_text()) if cachepath.exists() else {}
    old = WORK / 'releases/kinematic-residual-vla-checkpoints-20261108'
    oldcache = json.loads((old / 'hash_cache.private.json').read_text())
    for rel, entry in oldcache.items():
        p = old / 'staging' / rel
        if p.exists():
            s = p.stat()
            if s.st_size == entry['bytes'] and s.st_mtime_ns == entry['mtime_ns']:
                cache[f'{s.st_dev}:{s.st_ino}:{s.st_size}:{s.st_mtime_ns}'] = entry['sha256']
    records = []
    paths = sorted(p for p in (WORK / 'training').rglob('*') if p.is_file() and p.suffix in {'.pt', '.pth', '.safetensors'}
                   and not any(x in p.parts for x in ['distributed-smoke', 'initialization', '__pycache__']))
    for p in paths:
        s = p.stat()
        key = f'{s.st_dev}:{s.st_ino}:{s.st_size}:{s.st_mtime_ns}'
        sha = cache.get(key)
        if not sha:
            h = hashlib.sha256()
            with p.open('rb') as stream:
                while block := stream.read(16 * 1024**2):
                    h.update(block)
            sha = h.hexdigest()
            now = p.stat()
            assert (now.st_ino, now.st_size, now.st_mtime_ns) == (s.st_ino, s.st_size, s.st_mtime_ns), 'Checkpoint changed during hashing'
            cache[key] = sha
            write(cachepath, cache)
        rel = str(p.relative_to(WORK / 'training'))
        dest = remote.get(sha)
        if not dest:
            dest = 'checkpoints/retained-20261117/' + rel
            print('uploading', rel, s.st_size, flush=True)
            api.upload_file(repo_id=REPO, path_or_fileobj=p, path_in_repo=dest,
                            commit_message='Back up retained checkpoint: ' + p.parent.name)
            remote[sha] = dest
        now = p.stat()
        assert (now.st_ino, now.st_size, now.st_mtime_ns) == (s.st_ino, s.st_size, s.st_mtime_ns), 'Checkpoint changed during upload'
        records.append({'local_relative_identity': rel, 'path': dest, 'bytes': s.st_size, 'sha256': sha})
        write(OUT / 'checkpoint-progress.json', records)
        # Include companion configs, normalizers, and progress records without loading pickle.
        for q in p.parent.iterdir():
            if q.is_file() and q.suffix in {'.json', '.yaml', '.yml'} and q.stat().st_size < 10 * 1024**2:
                target = OUT / 'staging/companions' / Path(rel).parent / q.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(clean(q.read_text()))
    stage = OUT / 'staging'
    # Preserve summary scope and denominators; exclude videos, command dumps, logs and PID files.
    names = {'live_results.md', 'results.json', 'error_metrics.json', 'completion.json', 'category_scores.json',
             'scope-completion.json', 'performance.md', 'answers.md', 'protocol.json'}
    for p in (WORK / 'evaluations').rglob('*'):
        if p.is_file() and p.name in names and p.stat().st_size < 20 * 1024**2:
            target = stage / 'metrics' / p.relative_to(WORK / 'evaluations')
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(clean(p.read_text()))
    manifest = {'repo_id': REPO, 'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'checkpoints': records,
                'scope': 'Locally retained training checkpoints; excludes initialization and distributed smoke fixtures. Metrics are snapshots; partial evaluations retain their reported denominators.'}
    write(stage / 'manifest.json', manifest)
    (stage / 'README.md').write_text('# Retained checkpoint update\n\nSee manifest.json for content hashes and exact remote checkpoint paths. Companion configurations and evaluation summary snapshots are included. Partial results are not completed benchmark claims.\n')
    api.upload_folder(repo_id=REPO, folder_path=stage, path_in_repo='updates/20261117',
                      commit_message='Publish retained checkpoint manifest, companion configs and metric snapshots')
    tree = {f.path: f for f in api.list_repo_tree(REPO, recursive=True) if hasattr(f, 'size')}
    for r in records:
        f = tree[r['path']]
        lfs = f.lfs
        assert f.size == r['bytes'] and (lfs['sha256'] if isinstance(lfs, dict) else lfs.sha256) == r['sha256']
    write(OUT / 'complete.json', {'verified_checkpoints': len(records), 'repo_id': REPO, 'snapshot': 'updates/20261117'})
    training_metrics()
    print('Complete:', len(records), 'verified checkpoint files', flush=True)


if __name__ == '__main__':
    main()
