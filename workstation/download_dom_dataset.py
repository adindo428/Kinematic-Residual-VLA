#!/usr/bin/env python3
"""Resume a complete, pinned DOM repository download into the SSD dataset tree."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import time
import threading

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers, disable_progress_bars

REPOSITORY = 'hzxie/DOM'
SSD = Path('/media/ailab-students/Linux_SSD').resolve()
TRAINING_CAMERAS = ['observation.images.opst_cam','observation.images.wrist_cam']
HTTP = threading.local()


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def retry(function, label):
    for attempt in range(5):
        try:
            return function()
        except Exception as error:
            if attempt == 4:
                raise RuntimeError(f'{label}: {type(error).__name__}; retry the download to resume') from None
            response = getattr(error, 'response', None)
            http_status = getattr(response, 'status_code', None)
            delay = min(60, 5*2**attempt)
            if http_status == 429:
                reset = re.search(r'(?:^|;)\s*t=(\d+)', response.headers.get('RateLimit', ''))
                delay = int(reset.group(1))+2 if reset else 302
            print(f'Retrying {label}: {type(error).__name__}, HTTP {http_status}, wait {delay}s', flush=True)
            time.sleep(delay)


def checksum_matches(path, size, checksum, git_blob):
    if not path.is_file() or path.stat().st_size != size:
        return False
    value = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        value.update(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2*1024**2), b''):
            value.update(block)
    return value.hexdigest() == checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--state-root', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cameras',choices=('training','all'),default='training')
    parser.add_argument('--direct-get',action='store_true',
                        help='Use pinned resolver GETs and inventory hashes without redundant per-file HEAD requests')
    parser.add_argument('--verify-only', action='store_true',
                        help='Verify pending local files against the frozen inventory without any Hub requests')
    args = parser.parse_args()
    scope = ("(path NOT LIKE 'videos/%' OR path LIKE 'videos/%/observation.images.opst_cam/%' "
             "OR path LIKE 'videos/%/observation.images.wrist_cam/%')" if args.cameras=='training' else '1')
    data, state = args.dataset_root.resolve(), args.state_root.resolve()
    if not all(path.is_relative_to(SSD) for path in (data, state)):
        raise ValueError('Dataset and download state must be on the Linux SSD')
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        raise ValueError('Use an immutable 40-character dataset revision')
    if not 1 <= args.workers <= 16:
        raise ValueError('Use between one and sixteen download workers')
    data.mkdir(parents=True, exist_ok=True)
    state.mkdir(parents=True, exist_ok=True)
    disable_progress_bars()
    db = sqlite3.connect(state/'inventory.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS configuration (key TEXT PRIMARY KEY, value TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS directories (path TEXT PRIMARY KEY, listed INTEGER DEFAULT 0)')
    db.execute('CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, checksum TEXT, git_blob INTEGER, verified INTEGER DEFAULT 0)')
    db.execute('CREATE INDEX IF NOT EXISTS pending_files ON files(verified,path)')
    for key, value in (('revision', args.revision), ('dataset_root', str(data))):
        previous = db.execute('SELECT value FROM configuration WHERE key=?', (key,)).fetchone()
        if previous and previous[0] != value:
            raise ValueError('Download state belongs to a different revision or destination')
        db.execute('INSERT OR IGNORE INTO configuration VALUES (?,?)', (key, value))
    db.execute("INSERT OR IGNORE INTO directories(path) VALUES ('')")
    db.commit()
    status = {'repository':REPOSITORY, 'revision':args.revision, 'dataset_root':str(data),
              'workers':args.workers, 'all_camera_streams':args.cameras=='all', 'checksum_verification':True,
              'selected_cameras':TRAINING_CAMERAS if args.cameras=='training' else TRAINING_CAMERAS+['observation.images.side_cam'],
              'completion_scope':'all_episodes_training_cameras' if args.cameras=='training' else 'all_release_files',
              'download_method':'pinned_resolver_get' if args.direct_get else 'hub_sdk'}
    last_report = 0.

    def report(phase, force=False):
        nonlocal last_report
        if not force and time.monotonic()-last_report < 30:
            return
        last_report = time.monotonic()
        files, size = db.execute(f'SELECT COUNT(*),COALESCE(SUM(size),0) FROM files WHERE {scope}').fetchone()
        verified, verified_bytes = db.execute(f'SELECT COUNT(*),COALESCE(SUM(size),0) FROM files WHERE verified=1 AND {scope}').fetchone()
        listed, directories = db.execute('SELECT COALESCE(SUM(listed),0),COUNT(*) FROM directories').fetchone()
        status.update(phase=phase, complete=phase=='complete', inventory_complete=listed==directories,
                      listed_directories=listed, total_directories=directories, files=files, bytes=size,
                      verified_files=verified, verified_bytes=verified_bytes,
                      updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_json(state/'download_status.json', status)
        print(json.dumps(status), flush=True)

    api = HfApi()
    try:
        report('inventory', True)
        if args.verify_only and db.execute('SELECT COUNT(*) FROM directories WHERE listed=0').fetchone()[0]:
            raise ValueError('Offline verification requires a complete pinned inventory')
        # Listing small directories avoids the Hub's timeout on DOM's huge
        # recursive root tree. The SQLite inventory resumes after interruption.
        while True:
            row = db.execute('SELECT path FROM directories WHERE listed=0 ORDER BY LENGTH(path),path LIMIT 1').fetchone()
            if row is None:
                break
            directory = row[0]
            entries = retry(lambda:list(api.list_repo_tree(REPOSITORY, path_in_repo=directory or None,
                repo_type='dataset', revision=args.revision, recursive=False)), directory or 'root')
            for entry in entries:
                parts = PurePosixPath(entry.path)
                if parts.is_absolute() or '..' in parts.parts:
                    raise ValueError('Unsafe repository path')
                if not hasattr(entry, 'size'):
                    db.execute('INSERT OR IGNORE INTO directories(path) VALUES (?)', (entry.path,))
                else:
                    lfs = entry.lfs
                    checksum = lfs.sha256 if lfs else entry.blob_id
                    db.execute('INSERT OR IGNORE INTO files(path,size,checksum,git_blob) VALUES (?,?,?,?)',
                               (entry.path, entry.size, checksum, int(not bool(lfs))))
            db.execute('UPDATE directories SET listed=1 WHERE path=?', (directory,))
            db.commit()
            report('inventory')
        report('verifying_local' if args.verify_only else 'downloading', True)
        remaining = db.execute(f'SELECT COALESCE(SUM(size),0) FROM files WHERE verified=0 AND {scope}').fetchone()[0]
        if not args.verify_only and shutil.disk_usage(data).free < remaining+64*1024**3:
            raise RuntimeError('SSD lacks enough space for the remaining dataset plus 64 GiB headroom')

        def fetch(row):
            name, size, checksum, git_blob = row
            path = data/name
            if checksum_matches(path, size, checksum, git_blob):
                return name
            def download():
                if args.direct_get:
                    url = hf_hub_url(REPOSITORY,name,repo_type='dataset',revision=args.revision)
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temporary = path.with_name(path.name+'.krvla-incomplete')
                    # Inventory pins both content and size. A single resolver
                    # GET is sufficient; obey 429 reset headers through retry().
                    if not hasattr(HTTP,'session'):
                        HTTP.session = requests.Session()
                    with HTTP.session.get(url,headers=build_hf_headers(),stream=True,timeout=(30,60)) as response:
                        response.raise_for_status()
                        with temporary.open('wb') as stream:
                            for block in response.iter_content(chunk_size=2*1024**2):
                                if block:stream.write(block)
                    if not checksum_matches(temporary,size,checksum,git_blob):
                        raise ValueError('Resolver download checksum mismatch')
                    temporary.replace(path)
                    downloaded = path
                else:
                    downloaded = Path(hf_hub_download(REPOSITORY, name, repo_type='dataset',
                        revision=args.revision, local_dir=data, force_download=path.exists()))
                if not checksum_matches(downloaded, size, checksum, git_blob):
                    raise ValueError('Downloaded object checksum mismatch')
                return name
            return retry(download, name)

        if args.verify_only:
            last = ''
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                while True:
                    rows = db.execute(f'SELECT path,size,checksum,git_blob FROM files WHERE verified=0 AND {scope} AND path>? ORDER BY path LIMIT 64',
                                      (last,)).fetchall()
                    if not rows:
                        break
                    matched = pool.map(lambda row:checksum_matches(data/row[0],*row[1:]),rows)
                    for row, valid in zip(rows,matched):
                        if valid:
                            db.execute('UPDATE files SET verified=1 WHERE path=?',(row[0],))
                    db.commit()
                    last = rows[-1][0]
                    report('verifying_local')
            if db.execute(f'SELECT COUNT(*) FROM files WHERE verified=0 AND {scope}').fetchone()[0]:
                report('waiting_for_missing_files', True)
                return 2
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            # Metadata first, then the data and every video camera tree.
            for prefix in ('meta/', 'data/', 'videos/', None):
                clause = "path LIKE ?" if prefix else "path NOT LIKE 'meta/%' AND path NOT LIKE 'data/%' AND path NOT LIKE 'videos/%'"
                while True:
                    rows = db.execute(f'SELECT path,size,checksum,git_blob FROM files WHERE verified=0 AND {scope} AND {clause} ORDER BY path LIMIT 64',
                                      (prefix+'%',) if prefix else ()).fetchall()
                    if not rows:
                        break
                    for name in pool.map(fetch, rows):
                        db.execute('UPDATE files SET verified=1 WHERE path=?', (name,))
                    db.commit()
                    report('downloading')
        if db.execute(f'SELECT COUNT(*) FROM files WHERE verified=0 AND {scope}').fetchone()[0]:
            raise RuntimeError('Unverified dataset files remain')
        info = json.loads((data/'meta/info.json').read_text())
        parquet_count = db.execute("SELECT COUNT(*) FROM files WHERE path LIKE 'data/%.parquet'").fetchone()[0]
        video_count = db.execute("SELECT COUNT(*) FROM files WHERE path LIKE 'videos/%.mp4'").fetchone()[0]
        if parquet_count != info['total_episodes'] or video_count != info['total_videos']:
            raise ValueError('Remote inventory does not match the published episode/video totals')
        selected_videos = db.execute(f"SELECT COUNT(*) FROM files WHERE path LIKE 'videos/%.mp4' AND {scope}").fetchone()[0]
        if selected_videos!=(info['total_episodes']*len(status['selected_cameras'])):
            raise ValueError('Selected camera inventory does not cover every published episode')
        status.update(episodes=parquet_count, videos=selected_videos, published_frames=info['total_frames'],
                      published_video_total=video_count)
        report('complete', True)
    except BaseException as error:
        db.commit()
        status['failure_type'] = type(error).__name__
        report('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', True)
        raise
    finally:
        db.close()


if __name__ == '__main__':
    raise SystemExit(main())
