"""Content-addressed, bounded parallel validation of immutable dataset videos."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import os
from pathlib import Path
import sqlite3
import threading
import time

CONTRACT='av1-dimensions-frame-count-pts-v1'
DEFAULT_CACHE=Path(__file__).resolve().parents[1]/'scratch/real_robot/validation-cache/videos.sqlite3'


def fingerprint(path):
    stat=path.stat()
    return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


def content_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def decode_video(path,width,height,frames,fps):
    import av
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        if stream.width!=width or stream.height!=height or stream.codec_context.name not in ('av1','libdav1d'):
            raise ValueError('Video dimensions/codec mismatch: '+str(path))
        # Two files x two decoder threads bounds CPU use during live collection.
        stream.codec_context.thread_count=2
        count=0
        for i,frame in enumerate(container.decode(video=0)):
            count+=1
            if frame.pts is None or abs(float(frame.pts*frame.time_base)-i/fps)>1e-4:
                raise ValueError('Video timestamp mismatch: '+str(path))
        if count!=frames:raise ValueError('Video decode count mismatch: '+str(path))


def validate_videos(jobs,*,cache_path=DEFAULT_CACHE,workers=None,verbose=False):
    """Hash every file; reuse decoding only for identical bytes and expectations.

    Jobs are (path, width, height, frame_count, fps). None disables the cache.
    Cached checks survive renames, hard links, copies, and process restarts.
    """
    workers=int(os.environ.get('KRVLA_VIDEO_VALIDATION_WORKERS','2')) if workers is None else workers
    if not isinstance(workers,int) or not 1<=workers<=4:raise ValueError('Video validation workers must be between 1 and 4')
    started=time.monotonic();lock=threading.Lock();database=None
    if cache_path is not None:
        cache_path=Path(cache_path);cache_path.parent.mkdir(parents=True,exist_ok=True)
        database=sqlite3.connect(str(cache_path),timeout=30,check_same_thread=False)
        database.execute('PRAGMA journal_mode=WAL')
        database.execute('CREATE TABLE IF NOT EXISTS verified_video (cache_key TEXT PRIMARY KEY)')
        database.commit()
    def check(job):
        path,width,height,frames,fps=job;path=Path(path)
        before=fingerprint(path)
        digest=content_hash(path)
        if fingerprint(path)!=before:raise ValueError('Video changed during hashing: '+str(path))
        key=f'{CONTRACT}:{digest}:{width}:{height}:{frames}:{fps}'
        hit=False
        if database is not None:
            with lock:hit=database.execute('SELECT 1 FROM verified_video WHERE cache_key=?',(key,)).fetchone() is not None
        if hit:
            if fingerprint(path)!=before:raise ValueError('Video changed during cache lookup: '+str(path))
            return 'cached'
        decode_video(path,width,height,frames,fps)
        if fingerprint(path)!=before:raise ValueError('Video changed during validation: '+str(path))
        if database is not None:
            with lock:
                database.execute('INSERT OR IGNORE INTO verified_video VALUES (?)',(key,));database.commit()
        return 'decoded'
    counts=dict(cached=0,decoded=0)
    if verbose:print(f'Video validation: {len(jobs)} files, {workers} workers; unchanged content reuses verified checks.',flush=True)
    try:
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='video-validation') as pool:
            pending=[pool.submit(check,job) for job in jobs]
            try:
                for future in as_completed(pending):counts[future.result()]+=1
            except BaseException:
                for future in pending:future.cancel()
                raise
    finally:
        if database is not None:database.close()
    counts['elapsed_s']=time.monotonic()-started
    if verbose:print(f'Video validation complete: {counts["decoded"]} decoded, {counts["cached"]} cached; {counts["elapsed_s"]:.1f}s.',flush=True)
    return counts
