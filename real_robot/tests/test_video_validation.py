"""Cached validation must detect changed bytes and changed frame expectations."""
import os
from pathlib import Path
import shutil
import threading
from unittest.mock import patch

import av
import numpy as np
import pytest

from real_robot.video_validation import validate_videos,decode_video


def video(path):
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libsvtav1',rate=25)
        stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
        stream.options={'preset':'12','crf':'35','svtav1-params':'lp=1'}
        for i in range(5):
            image=np.full((64,64,3),i*40,np.uint8)
            frame=av.VideoFrame.from_ndarray(image,format='rgb24');frame.pts=i
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    return (path,64,64,5,25)


def test_identical_video_bytes_reuse_decode_after_copy_and_process_restart(tmp_path):
    job=video(tmp_path/'a.mp4');cache=tmp_path/'cache.sqlite3'
    first=validate_videos([job],cache_path=cache)
    assert first['decoded']==1 and first['cached']==0
    other=tmp_path/'b.mp4';shutil.copy2(job[0],other)
    # A new database connection represents a restarted background worker.
    with patch('real_robot.video_validation.decode_video',side_effect=AssertionError('Unexpected decode')):
        second=validate_videos([(other,*job[1:])],cache_path=cache)
    assert second['cached']==1


def test_changed_bytes_cannot_reuse_cache_even_with_same_size_and_mtime(tmp_path):
    job=video(tmp_path/'a.mp4');cache=tmp_path/'cache.sqlite3'
    validate_videos([job],cache_path=cache)
    stat=job[0].stat();data=bytearray(job[0].read_bytes());data[-1]^=1;job[0].write_bytes(data)
    os.utime(job[0],ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with patch('real_robot.video_validation.decode_video',side_effect=ValueError('changed content checked')):
        with pytest.raises(ValueError,match='changed content checked'):validate_videos([job],cache_path=cache)


def test_new_frame_count_expectation_is_checked_and_failed_validation_is_not_cached(tmp_path):
    job=video(tmp_path/'a.mp4');cache=tmp_path/'cache.sqlite3'
    validate_videos([job],cache_path=cache)
    wrong=(job[0],64,64,6,25)
    for _ in range(2):
        with pytest.raises(ValueError,match='decode count mismatch'):validate_videos([wrong],cache_path=cache)
    with pytest.raises(ValueError,match='dimensions'):validate_videos([(job[0],128,64,5,25)],cache_path=cache)


def test_force_decode_bypasses_verified_cache(tmp_path):
    job=video(tmp_path/'a.mp4');cache=tmp_path/'cache.sqlite3'
    validate_videos([job],cache_path=cache)
    with patch('real_robot.video_validation.decode_video',wraps=decode_video) as decode:
        result=validate_videos([job],cache_path=None)
    assert result['decoded']==1 and decode.call_count==1


def test_worker_count_bounds_parallel_decoding(tmp_path):
    # A barrier proves two jobs really run concurrently; failures cannot deadlock.
    barrier=threading.Barrier(2,timeout=3);lock=threading.Lock();active=0;peak=0
    def decode(*_):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        barrier.wait()
        with lock:active-=1
    jobs=[]
    for i in range(4):
        path=tmp_path/f'{i}.mp4';path.write_bytes(bytes([i]));jobs.append((path,64,64,5,25))
    with patch('real_robot.video_validation.decode_video',side_effect=decode):
        result=validate_videos(jobs,workers=2,cache_path=None)
    assert result['decoded']==4 and peak==2
    with pytest.raises(ValueError,match='between 1 and 4'):validate_videos(jobs,workers=10)
