"""Episode-directory renames are the commit boundary; partial episodes never resume."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("w") as f:
        json.dump(jsonable(value), f, indent=2, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_jsonl(path, rows):
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), allow_nan=False)+"\n")


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(path, records, format):
    if format == "jsonl":
        write_jsonl(Path(path).with_suffix(".jsonl"), records)
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(jsonable(records)), Path(path).with_suffix(".parquet"))


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        return read_jsonl(path)
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


class RunStore:
    def __init__(self, root, provenance):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / ".collection.lock"
        # flock is released on process death; the file itself is intentionally persistent.
        import fcntl
        self._lock_file = self.lock.open("a")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock_file.close()
            raise ValueError("Another collector owns this output directory")
        old = self.root / "provenance.json"
        if old.exists():
            if json.loads(old.read_text()) != jsonable(provenance):
                self.close()
                raise ValueError("Resume refused: configuration/assets/software provenance changed")
        else:
            atomic_json(old, provenance)
        for folder in ("episodes", "rejected", "interrupted"):
            (self.root / folder).mkdir(exist_ok=True)
        for part in sorted(self.root.glob(".attempt-*.partial")):
            destination = self.root / "interrupted" / f"{part.name}-{time.time_ns()}"
            part.rename(destination)
        self.state_path = self.root / "state.json"
        # The attempt counter is persisted BEFORE work begins to avoid reusing crash seeds.
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"next_attempt": 0}

    def completed(self):
        return sorted((self.root / "episodes").glob("episode_*"))

    def begin(self):
        attempt = self.state["next_attempt"]
        self.state["next_attempt"] += 1
        atomic_json(self.state_path, self.state)
        tmp = self.root / f".attempt-{attempt:06d}.partial"
        tmp.mkdir()
        return attempt, tmp

    def commit(self, tmp, metadata, accepted):
        atomic_json(tmp / "episode.json", metadata)
        if accepted:
            destination = self.root / "episodes" / f"episode_{len(self.completed()):06d}"
        else:
            destination = self.root / "rejected" / f"attempt_{metadata['attempt']:06d}"
        if destination.exists():
            raise ValueError(f"Refusing to overwrite {destination}")
        tmp.rename(destination)

    def close(self):
        if not self._lock_file.closed:
            self._lock_file.close()


class VideoWriter:
    def __init__(self, path, fps, width, height, codec):
        import av
        self.container = av.open(str(path), "w")
        # DOM uses short GOPs to make random training-frame access inexpensive.
        options = {"crf": "25", "preset": "10", "g": "2"} if codec == "libsvtav1" else {}
        self.stream = self.container.add_stream(codec, rate=fps, options=options)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.thread_count = 2
        self.count = 0

    def add(self, rgb):
        import av
        rgb = np.asarray(rgb)
        if rgb.shape != (self.stream.height, self.stream.width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Camera must provide uint8 HWC RGB with configured dimensions")
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        for packet in self.stream.encode(frame):
            self.container.mux(packet)
        self.count += 1

    def close(self):
        if self.container is not None:
            try:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            finally:
                self.container.close()
                self.container = None


def video_summary(path, expected=None):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        times = []
        histogram = np.zeros((3, 256), dtype=np.int64)
        shape = None
        # Decode every frame: header duration alone is not an integrity check.
        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            if shape is not None and shape != list(rgb.shape):
                raise ValueError(f"Video dimensions change within stream: {path}")
            shape = list(rgb.shape)
            times.append(float(frame.pts * frame.time_base))
            # Exact uint8 pixel histograms avoid several costly HWC reductions
            # and floating-point image copies for every decoded frame.
            histogram += [np.bincount(rgb[..., channel].ravel(), minlength=256) for channel in range(3)]
        if not times or (expected is not None and len(times) != expected):
            raise ValueError(f"Video frame count mismatch: {path}")
        if not np.allclose(times, np.arange(len(times))/fps, atol=1e-5):
            raise ValueError(f"Nonuniform video timestamps: {path}")
        values = np.arange(256, dtype=float)/255
        probabilities = histogram/histogram.sum(axis=1, keepdims=True)
        mean = probabilities@values
        std = np.sqrt(np.maximum(0, probabilities@(values**2)-mean**2))
        minimum = np.argmax(histogram > 0, axis=1)/255
        maximum = (255-np.argmax(histogram[:, ::-1] > 0, axis=1))/255
        stats = {"mean": mean[:, None, None].tolist(), "std": std[:, None, None].tolist(),
                 "min": minimum[:, None, None].tolist(),
                 "max": maximum[:, None, None].tolist(), "count": [len(times)]}
        codec = "av1" if stream.codec_context.codec.id == av.Codec("av1", "r").id else stream.codec_context.name
        return dict(frames=len(times), fps=fps, shape=shape, codec=codec, stats=stats)
