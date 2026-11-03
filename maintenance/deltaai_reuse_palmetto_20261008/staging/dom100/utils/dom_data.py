"""Replayable sampling and bounded caching around the existing DOM adapter."""
from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch

from proactive_vla.phase_gate import PhaseGateConfig, replay_prefix_states, gate_from_state
from utils.dom_experiment import PREDICTION_HORIZON


@contextmanager
def sample_seed(seed):
    py, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed)
        np.random.seed(seed % 2**32)
        # Augmentation runs on CPU. torch.manual_seed also changes CUDA RNG,
        # which would make model noise depend on the loader's worker count.
        torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(np_state)


class EpochBatchSampler:
    """Visit every anchor once; cursor counts consumed samples, not prefetch."""
    def __init__(self, lengths, microbatch, epoch=0, cursor=0, seed=42, sampling="frames"):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.total = int(self.lengths.sum())
        self.microbatch, self.epoch, self.cursor = int(microbatch), int(epoch), int(cursor)
        self.seed, self.sampling = int(seed), sampling
        if microbatch < 1 or not 0 <= cursor <= self.total or sampling not in ("frames", "blocks"):
            raise ValueError("Invalid sampler configuration")

    def __len__(self):
        return (self.total-self.cursor+self.microbatch-1)//self.microbatch

    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        if self.sampling == "frames":
            order = rng.permutation(self.total)[self.cursor:]
            for pos in range(0, len(order), self.microbatch):
                yield [(self.epoch, int(i)) for i in order[pos:pos+self.microbatch]]
            return
        starts = np.r_[0, np.cumsum(self.lengths)[:-1]]
        block_starts = np.concatenate([np.arange(s, s+n, 8) for s, n in zip(starts, self.lengths)])
        ends = np.repeat(starts+self.lengths, (self.lengths+7)//8)
        batch, skipped = [], 0
        for block in rng.permutation(len(block_starts)):
            indices = np.arange(block_starts[block], min(block_starts[block]+8, ends[block]))
            rng.shuffle(indices)
            for idx in indices:
                if skipped < self.cursor:
                    skipped += 1
                    continue
                batch.append((self.epoch, int(idx)))
                if len(batch) == self.microbatch:
                    yield batch
                    batch = []
        if batch:
            yield batch


class DOMData(torch.utils.data.Dataset):
    def __init__(self, base, horizon=PREDICTION_HORIZON, seed=42, phase_config=None, cache_bytes=512*1024**2,
                 deterministic_eval=False, phase_cache=None, frozen_files=None):
        self.base, self.horizon, self.seed = base, horizon, seed
        self.phase_config = PhaseGateConfig(**(phase_config or {}))
        self.cache_bytes, self.bytes = cache_bytes, 0
        self.episodes, self.meta = base.episodes, base.meta
        self.lengths = base.episode_data_index["length"].numpy()
        self.cache, self.decoders, self.phases = OrderedDict(), OrderedDict(), OrderedDict()
        self.batched_frames = {}
        self.deterministic_eval = deterministic_eval
        self.phase_cache = phase_cache
        self.phase_hashes = None
        self.parquet_hashes = self.video_hashes = None
        if frozen_files is not None:
            import json
            from pathlib import Path
            self.parquet_hashes = json.loads((Path(frozen_files)/"parquet_hashes.json").read_text())
            self.video_hashes = json.loads((Path(frozen_files)/"video_hashes.json").read_text())
        if phase_cache is not None:
            import json
            from pathlib import Path
            self.phase_hashes = json.loads((Path(phase_cache).parent/"phase_hashes.json").read_text())
        reader = base._get_episode
        # Spawn pickles bound methods by attribute name. That name is replaced
        # below, so retaining the bound method would recurse in a worker.
        self._episode_reader_is_bound = getattr(reader, "__self__", None) is base
        self._episode_reader = reader.__func__ if self._episode_reader_is_bound else reader
        base._get_episode = self._episode
        base._query_videos = self._videos

    def __len__(self):
        return len(self.base)

    def _episode(self, episode_id):
        if episode_id in self.cache:
            value, size = self.cache.pop(episode_id)
            self.cache[episode_id] = (value, size)
            return value
        value = (self._episode_reader(self.base, episode_id) if self._episode_reader_is_bound
                 else self._episode_reader(episode_id))
        if self.parquet_hashes is not None:
            from utils.dom_experiment import file_digest
            path = self.base.root/self.meta.get_data_file_path(episode_id)
            if file_digest(path) != self.parquet_hashes[str(episode_id)]:
                raise ValueError(f"Frozen DOM parquet changed: {episode_id}")
            for camera in self.base.video_keys:
                key = str(self.meta.get_video_file_path(episode_id, camera))
                if hashlib.sha256(value[camera]).hexdigest() != self.video_hashes[key]["sha256"]:
                    raise ValueError(f"Frozen DOM video changed: {key}")
        size = sum(v.numel()*v.element_size() if torch.is_tensor(v) else len(v)
                   if isinstance(v, bytes) else 0 for v in value.values())
        value["_dom_episode_id"] = episode_id
        while self.cache and self.bytes+size > self.cache_bytes:
            key, (_, removed) = self.cache.popitem(last=False)
            self.bytes -= removed
            self.phases.pop(key, None)
            for decoder_key in list(self.decoders):
                if decoder_key[0] == key:
                    del self.decoders[decoder_key]
        if size <= self.cache_bytes:
            self.cache[episode_id] = (value, size)
            self.bytes += size
        return value

    def _videos(self, episode, timestamps):
        import torchcodec.decoders
        result = {}
        for key, ts in timestamps.items():
            prepared = self.batched_frames.get((episode["_dom_episode_id"], key))
            if prepared is not None and all(float(t) in prepared for t in ts):
                result[key] = torch.stack([prepared[float(t)] for t in ts]).squeeze(0)
                continue
            cache_key = (episode["_dom_episode_id"], key)
            decoder = self.decoders.pop(cache_key, None)
            if decoder is None:
                decoder = torchcodec.decoders.VideoDecoder(episode[key], seek_mode="approximate", num_ffmpeg_threads=1)
            self.decoders[cache_key] = decoder
            while len(self.decoders) > 8:
                self.decoders.popitem(last=False)
            result[key] = self.base._get_video_frames(decoder, ts).squeeze(0)
        return result

    def __getitem__(self, item):
        epoch, idx = item if isinstance(item, tuple) else (0, int(item))
        key = int.from_bytes(hashlib.sha256(f"{self.seed}:{epoch}:{idx}".encode()).digest()[:8], "little") % (2**63-1)
        with sample_seed(key):
            batch = self.base[int(idx)]
        ep = int(batch["episode_index"])
        if ep not in self.phases:
            if self.phase_cache is not None:
                from pathlib import Path
                from utils.dom_experiment import file_digest
                path = Path(self.phase_cache)/f"{ep:06d}.npz"
                if file_digest(path) != self.phase_hashes[str(ep)]:
                    raise ValueError(f"Frozen phase cache was modified: {ep}")
                with np.load(path) as cached:
                    compact, phases = cached["state"].copy(), cached["phase"].copy()
            else:
                episode = self._episode(ep)
                compact, phases = replay_prefix_states(episode["observation.state"].numpy(),
                    episode["observation.environment_state"].numpy(), episode["action"].numpy(),
                    episode["timestamp"].numpy().reshape(-1), self.phase_config)
            self.phases[ep] = (compact, phases)
            while len(self.phases) > 8:
                self.phases.popitem(last=False)
        compact, phases = self.phases[ep]
        frame = int(batch["frame_index"])
        batch["dynamic_object.phase_gate"] = torch.from_numpy(gate_from_state(compact[frame], self.horizon, 1/self.meta.fps))
        batch["dynamic_object.phase_id"] = torch.tensor(int(phases[frame]))
        return batch

    def __getitems__(self, items):
        # Decode the union of each episode's requested frames in one decoder
        # call per camera, then preserve the adapter's original transforms and
        # sample ordering. Buffers exist only for this episode group.
        groups = {}
        for i, item in enumerate(items):
            idx = item[1] if isinstance(item, tuple) else item
            local = int(torch.searchsorted(self.base.episode_data_index["to"], torch.tensor(idx), right=True))
            groups.setdefault(local, []).append((i, idx))
        result = [None]*len(items)
        for local, anchors in groups.items():
            ep = int(self.episodes[local]); episode = self._episode(ep)
            timestamps = {key: set() for key in self.base.video_keys}
            for _, idx in anchors:
                frame = int(idx-self.base.episode_data_index["from"][local])
                queries, _ = self.base._get_query_indices(frame, int(self.lengths[local]), self.base.delta_indices)
                values = self.base._get_query_timestamps(episode["timestamp"], float(episode["timestamp"][frame]), queries)
                for key, ts in values.items():
                    timestamps[key].update(map(float, ts))
            ts = {key: sorted(values) for key, values in timestamps.items()}
            decoded = self._videos(episode, ts)
            self.batched_frames = {(ep, key): dict(zip(ts[key], value.unsqueeze(0) if len(ts[key])==1 else value))
                                   for key, value in decoded.items()}
            try:
                for i, _ in anchors:
                    result[i] = self[items[i]]
            finally:
                self.batched_frames = {}
        return result
