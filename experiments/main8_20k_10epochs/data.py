"""DOM adapter using the frozen subset and inherited per-access byte hashes.

The backbone sees its native single observation, two camera views, task and
proprioception. Environment velocity is analytic-only and never tokenized.
VLASH samples one uniform delay, its official non-shared-observation variant.
"""
from collections import OrderedDict
import json
from pathlib import Path
import random

import numpy as np
import torch
from common import sha


def reference_state(states, actions, frame, delay):
    """DOM proprioception is pose6; gripper is only an action channel."""
    return (actions[min(frame+delay-1,len(actions)-1), :6].copy()
            if delay else states[frame].copy())


class DOM(torch.utils.data.Dataset):
    cameras = ('observation.images.opst_cam', 'observation.images.wrist_cam')

    def __init__(self, root, split='train', max_delay=0, training=False, seed=42):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'data-provenance.json').read_text())
        self.data = Path(self.manifest['dataset_root'])
        self.episodes = json.loads((self.root / 'split.json').read_text())['splits'][split]
        if not self.episodes or len(self.episodes) != len(set(self.episodes)):
            raise ValueError('Empty or duplicate episode membership')
        self.seed = seed
        metadata = {}
        for line in (self.data / 'meta/episodes.jsonl').open():
            row = json.loads(line)
            metadata[row['episode_index']] = row
        self.rows = [metadata[ep] for ep in self.episodes]
        self.lengths = np.array([r['length'] for r in self.rows])
        self.ends = self.lengths.cumsum()
        self.max_delay, self.training = max_delay, training
        self.cache = OrderedDict()
        self.parquet_hashes = json.loads((self.root / 'parquet_hashes.json').read_text())
        self.video_hashes = json.loads((self.root / 'video_hashes.json').read_text())
        self.horizon = 20

    def __len__(self):
        return int(self.ends[-1])

    def episode(self, local):
        import pyarrow.parquet as pq
        from torchcodec.decoders import VideoDecoder
        ep = self.episodes[local]
        if ep not in self.cache:
            rel = f'data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet'
            if sha(self.data / rel) != self.parquet_hashes[str(ep)]:
                raise ValueError(f'Parquet changed: {ep}')
            table = pq.read_table(self.data / rel, columns=['observation.state', 'action', 'observation.environment_state'])
            values = {k: np.array(table[k].to_pylist(), dtype=np.float32) for k in table.column_names}
            decoders = {}
            for camera in self.cameras:
                path = f'videos/chunk-{ep // 1000:03d}/{camera}/episode_{ep:06d}.mp4'
                if sha(self.data / path) != self.video_hashes[path]['sha256']:
                    raise ValueError(f'Video changed: {path}')
                # Decode immutable verified bytes, bounded to two episodes per worker.
                decoders[camera] = VideoDecoder((self.data / path).read_bytes(), num_ffmpeg_threads=1)
            self.cache[ep] = (values, decoders)
            while len(self.cache) > 2:
                self.cache.popitem(last=False)
        return self.cache[ep]

    def __getitem__(self, index):
        from instruction_generator import InstructionGenerator
        epoch, idx = index if isinstance(index, tuple) else (0, int(index))
        local = int(np.searchsorted(self.ends, idx, side='right'))
        frame = idx - (int(self.ends[local - 1]) if local else 0)
        values, decoders = self.episode(local)
        n = len(values['action'])
        rng = random.Random(self.seed + epoch * len(self) + idx)
        delay = rng.randrange(self.max_delay + 1) if self.training else 0
        positions = frame + delay + np.arange(self.horizon)
        action = values['action'][positions.clip(max=n - 1)].copy()
        # VLASH's default state rollforward is the previous commanded absolute
        # action, not future observed state. No future image/object access.
        state = reference_state(values['observation.state'],values['action'],frame,delay)
        action[:, :6] -= state[:6]
        item = {'observation.state': torch.from_numpy(state),
                'action': torch.from_numpy(action),
                'action_is_pad': torch.from_numpy(positions >= n),
                'velocity': torch.from_numpy(values['observation.environment_state'][frame, 6:9].copy()),
                'offset': delay, 'sample_index': idx}
        old = random.getstate()
        random.setstate(rng.getstate())
        item['task'] = InstructionGenerator.generate_instruction(self.rows[local]['tasks'])
        random.setstate(old)
        for camera, decoder in decoders.items():
            image = decoder.get_frame_at(frame).data.float() / 255
            # Native policy performs its own model-specific padding/resizing.
            item[camera] = image
        return item


class Batches:
    """Deterministic global shuffle, exact consumed cursor, no prefetch cursor."""
    def __init__(self, count, epoch, cursor, global_batch, microbatch, rank, world, seed=42):
        assert global_batch % (microbatch * world) == 0
        self.count, self.epoch, self.cursor = count, epoch, cursor
        self.global_batch, self.microbatch, self.rank, self.world = global_batch, microbatch, rank, world
        self.seed = seed

    def __iter__(self):
        order = np.random.default_rng(self.seed + self.epoch).permutation(self.count)
        end = self.count // self.global_batch * self.global_batch
        for start in range(self.cursor, end, self.global_batch):
            chunk = order[start:start + self.global_batch].reshape(-1, self.world, self.microbatch)
            for group in chunk:
                yield [(self.epoch, int(i)) for i in group[self.rank]]

    def __len__(self):
        end = self.count // self.global_batch * self.global_batch
        return max(0, end - self.cursor) // (self.microbatch * self.world)
