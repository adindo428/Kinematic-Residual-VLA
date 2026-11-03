"""Exact global-batch sharding and coordinated checkpointing for DOM DDP."""
from contextlib import contextmanager, ExitStack
import math
from datetime import timedelta

import torch
import torch.distributed as dist

from core.dom_checkpoint import rng_state, run_lock, save_generation
from utils.dom_data import EpochBatchSampler


_CONTROL_GROUP = None


def initialize_control_group():
    """Create CPU communication for checkpoint metadata and runtime checks."""
    global _CONTROL_GROUP
    _CONTROL_GROUP = dist.new_group(backend='gloo', timeout=timedelta(minutes=5))


def control_group():
    if _CONTROL_GROUP is None:
        raise RuntimeError('CPU control communication was not initialized')
    return _CONTROL_GROUP


class RankBatchSampler:
    """Partition each global update once, keeping collective counts equal.

    Empty rank microsteps decode one zero-weight sample. These dummy forwards
    participate in DDP but never count toward the loss, cursor or epoch.
    """
    def __init__(self, lengths, microbatch, rank, world_size, epoch=0, cursor=0,
                 sampling='frames', effective_batch=352):
        if microbatch < 1 or not 0 <= rank < world_size or effective_batch % world_size:
            raise ValueError('Invalid distributed batch configuration')
        self.global_sampler = EpochBatchSampler(lengths, effective_batch, epoch, cursor, sampling=sampling)
        self.microbatch, self.rank, self.world_size = microbatch, rank, world_size
        self.effective_batch = effective_batch

    def __len__(self):
        remaining = self.global_sampler.total-self.global_sampler.cursor
        full, tail = divmod(remaining, self.effective_batch)
        return full*math.ceil(self.effective_batch/self.world_size/self.microbatch)+(math.ceil(math.ceil(tail/self.world_size)/self.microbatch) if tail else 0)

    def __iter__(self):
        for group in self.global_sampler:
            size = len(group)
            per_rank = math.ceil(size/self.world_size)
            local = group[self.rank*per_rank:(self.rank+1)*per_rank]
            steps = math.ceil(per_rank/self.microbatch)
            for step in range(steps):
                batch = local[step*self.microbatch:(step+1)*self.microbatch]
                valid = bool(batch)
                if not valid:
                    batch = [group[0]]
                yield [(epoch, index, int(valid), size, int(step==steps-1)) for epoch,index in batch]


class RankedData(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        value = dict(self.data[item[:2]])
        return self.annotate(value,item)
    @staticmethod
    def annotate(value,item):
        value['ddp.valid'] = torch.tensor(item[2],dtype=torch.int64)
        value['ddp.update_samples'] = torch.tensor(item[3],dtype=torch.int64)
        value['ddp.update_end'] = torch.tensor(item[4],dtype=torch.int64)
        return value
    def __getitems__(self, items):
        indices = [item[:2] for item in items]
        values = self.data.__getitems__(indices) if hasattr(self.data,'__getitems__') else [self.data[i] for i in indices]
        return [self.annotate(dict(value),item) for value,item in zip(values,items)]


def leader_call(function):
    """Broadcast leader failures rather than stranding peers in a barrier."""
    result = [None]
    if dist.get_rank()==0:
        try:
            result[0] = (True,function())
        except Exception as exc:
            result[0] = (False,f'{type(exc).__name__}: {exc}')
    dist.broadcast_object_list(result,src=0,group=control_group())
    okay,value = result[0]
    if not okay:
        raise RuntimeError(value)
    return value


@contextmanager
def distributed_lock(root):
    stack = ExitStack()
    leader_call(lambda: stack.enter_context(run_lock(root)))
    try:
        yield
    finally:
        # No final collective: a failed peer may already have exited.
        stack.close()


def save_distributed(root, policy, optimizer, scheduler, progress, fingerprint,
                     wandb_id, best=None, milestone=False, device='cuda', final_checkpoint=False):
    world = dist.get_world_size()
    states = [None]*world if dist.get_rank()==0 else None
    dist.gather_object(rng_state(device if str(device).startswith('cuda') else None),states,dst=0,group=control_group())
    def save():
        path = save_generation(root,policy,optimizer,scheduler,progress.copy(),fingerprint,
            wandb_id,best,milestone,distributed_state={'world_size':world,'rng_by_rank':states,'effective_batch':352},
            final_checkpoint=final_checkpoint)
        return str(path)
    return leader_call(save)
