"""Run with two CPU torchrun workers: deliberately unequal checkpoint clocks."""
import os
from datetime import timedelta
import torch.distributed as dist
from runtime import synchronized_flags

def main():
    dist.init_process_group('gloo',timeout=timedelta(seconds=30))
    rank=dist.get_rank()
    for step in range(20):
        stop,due=synchronized_flags(False,rank==1 and step in [3,9,15],'cpu')
        assert not stop and due==(step in [3,9,15])
    stop,due=synchronized_flags(rank==0,False,'cpu')
    assert stop and not due
    dist.destroy_process_group()
    if rank==0:print('Unequal-rank checkpoint clocks and coordinated stop: PASS',flush=True)


if __name__=='__main__':main()
