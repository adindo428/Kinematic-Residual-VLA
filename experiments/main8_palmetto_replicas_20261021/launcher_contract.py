"""Bounded actual two-node CPU collective and rank stream contract."""
import json
import os
from pathlib import Path
import signal
import argparse
import socket
stopped=[False]
signal.signal(signal.SIGUSR1,lambda *_:stopped.__setitem__(0,True))
import time
import torch
import torch.distributed as dist

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    dist.init_process_group('gloo')
    rank=dist.get_rank();world=dist.get_world_size()
    rows=[None]*world;dist.all_gather_object(rows,{'rank':rank,'host':socket.gethostname(),'local':int(os.environ['LOCAL_RANK'])})
    assert world==4 and len({x['host'] for x in rows})==2
    assert sorted(x['local'] for x in rows)==[0,0,1,1]
    value=torch.tensor([rank+1]);dist.all_reduce(value);assert value.item()==10
    if rank==0:(a.root/'contract-signal-ready.json').write_text('ready')
    dist.barrier()
    deadline=time.monotonic()+60
    while not stopped[0] and time.monotonic()<deadline:time.sleep(.1)
    count=torch.tensor([int(stopped[0])]);dist.all_reduce(count);assert count.item()==4, 'Signal did not reach all four ranks'
    if rank==0:(a.root/'two-node-contract.json').write_text(json.dumps({'passed':True,'world':world,'ranks':rows,'early_signal_handler':True},indent=2)+'\n')
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
