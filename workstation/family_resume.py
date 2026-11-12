"""Recover an exact consumed sample cursor from the original fixed-batch runner."""
import math
from pathlib import Path
import torch


def resume_order(snapshot,weights,count,completed_epoch=False,effective_batch=16):
    updates=math.ceil(count/effective_batch);epoch=snapshot['epoch'];step=snapshot['step']
    if completed_epoch:
        if step!=epoch*updates:raise ValueError('Completed epoch/update count differs')
        return epoch+1,None,0
    done=step-(epoch-1)*updates
    if not 0<=done<=updates:raise ValueError('Cannot infer the partial-epoch cursor')
    generator=torch.Generator().manual_seed(42)
    for _ in range(epoch):order=torch.multinomial(torch.as_tensor(weights,dtype=torch.double),count,True,generator=generator)
    if not torch.equal(generator.get_state(),snapshot['rng']['sampler']):
        raise ValueError('Sampler replay failed; refusing to guess the consumed sample order')
    consumed=min(done*effective_batch,count)
    return epoch,order[consumed:].tolist(),consumed


def select_checkpoint(output,explicit=None):
    paths=[Path(explicit)] if explicit else [Path(output)/name for name in ['latest.pt','recovery.pt']]
    choices=[]
    for path in paths:
        if not path.exists():continue
        snapshot=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if snapshot.get('optimizer') is None or snapshot.get('selected_weights')!='raw':continue
        completed=snapshot.get('completed_epoch',path.name=='latest.pt')
        choices.append((snapshot['step'],int(completed),path,snapshot,completed))
    if not choices:raise ValueError('No raw optimizer checkpoint available to resume')
    _,_,path,snapshot,completed=max(choices,key=lambda item:item[:2])
    return path,snapshot,completed
