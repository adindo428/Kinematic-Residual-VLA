"""Recorded tracking batches with observation-time velocity and original future labels."""
import json
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
import torch

DEFAULT_DATASET=Path('/scratch/aparame/.cache/huggingface/lerobot/adi2440/dom-franka-tracking')

def read_episode(root, episode):
    root=Path(root);chunk=f'chunk-{episode//1000:03d}';name=f'episode_{episode:06d}'
    rows=pq.read_table(root/'data'/chunk/(name+'.parquet')).to_pylist()
    images={}
    for camera in ('opst_cam','wrist_cam'):
        with av.open(str(root/'videos'/chunk/f'observation.images.{camera}'/(name+'.mp4'))) as container:
            images[camera]=np.stack([f.to_ndarray(format='rgb24') for f in container.decode(video=0)])
        assert len(images[camera])==len(rows)
    tasks={r['task_index']:r['task'] for r in map(json.loads,(root/'meta/tasks.jsonl').read_text().splitlines())}
    return rows,images,tasks

def make_batch(data, anchors, device='cuda', steps=20):
    rows,images,tasks=data
    obs=np.array([[max(0,i-2),i] for i in anchors])
    states=np.asarray([r['observation.state'] for r in rows],np.float32)
    env=np.asarray([r['observation.environment_state'] for r in rows],np.float32)
    actions=np.stack([np.asarray([r['action'] for r in rows[i:i+steps]],np.float32) for i in anchors])
    assert actions.shape==(len(anchors),steps,7)
    actions[:,:,:6]-=states[anchors,None,:6]
    from utils.instruction_generator import InstructionGenerator
    batch={'observation.state':torch.from_numpy(states[obs]).to(device),
           'observation.environment_state':torch.from_numpy(env[obs]).to(device),
           'action':torch.from_numpy(actions).to(device),
           'task':[InstructionGenerator.generate_instruction(tasks[rows[i]['task_index']]) for i in anchors]}
    for camera,frames in images.items():
        batch[f'observation.images.{camera}']=torch.from_numpy(frames[obs].copy()).permute(0,1,4,2,3).contiguous().to(device).float()/255.
    return batch

def noise_for(episode,anchors,steps=20,dim=32,device='cuda'):
    return torch.stack([torch.randn(steps,dim,generator=torch.Generator().manual_seed(42+episode*10000+i)) for i in anchors]).to(device)
