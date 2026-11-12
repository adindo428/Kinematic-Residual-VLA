"""Frozen real-data adapter preserving each family's native observation/delay layout."""
import json
from pathlib import Path
import random
from collections import OrderedDict

import numpy as np
import torch


class RealFamilyDataset(torch.utils.data.Dataset):
    def __init__(self,campaign,family,split='train'):
        self.family=family;self.training=split=='train';self.max_delay=3 if family.startswith('vlash') and self.training else 0
        self.contract=campaign['training_recipe'];c=self.contract;self.root=Path(c['local_verified_dataset'])
        settings=dict(campaign.get('performance',{}).get(family.split('_')[0],{}))
        settings.update(campaign.get('family_performance',{}).get(family,{}))
        self.cache_limit=settings.get('decoder_cache_episodes',2)
        import hashlib
        if hashlib.sha256((self.root/'meta/quality_policy.json').read_bytes()).hexdigest()!=c['dataset_quality_policy_sha256']:
            raise ValueError('Frozen data quality policy changed')
        splits=json.loads((self.root/'meta/splits.json').read_text())
        quality=json.loads((self.root/'meta/quality_policy.json').read_text())
        self.metadata={r['episode_index']:r for r in map(json.loads,(self.root/'meta/episodes.jsonl').read_text().splitlines())}
        self.anchors=[];self.cache=OrderedDict()
        for ep in splits[split]:
            usable=set(quality['usable_frames'][str(ep)])
            self.anchors.extend((ep,i) for i in range(self.metadata[ep]['length']) if i in usable and max(0,i-2) in usable)

    def __len__(self):return len(self.anchors)

    def __getstate__(self):
        state=dict(self.__dict__);state['cache']=OrderedDict();return state

    def episode(self,ep):
        import pyarrow.parquet as pq
        from torchcodec.decoders import VideoDecoder
        if ep not in self.cache:
            table=pq.read_table(self.root/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet',
                columns=['observation.state','action','observation.environment_state'])
            arrays={k:torch.tensor(np.asarray(table[k].to_pylist()),dtype=torch.float32) for k in table.column_names}
            quality=[json.loads(x) for x in (self.root/f'quality/episode_{ep:06d}.jsonl').read_text().splitlines()]
            decoders={k:VideoDecoder((self.root/f'videos/chunk-{ep//1000:03d}/{k}/episode_{ep:06d}.mp4').read_bytes(),num_ffmpeg_threads=1)
                for k in ['observation.images.opst_cam','observation.images.wrist_cam']}
            if any(len(decoder)!=len(quality) for decoder in decoders.values()):raise ValueError('Video and label frame counts differ')
            self.cache[ep]=(arrays,quality,decoders)
            while len(self.cache)>getattr(self,'cache_limit',2):self.cache.popitem(last=False)
        return self.cache[ep]

    def __getitem__(self,index):
        from utils.real_robot_contract import delta_pose
        from utils.instruction_generator import InstructionGenerator
        ep,frame=self.anchors[index];arrays,quality,decoders=self.episode(ep)
        current=arrays['observation.state'][frame].clone();n=len(quality)
        delay=random.randrange(self.max_delay+1) if self.max_delay else 0
        positions=frame+delay+torch.arange(20)
        absolute=arrays['action'][positions.clamp(max=n-1)].clone()
        reference=arrays['action'][min(frame+delay-1,n-1),:6].clone() if delay else current
        q=quality[frame];required=bool(q['velocity_required'])
        velocity=arrays['observation.environment_state'][frame,6:9].clone()
        valid_velocity=bool(q['velocity_estimated'] and not q['velocity_zero_filled']) or q.get('velocity_repaired',False)
        if required and not valid_velocity:raise ValueError('Invalid required velocity')
        if not required:velocity.zero_()
        item={key:decoder.get_frame_at(frame).data.float()/255 for key,decoder in decoders.items()}
        item.update({'observation.state':reference,'reference_state':reference.clone(),
            'action':delta_pose(absolute,reference[None,:],'wrapped_xyz_v1'),
            'action_is_pad':positions>=n, 'velocity':velocity,
            'velocity_required':torch.tensor(required),'offset':delay,
            'task':InstructionGenerator.generate_instruction(self.metadata[ep]['tasks'][0]),
            'episode_index':torch.tensor(ep),'frame_index':torch.tensor(frame)})
        return item


def prepare_statistics(campaign,output):
    """Unaugmented train-only delta-chunk statistics shared by every family/pair."""
    import pyarrow.parquet as pq
    c=campaign['training_recipe'];root=Path(c['local_verified_dataset'])
    splits=json.loads((root/'meta/splits.json').read_text());quality=json.loads((root/'meta/quality_policy.json').read_text())
    states=[];targets=[];velocities=[];count=0
    for ep in splits['train']:
        table=pq.read_table(root/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet',columns=['observation.state','action','observation.environment_state'])
        state=np.array(table['observation.state'].to_pylist());action=np.array(table['action'].to_pylist());env=np.array(table['observation.environment_state'].to_pylist())
        usable=set(quality['usable_frames'][str(ep)]);n=len(state)
        anchors=[i for i in range(n) if max(0,i-2) in usable and i in usable]
        q=[json.loads(x) for x in (root/f'quality/episode_{ep:06d}.jsonl').read_text().splitlines()]
        for i in anchors:
            future=action[i:min(n,i+20)].copy();future[:,:6]-=state[i]
            future[:,3:6]=(future[:,3:6]+np.pi)%(2*np.pi)-np.pi
            targets.append(future);states.append(state[i]);count+=1
            if q[i]['velocity_required']:velocities.append(env[i,6:9])
    stats={}
    for key,array in [('observation.state',np.asarray(states)),('action',np.concatenate(targets))]:
        stats[key]={k:v.tolist() for k,v in dict(mean=array.mean(0),std=np.maximum(array.std(0),1e-6),
            q01=np.quantile(array,.01,axis=0),q99=np.quantile(array,.99,axis=0)).items()}
    result=dict(statistics=stats,velocity_mean=np.asarray(velocities).mean(0).tolist(),training_anchors=count,
        semantics='Unaugmented train-only eligible anchors; valid 20-step wrapped XYZ Euler delta targets; same statistics for baseline/KR, no KR transform in statistics. Velocity centering uses required-velocity train anchors only.')
    Path(output).write_text(json.dumps(result,indent=2)+'\n');return result
