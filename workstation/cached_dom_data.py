"""Lossless predecoded RGB cache; preserves DOM adapter's sampling and labels."""
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import torch
from utils.dom_data import DOMData


class CachedDOMData(DOMData):
    def __init__(self,base,cache_root,instruction_mode='generated',**kwargs):
        if instruction_mode not in ('generated','recorded'):raise ValueError('Unknown instruction mode')
        super().__init__(base,**kwargs);self.rgb_root=Path(cache_root);self.rgb_maps=OrderedDict()
        self.instruction_mode=instruction_mode;self.recorded_instructions={}
        if instruction_mode=='recorded':
            for row in map(json.loads,(Path(base.root)/'meta/episodes.jsonl').read_text().splitlines()):
                if len(row['tasks'])!=1:raise ValueError('Expected one recorded task per episode')
                value=row['tasks'][0]
                metadata=json.loads(value) if isinstance(value,str) else value
                if not metadata.get('instruction'):raise ValueError('Recorded instruction is missing')
                self.recorded_instructions[row['episode_index']]=metadata['instruction']

    def __getitem__(self,item):
        batch=super().__getitem__(item)
        if getattr(self,'instruction_mode','generated')=='recorded':batch['task']=self.recorded_instructions[int(batch['episode_index'])]
        return batch

    def __getstate__(self):
        state=super().__getstate__();state['rgb_maps']=OrderedDict();return state

    def _videos(self,episode,timestamps):
        result={};ep=episode['_dom_episode_id']
        for key,ts in timestamps.items():
            cache_key=(ep,key);array=self.rgb_maps.pop(cache_key,None)
            if array is None:array=np.load(self.rgb_root/key/f'{ep:06d}.npy',mmap_mode='r')
            self.rgb_maps[cache_key]=array
            while len(self.rgb_maps)>32:self.rgb_maps.popitem(last=False)
            indices=[round(float(t)*self.base.meta.fps) for t in ts]
            result[key]=torch.from_numpy(np.array(array[indices],copy=True)).float().div_(255).squeeze(0)
        return result
