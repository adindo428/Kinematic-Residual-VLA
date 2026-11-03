"""Verify batching preserves the original frame lookup and decoded values."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from proactive_vla.residual_flow import extract_dom_dynamic_fields
from utils.dom_data import DOMData


def test_union_decoding_preserves_frames_and_reproducible_augmentation(monkeypatch):
    calls=[]
    class Decoder:
        def __init__(self, data, *, seek_mode, num_ffmpeg_threads):
            assert seek_mode == 'approximate' and num_ffmpeg_threads == 1
            self.camera=data
        def get_frames_at(self, indices):
            calls.append((self.camera,indices))
            return SimpleNamespace(data=torch.stack([torch.full((3,2,2),int(i),dtype=torch.uint8) for i in indices]),
                pts_seconds=torch.tensor(indices,dtype=torch.float64)/25)
    codec=ModuleType('torchcodec');decoders=ModuleType('torchcodec.decoders')
    decoders.VideoDecoder=Decoder;codec.decoders=decoders
    monkeypatch.setitem(sys.modules,'torchcodec',codec);monkeypatch.setitem(sys.modules,'torchcodec.decoders',decoders)
    source=Path(__file__).resolve().parents[1]/'utils/datasets.py'
    cls=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='LeRobotDataset')
    ns={'torch':torch,'torchcodec':codec,'extract_dom_dynamic_fields':extract_dom_dynamic_fields,
        'InstructionGenerator':SimpleNamespace(generate_instruction=lambda x:str(float(torch.rand(()))))}
    methods=('__getitem__','_get_episode_index','_get_query_indices','_query_episode','_get_query_timestamps','_get_video_frames')
    attrs={}
    for node in cls.body:
        if isinstance(node,ast.FunctionDef) and node.name in methods:
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
            attrs[node.name]=ns[node.name]
    Base=type('Base',(),attrs)
    def base():
        b=Base();b.episodes=[0,1]
        b.meta=SimpleNamespace(fps=25,video_keys=['opst','wrist'])
        b.video_keys=['opst','wrist'];b.camera_keys=b.video_keys;b.image_keys=[]
        b.required_features=['observation.state','observation.environment_state','action']
        b.tolerance_s=.02;b.delta_action=True
        b.delta_indices={'observation.state':[-2,0],'observation.environment_state':[-2,0],
            'action':list(range(20)),'opst':[-2,0],'wrist':[-2,0]}
        b.episode_data_index={'from':torch.tensor([0,20]),'to':torch.tensor([20,40]),'length':torch.tensor([20,20])}
        b.__class__.__len__=lambda self:40
        def episode(ep):
            return {'observation.state':torch.ones(20,6)*.1,'observation.environment_state':torch.zeros(20,9),
                'action':torch.ones(20,7)*.2,'timestamp':torch.arange(20,dtype=torch.float64)/25,
                'task_index':torch.zeros(20,dtype=torch.int64),'opst':b'opst','wrist':b'wrist'}
        b._get_episode=episode;b.meta.tasks=['pick']
        def transform(item,keys):
            amount=torch.rand(())*.01
            for key in keys:item[key]=item[key]+amount
            return item
        b.image_transforms=transform
        return b
    items=[(4,25),(4,3),(4,27),(4,5),(4,4)]
    data=DOMData(base(),cache_bytes=1024*1024)
    individually=[data[item] for item in items]
    assert all(item['action'].shape[0] == item['dynamic_object.phase_gate'].shape[0] == 20 for item in individually)
    calls.clear()
    batched=data.__getitems__(items)
    assert len(calls)==4 # two cameras, two episode unions
    assert max(len(indices) for _,indices in calls)>2
    for before,after in zip(individually,batched):
        assert before.keys()==after.keys()
        for key in before:
            if torch.is_tensor(before[key]):torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)
            else:assert before[key]==after[key]
    assert not data.batched_frames
    assert len(data.decoders)<=8 and data.bytes<=data.cache_bytes
    # Cached raw actions must not be mutated by repeated delta conversion.
    torch.testing.assert_close(data._episode(0)['action'],torch.ones(20,7)*.2)
