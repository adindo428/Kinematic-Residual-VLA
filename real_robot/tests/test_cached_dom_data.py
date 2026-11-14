import sys
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'third_party/DynamicVLA'))
from cached_dom_data import CachedDOMData


def test_predecoded_cache_preserves_historical_frames_and_pixel_scale(tmp_path):
    key='observation.images.opst_cam';folder=tmp_path/key;folder.mkdir()
    frames=np.random.default_rng(4).integers(0,256,(5,3,4,4),dtype=np.uint8);np.save(folder/'000007.npy',frames)
    # Exercise cache-only sampling independently of video metadata/model loading.
    cache=CachedDOMData.__new__(CachedDOMData);cache.rgb_root=tmp_path;cache.rgb_maps=OrderedDict();cache.base=SimpleNamespace(meta=SimpleNamespace(fps=25))
    result=cache._videos({'_dom_episode_id':7},{key:[.04,.12]})[key]
    torch.testing.assert_close(result,torch.tensor(frames[[1,3]],dtype=torch.float32)/255,atol=0,rtol=0)
    single=cache._videos({'_dom_episode_id':7},{key:[.08]})[key]
    torch.testing.assert_close(single,torch.tensor(frames[2],dtype=torch.float32)/255,atol=0,rtol=0)


def test_checkpoint_retention_preserves_resume_patience(tmp_path):
    import json
    from train_real_finetune import completed_progress
    def write(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
    write(tmp_path/'baseline_selection.json',dict(selection_score=10.))
    for epoch,score in [(1,5.),(2,4.),(3,4.5),(4,4.7)]:
        record=dict(epoch=epoch,selection_score=score,validation_loss=score,completed_epoch=True)
        path=tmp_path/f'epoch-{epoch:03d}/provenance.json' if epoch in [2,4] else tmp_path/f'epoch_history/epoch-{epoch:03d}.json'
        write(path,record)
    latest,best,stale=completed_progress(tmp_path)
    assert best==4. and stale==2 and latest[1]['epoch']==4


def test_recorded_prompt_is_returned_verbatim(monkeypatch):
    from utils.dom_data import DOMData
    original='Pick up the rolling orange and place it in the box.'
    monkeypatch.setattr(DOMData,'__getitem__',lambda self,index:dict(task='grab the rolling orange and put on the box.',episode_index=torch.tensor(0)))
    cache=CachedDOMData.__new__(CachedDOMData);cache.instruction_mode='recorded';cache.recorded_instructions={0:original}
    assert cache[3]['task']==original
