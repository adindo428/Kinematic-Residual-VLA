import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tracking_checkpoint import checkpoint_files


def test_checkpoint_selection_never_falls_back_to_latest(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'delta_timestamps':{'observation':[-2,0]}}))
    (tmp_path/'model.safetensors').write_bytes(b'latest')
    with pytest.raises(FileNotFoundError,match='validation-selected'):
        checkpoint_files(tmp_path)
    (tmp_path/'model.best_validation_action_mse.safetensors').write_bytes(b'best')
    assert checkpoint_files(tmp_path)[2].name=='model.best_validation_action_mse.safetensors'
    with pytest.raises(ValueError,match='training_ready'):
        checkpoint_files(tmp_path,require_complete=True)


def test_live_worker_uses_past_and_current_observations_and_no_future_labels(monkeypatch):
    import torch
    from tracking_online_policy import Worker
    monkeypatch.setattr(torch.Tensor,'cuda',lambda self,*args,**kwargs:self)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    captured=[]
    class Policy:
        config=SimpleNamespace(chunk_size=20,n_action_steps=20,max_action_dim=32)
        def reset(self):pass
        def predict_action_chunk(self,batch,noise):
            captured.append(batch)
            assert set(batch)=={'observation.state','observation.environment_state','task','observation.images.opst_cam','observation.images.wrist_cam'}
            assert batch['observation.images.opst_cam'].is_contiguous()
            assert noise.shape==(1,20,32)
            return torch.zeros(1,20,7)
    worker=Worker({});worker.policy=Policy();worker.next_frame=0;worker.seed=42
    image=np.zeros((360,480,3),np.uint8)
    for frame in range(21):
        result=worker.observe({'frame':frame,'time':frame/25,'state':np.full(6,frame,np.float32),
            'environment_state':np.full(9,frame,np.float32),'images':{'opst_cam':image,'wrist_cam':image},
            'action':np.full((20,7),999),'dynamic_object.lookahead_position_xyz':np.full(3,999)})
        assert result['replanned']==(frame in (0,20))
    np.testing.assert_array_equal(captured[0]['observation.state'][0,:,0],[0,0])
    np.testing.assert_array_equal(captured[1]['observation.state'][0,:,0],[18,20])
    np.testing.assert_array_equal(captured[1]['observation.environment_state'][0,:,6],[18,20])
    np.testing.assert_array_equal(result['chunk_absolute'][:,:6],np.full((20,6),20))
    np.testing.assert_array_equal(result['chunk_absolute'][:,6],0)
