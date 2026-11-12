import math
import pickle
from collections import OrderedDict
from pathlib import Path
import sys
import torch
import torchvision.transforms.v2.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'workstation'))
from family_resume import resume_order,select_checkpoint
from family_data import RealFamilyDataset
from real_robot.fast_training_augmentation import photometric_factors,augment_family_batch


def test_vectorized_photo_matches_existing_operations_for_frames_and_history():
    torch.manual_seed(7)
    for shape in [(3,3,11,13),(3,2,3,11,13)]:
        images=torch.rand(shape);factors=torch.tensor([[.9,1.1,.95],[1.1,.9,1.05],[1.,1.,1.]])
        expected=torch.stack([F.adjust_saturation(F.adjust_contrast(F.adjust_brightness(image,float(v[0])),float(v[1])),float(v[2])) for image,v in zip(images,factors)])
        actual=photometric_factors(images,factors)
        torch.testing.assert_close(actual,expected,atol=2e-7,rtol=1e-6)


def test_fast_photo_shares_factors_between_views_and_preserves_targets():
    image=torch.rand(4,2,3,8,8);image[:,1]=image[:,0]
    batch={'observation.state':torch.zeros(4,6),'observation.images.a':image,'observation.images.b':image.clone(),'action':torch.rand(4,20,7)}
    original=image.clone();out,_=augment_family_batch(batch,{'photometric':{'enabled':True}})
    torch.testing.assert_close(out['observation.images.a'],out['observation.images.b'])
    torch.testing.assert_close(out['observation.images.a'][:,0],out['observation.images.a'][:,1])
    torch.testing.assert_close(batch['observation.images.a'],original)
    assert out['action'] is batch['action']


def test_partial_sampler_resume_neither_skips_nor_repeats_consumed_anchors():
    weights=torch.arange(1,14,dtype=torch.double);n=len(weights);batch=4
    sampler=torch.utils.data.WeightedRandomSampler(weights,n,replacement=True,generator=torch.Generator().manual_seed(42))
    for _ in range(3):order=list(iter(sampler))
    snapshot={'epoch':3,'step':2*math.ceil(n/batch)+2,'rng':{'sampler':sampler.generator.get_state()}}
    epoch,remaining,consumed=resume_order(snapshot,weights,n,effective_batch=batch)
    assert epoch==3 and consumed==8 and remaining==order[8:]
    generator=torch.Generator();generator.set_state(snapshot['rng']['sampler'])
    following=torch.multinomial(weights,n,True,generator=generator).tolist()
    assert following==list(iter(sampler))


def test_completed_epoch_advances_without_replaying_that_epoch():
    snapshot={'epoch':2,'step':8,'rng':{}}
    assert resume_order(snapshot,torch.ones(13),13,completed_epoch=True,effective_batch=4)==(3,None,0)


def test_checkpoint_selection_uses_latest_raw_optimizer_state(tmp_path):
    common={'selected_weights':'raw','optimizer':{},'epoch':1}
    torch.save(dict(common,step=100,completed_epoch=True),tmp_path/'latest.pt')
    torch.save(dict(common,step=120,completed_epoch=False),tmp_path/'recovery.pt')
    path,state,completed=select_checkpoint(tmp_path)
    assert path.name=='recovery.pt' and state['step']==120 and completed is False


def test_worker_serialization_drops_native_decoder_handles():
    d=RealFamilyDataset.__new__(RealFamilyDataset);d.cache=OrderedDict({0:lambda:None});d.anchors=[(1,2)]
    restored=pickle.loads(pickle.dumps(d))
    assert not restored.cache and restored.anchors==d.anchors
