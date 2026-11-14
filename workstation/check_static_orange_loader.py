"""Check spawned train/validation workers and exact recorded prompt; no devices."""
import argparse,json
from pathlib import Path
import torch
from smoke_real_finetune import LeRobotDataset
from cached_dom_data import CachedDOMData
from train_real_finetune import worker_init


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();c=json.loads((root/'launch_contract.json').read_text());report={}
    for split in ['train','validation']:
        base=LeRobotDataset(c['repo_id'],root=Path(c['local_verified_dataset']),split=split,revision=c['revision'],euler_delta_convention=c['euler_delta_convention'],required_features=['observation.images.opst_cam','observation.images.wrist_cam','observation.state','observation.environment_state','action'],delta_timestamps={'observation':[-2,0],'action':list(range(20))})
        data=CachedDOMData(base,c['decoded_video_cache'],instruction_mode=c['instruction_mode'],horizon=20,cache_bytes=c['decoder_cache_bytes'])
        order=torch.linspace(0,len(data)-1,64).long().tolist()
        loader=torch.utils.data.DataLoader(data,batch_size=c['microbatch'],sampler=order,num_workers=c['num_workers'],pin_memory=True,prefetch_factor=c['prefetch_factor'],multiprocessing_context='spawn',persistent_workers=False,worker_init_fn=worker_init)
        count=0
        for batch in loader:
            assert all(text==c['prompt'] for text in batch['task'])
            assert torch.isfinite(batch['action']).all();count+=len(batch['task'])
        report[split]=dict(episodes=len(base.episodes),eligible_anchors=len(data),spawned_worker_samples=count,prompt=c['prompt'])
        del loader,data,base
    report['passed']=True;(root/'loader_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
