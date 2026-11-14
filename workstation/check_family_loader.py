"""Exercise spawned workers and recorded prompts before starting family training."""
import argparse,json
from pathlib import Path
import torch
from family_data import RealFamilyDataset
from train_family_finetune import worker_init


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();campaign=json.loads((root/'campaign.json').read_text());settings=campaign['family_performance']['smolvla_baseline'];report={}
    for split in ['train','validation']:
        data=RealFamilyDataset(campaign,'smolvla_baseline',split);order=torch.linspace(0,len(data)-1,64).long().tolist()
        loader=torch.utils.data.DataLoader(data,batch_size=settings['flow_microbatch'],sampler=order,num_workers=settings['loader_workers'],prefetch_factor=settings['prefetch_factor'],multiprocessing_context='spawn',worker_init_fn=worker_init)
        count=0
        for batch in loader:
            assert all(text==campaign['evaluation_instruction'] for text in batch['task'])
            assert torch.isfinite(batch['action']).all();count+=len(batch['task'])
        report[split]=dict(anchors=len(data),checked=count,prompt=campaign['evaluation_instruction'])
    report['passed']=True;(root/'loader_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
