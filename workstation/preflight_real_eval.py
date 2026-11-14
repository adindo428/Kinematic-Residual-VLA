"""Read saved observations and check learned action generation; no hardware access."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
from unittest.mock import patch
import time
import torch
from safetensors.torch import load_model
from smoke_real_finetune import DynamicVLAPolicy, get_policy_cfg, LeRobotDataset, AutoConfig, AutoTokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
    contract=json.loads((args.run/'launch_contract.json').read_text())
    bundle=Path('scratch/real_robot/finetuning-checkpoints').resolve()
    out=args.run/'evaluation';out.mkdir(exist_ok=True)
    report={'task':'striped-ball','hardware_accessed':False,'closed_loop_ready':False,
            'remaining':'Live camera/tracking/robot-action adapter and streaming integration have not been verified.', 'models':{}}
    for family in ['krv100','domv']:
        selected=json.loads((args.run/family/'best_checkpoint.json').read_text())
        source=Path(selected['path']);target=out/family;target.mkdir(exist_ok=True)
        config=json.loads((source/'config.json').read_text())
        config['delta_timestamps']={'observation':[-2,0],'action':list(range(config['chunk_size']))}
        config['use_checkpoint_normalization']=True
        (target/'config.json').write_text(json.dumps(config,indent=2)+'\n')
        weights=target/'model.safetensors'
        if not weights.exists():weights.symlink_to(source/'model.safetensors')
        cfg=get_policy_cfg(cfg_file=target/'config.json')
        cfg.enable_streaming=False
        with ExitStack() as stack:
            for cls in [AutoConfig,AutoTokenizer]:
                original=cls.from_pretrained
                def local(name,*pos,_original=original,**kw):
                    asset=bundle/'assets'/str(name)
                    if asset.is_dir():name=str(asset);kw['local_files_only']=True
                    return _original(name,*pos,**kw)
                stack.enter_context(patch.object(cls,'from_pretrained',side_effect=local))
            model=DynamicVLAPolicy(cfg)
        load_model(model,str(weights),strict=True)
        model.eval().cuda()
        data=LeRobotDataset(contract['repo_id'],root=Path(contract['local_verified_dataset']),split='validation',
            revision=contract['revision'],euler_delta_convention=contract['euler_delta_convention'],
            required_features=['observation.images.opst_cam','observation.images.wrist_cam','observation.state','observation.environment_state','action'],
            delta_timestamps={'observation':[-2,0],'action':list(range(cfg.chunk_size))})
        sample=None
        for i in range(len(data)):
            candidate=data[i]
            if 'ball' in str(candidate.get('task','')).lower():sample=candidate;break
        if sample is None:raise RuntimeError('No ball validation sample found')
        batch=torch.utils.data.default_collate([sample]);batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            torch.manual_seed(42)
            start=time.monotonic();action=model._get_action_chunk(model._prepare_batch(batch))
            torch.cuda.synchronize();elapsed=time.monotonic()-start
        assert torch.isfinite(action).all()
        report['models'][family]={'checkpoint':str(source),'evaluation_checkpoint':str(target.resolve()),
            'epoch':selected['epoch'],'validation_loss':selected['validation_loss'],
            'sample_task':str(sample['task']),'action_shape':list(action.shape),'finite_actions':True,
            'first_chunk_seconds_including_warmup':elapsed,'action_min':action.float().amin(dim=(0,1)).cpu().tolist(),
            'action_max':action.float().amax(dim=(0,1)).cpu().tolist(),
            'note':'Offline action chunk only, not closed-loop success or warmed LAAS latency.'}
        print(json.dumps(report['models'][family]),flush=True)
        del model,batch,action,data;torch.cuda.empty_cache()
    (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
