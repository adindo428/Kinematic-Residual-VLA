"""Native sampler/gradient checks and disposable SmolVLA throughput updates."""
import argparse,json,time,statistics,gc
from pathlib import Path
import torch
from torch.utils.data import default_collate
from family_data import RealFamilyDataset
from family_policy import load_released_policy
from train_family_finetune import disposable_check,cuda,slice_batch
from real_robot.action_objective import physical_loss,horizon_weights
from real_robot.training_ema import TrainingEMA
from utils.real_robot_contract import absolute_pose


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();campaign=json.loads((root/'campaign.json').read_text());c=campaign['training_recipe'];entry=campaign['entries'][0];stats=json.loads((root/'real_statistics.json').read_text());objective=c['action_objective']
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
    model=load_released_policy(root,entry,stats);data=RealFamilyDataset(campaign,'smolvla_baseline')
    disposable_check(model,data,objective,root/'smolvla_baseline-gpu-readiness.json')
    batch=cuda(default_collate([data[int(i)] for i in torch.randperm(len(data),generator=torch.Generator().manual_seed(42))[:16]]));rows=[]
    parameters=[p for p in model.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(parameters,lr=c['learning_rate'],betas=(.9,.95),weight_decay=c['weight_decay'],fused=True);ema=TrainingEMA(model)
    def update(micro,auxiliary):
        model.train();optimizer.zero_grad(set_to_none=True)
        for start in range(0,16,micro):
            b=slice_batch(batch,start,start+micro)
            with torch.autocast('cuda',dtype=torch.bfloat16):flow=model(b,horizon_weights(objective,20,'cuda'))
            if not torch.isfinite(flow):raise RuntimeError('Nonfinite flow loss')
            (flow*len(b['action'])/16).backward()
        if auxiliary:
            b=slice_batch(batch,0,objective['auxiliary_batch_size']);target=absolute_pose(b['action'],b['reference_state'][:,None,:],'wrapped_xyz_v1')
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=physical_loss(model.sample_absolute(b),target,~b['action_is_pad'],objective)
            (loss*objective['auxiliary_weight']).backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step();ema.update(model)
    for micro in [8,16]:
        try:
            torch.cuda.reset_peak_memory_stats();update(micro,True);torch.cuda.synchronize();times=[]
            for step in range(4):
                start=time.perf_counter();update(micro,step%4==0);torch.cuda.synchronize();times.append(time.perf_counter()-start)
            row=dict(flow_microbatch=micro,auxiliary_microbatch=2,effective_batch=16,passed=True,samples_per_second=16/statistics.mean(times),update_seconds=times,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        except torch.OutOfMemoryError:
            row=dict(flow_microbatch=micro,passed=False,reason='CUDA OOM');model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        rows.append(row);print(json.dumps(row),flush=True)
        if not row['passed']:break
    best=max([r for r in rows if r['passed']],key=lambda r:r['samples_per_second']);campaign['family_performance']['smolvla_baseline'].update(flow_microbatch=best['flow_microbatch'],auxiliary_microbatch=2)
    (root/'campaign.json').write_text(json.dumps(campaign,indent=2)+'\n');(root/'benchmark.json').write_text(json.dumps(dict(passed=True,weights_saved=False,initialization='upstream_only',selected=best,candidates=rows),indent=2)+'\n')


if __name__=='__main__':main()
