"""Disposable optimizer/backprop and paired inference check; never saves weights."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from smoke_real_finetune import build_training_components,PROJECT
import torch
import sys
sys.path.insert(0,str(PROJECT))
from real_robot.action_objective import sampled_absolute_actions,physical_loss,horizon_weights
from real_robot.training_augmentation import model_inputs
from utils.real_robot_contract import absolute_pose


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--family',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.contract.read_text());pref=next(x for x in c['checkpoint_preferences'] if x['checkpoint_id']==c['checkpoint'][a.family])
    model,cfg,data,_=build_training_components(SimpleNamespace(bundle=PROJECT/'scratch/real_robot/finetuning-checkpoints',
        dataset=Path(c['local_verified_dataset']),revision=c['revision'],family=a.family,
        checkpoint=Path(pref['path']),checkpoint_id=pref['checkpoint_id'],expected_sha256=pref['sha256']))
    model.cuda();torch.manual_seed(42)
    batch=next(iter(torch.utils.data.DataLoader(data,batch_size=2,num_workers=0)))
    batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()};batch=model_inputs(batch,cfg)
    noise=torch.randn(2,cfg.chunk_size,cfg.max_action_dim,device='cuda')
    model.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        model.reset();regular=model.predict_action_chunk(dict(batch),noise=noise)
        regular=absolute_pose(regular,batch['observation.state'][:,-1:,:],cfg.euler_delta_convention)
        explicit=sampled_absolute_actions(model,dict(batch),noise=noise)
    torch.testing.assert_close(regular.float(),explicit.float(),atol=2e-6,rtol=2e-6)
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'])
    optimizer.zero_grad();torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.bfloat16):flow,_=model(batch,horizon_weights=horizon_weights(c['action_objective'],cfg.chunk_size,'cuda'))
    flow.backward()
    optimizer.zero_grad() # Prove auxiliary alone reaches the learned model, including visual parameters.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        prediction=sampled_absolute_actions(model,batch,noise=noise)
        target=absolute_pose(batch['action'],batch['observation.state'][:,-1:,:],cfg.euler_delta_convention)
        aux,terms=physical_loss(prediction,target,~batch['action_is_pad'],c['action_objective'])
    aux.backward()
    gradients={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None and p.requires_grad}
    assert gradients and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    visual=sum(v for k,v in gradients.items() if 'vision' in k)
    assert visual>0,'Auxiliary loss must reach vision backbone'
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    result=dict(family=a.family,paired_decoding_max_error=float((regular-explicit).abs().max()),flow_loss=float(flow),
        auxiliary_loss=float(aux),auxiliary_terms=terms,gradient_parameters=len(gradients),vision_gradient_norm_sum=visual,
        finite_gradients=True,optimizer_step=True,weights_saved=False,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
