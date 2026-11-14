"""One disposable real-data optimizer step; never saves or modifies weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from contextlib import ExitStack
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'third_party/DynamicVLA'))

import numpy as np
import torch
from safetensors.torch import load_model
from lerobot.policies.normalize import Normalize, Unnormalize
from utils.datasets import LeRobotDataset
from utils.helpers import get_policy_cfg, fix_0std_dataset_stats
from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
from transformers import AutoConfig, AutoTokenizer


def build_training_components(args):
    bundle = args.bundle.resolve()
    if getattr(args,'checkpoint',None):
        directory=Path(args.checkpoint).resolve()
        item=dict(id=getattr(args,'checkpoint_id',None) or directory.name,path=str(directory))
        checksums=getattr(args,'expected_sha256',None)
        if not checksums:raise ValueError('Direct checkpoint initialization requires pinned config/weight SHA256 values')
    else:
        index = json.loads((bundle/'model-index.json').read_text())
        item = next(c for c in index['checkpoints'] if c['id'] == index['aliases'][args.alias])
        directory = bundle/item['path']
        checksums = {line[66:]:line[:64] for line in (bundle/'SHA256SUMS').read_text().splitlines()}
    for name in ['config.json', 'model.safetensors']:
        path = directory/name
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None
        if digest is None:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            digest = h.hexdigest()
        expected=checksums[name] if getattr(args,'checkpoint',None) else checksums[str(path.relative_to(bundle))]
        assert digest == expected, name
    cfg = get_policy_cfg(cfg_file=directory/'config.json')
    family=getattr(args,'family',None)
    assert cfg.chunk_size == 20
    # Preserve the selected checkpoints' FastViT architecture during profiling.
    assert cfg.vlm_model_name == 'HuggingFaceTB/SmolLM2-360M'
    assert cfg.fastvlm_inference_mode is True
    assert cfg.freeze_vision_model is False
    # Continuous inference / LAAS is an execution policy, not a training worker.
    cfg.enable_streaming = False
    assert cfg.n_obs_steps==2
    cfg.delta_timestamps={'observation':[-2,0],'action':list(range(cfg.chunk_size))}
    cfg.use_checkpoint_normalization=True  # Construct locally; strict load restores native statistics first.
    cfg.device = 'cuda'
    with ExitStack() as stack:
        for cls in [AutoConfig, AutoTokenizer]:
            original = cls.from_pretrained
            def local_asset(name, *pos, _original=original, **kw):
                path = bundle/'assets'/str(name)
                if path.is_dir():
                    name = str(path)
                    kw['local_files_only'] = True
                return _original(name, *pos, **kw)
            stack.enter_context(patch.object(cls,'from_pretrained',side_effect=local_asset))
        model = DynamicVLAPolicy(cfg)
    load_model(model, str(directory/'model.safetensors'), strict=True)
    print('Checkpoint hashes and strict weight loading passed',flush=True)
    # Load the common DOM weights strictly before enabling parameter-free method
    # differences. Velocity uses existing padded state-projection slots; KRV
    # changes analytical drift/flow coordinates, not the parameter architecture.
    overrides=getattr(args,'method_overrides',None) or {}
    allowed={'condition_on_object_velocity','enable_residual_flow','residual_flow_formulation',
        'residual_phase_gate_mode','residual_phase_gate_config','residual_lambda_type','residual_drift_scale','residual_drift_mode',
        'residual_action_dt','residual_axis_scale_xyz','residual_normalize_drift',
        'residual_center_drift','residual_estimate_drift_mean'}
    if set(overrides)-allowed:raise ValueError('Unsupported method override')
    for key,value in overrides.items():setattr(cfg,key,value)
    cfg.validate_residual_formulation()
    expected={'dom':(False,False),'domv':(True,False),'krv100':(True,True)}
    if family is not None:
        assert (cfg.condition_on_object_velocity,cfg.enable_residual_flow)==expected[family],family
    if cfg.enable_residual_flow:assert cfg.residual_phase_gate_mode in ('constant','causal_rules')
    stats = fix_0std_dataset_stats({k:{n:np.asarray(v) for n,v in d.items()}
        for k,d in json.loads((args.dataset/'meta/train_stats.json').read_text()).items()})
    # New real-data experiment, not exact continuation of the simulation run.
    cfg.euler_delta_convention = 'wrapped_xyz_v1'
    cfg.require_tracking_quality = True
    cfg.use_checkpoint_normalization = False
    cfg.velocity_input_mean = stats['observation.environment_state']['mean'][6:9].tolist()
    cfg.velocity_input_std = np.maximum(stats['observation.environment_state']['std'][6:9],1e-6).tolist()
    model.normalize_inputs = Normalize(cfg.input_features,cfg.normalization_mapping,stats)
    model.normalize_targets = Normalize(cfg.output_features,cfg.normalization_mapping,stats)
    model.unnormalize_outputs = Unnormalize(cfg.output_features,cfg.normalization_mapping,stats)
    drift = np.zeros((cfg.chunk_size,cfg.max_action_dim))
    drift[:,:3] = np.arange(cfg.chunk_size)[:,None]*cfg.residual_action_dt*np.asarray(cfg.velocity_input_mean)/stats['action']['std'][:3]
    cfg.residual_drift_mean = drift.tolist()
    data = LeRobotDataset('ailab-Clemson/KRVLA-dataset',root=args.dataset,split='train',
        revision=args.revision,euler_delta_convention='wrapped_xyz_v1',
        required_features=['observation.images.opst_cam','observation.images.wrist_cam',
                           'observation.state','observation.environment_state','action'],
        delta_timestamps={'observation':[-2,0],'action':list(range(cfg.chunk_size))})
    return model, cfg, data, item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alias', required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--microbatch', type=int, default=1)
    parser.add_argument('--profile-steps', type=int, default=1)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    model, cfg, data, item = build_training_components(args)
    torch.manual_seed(42)
    loader_kwargs = dict(batch_size=args.microbatch,num_workers=args.workers,
                         shuffle=False,pin_memory=True,
                         generator=torch.Generator().manual_seed(42))
    if args.workers:
        loader_kwargs.update(prefetch_factor=2,multiprocessing_context='spawn')
    count=args.microbatch*(args.warmup_steps+args.profile_steps)
    indices=(torch.randperm(len(data),generator=torch.Generator().manual_seed(42))[:count].tolist()
             if args.profile_steps>1 else list(range(count)))
    loader = iter(torch.utils.data.DataLoader(torch.utils.data.Subset(data,indices),**loader_kwargs))
    model.train().cuda()
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.95))
    torch.cuda.reset_peak_memory_stats()
    timings=[];losses=[]
    for step in range(args.warmup_steps+args.profile_steps):
        torch.cuda.synchronize();start=time.monotonic()
        batch=next(loader)
        batch = {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
        torch.cuda.synchronize();loaded=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,_ = model(batch)
        assert torch.isfinite(loss)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize();done=time.monotonic()
        if step>=args.warmup_steps:
            timings.append(dict(loader_s=loaded-start,compute_s=done-loaded,total_s=done-start))
            losses.append(float(loss.detach()))
    # Exhaust the finite profile subset, allowing decoder workers to stop cleanly.
    try:
        next(loader)
        raise RuntimeError('Profile loader unexpectedly has extra samples')
    except StopIteration:
        pass
    report = dict(checkpoint=item['id'],strict_weights_loaded=True,constant_phase_gate=True,
        dataset_revision=args.revision,usable_train_samples=len(data),microbatch=args.microbatch,
        dataset_path=str(args.dataset.resolve()),
        quality_policy_sha256=hashlib.sha256((args.dataset/'meta/quality_policy.json').read_bytes()).hexdigest(),
        loss=float(loss.detach()),finite_gradients=True,optimizer_step=True,
        parameters=sum(p.numel() for p in model.parameters()),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        step_seconds=float(np.mean([t['total_s'] for t in timings])),weights_saved=False,
        workers=args.workers,loader_shutdown_clean=True,warmup_steps=args.warmup_steps,profile_steps=args.profile_steps,
        step_timings=timings,losses=losses,
        samples_per_second=args.microbatch/float(np.mean([t['total_s'] for t in timings])),
        compute_samples_per_second=args.microbatch/float(np.mean([t['compute_s'] for t in timings])))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':main()
