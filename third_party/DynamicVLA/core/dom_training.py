"""Matched full-DOM training, shared by profiling, pilots, and production."""
import contextlib
import json
import logging
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch

from core.dom_checkpoint import load_generation, run_lock, save_generation
from utils.dom_data import DOMData, EpochBatchSampler
from utils.dom_experiment import VARIANTS, atomic_json, digest, file_digest, source_digest, simulator_evaluation_mode, experiment_horizons

FEATURES = ["observation.images.opst_cam", "observation.images.wrist_cam",
            "observation.state", "observation.environment_state", "action"]
PROJECT = Path(__file__).resolve().parents[1]


def load_family(root, check_source=True):
    root = Path(root)
    manifest = json.loads((root/"manifest.json").read_text())
    fingerprint = manifest.pop("fingerprint")
    if digest(manifest) != fingerprint:
        raise ValueError("Frozen manifest was modified")
    manifest["fingerprint"] = fingerprint
    if file_digest(root/"split.json") != manifest["split_sha256"]:
        raise ValueError("Frozen split was modified")
    for name, expected in manifest.get("artifacts", {}).items():
        if file_digest(root/name) != expected:
            raise ValueError(f"Frozen artifact was modified: {name}")
    if check_source and source_digest(PROJECT) != manifest["source_sha256"]:
        raise ValueError("Source differs from the frozen family; prepare a new family")
    return manifest


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_dataset(root, episodes, training=False):
    from utils.datasets import LeRobotDataset, ImageTransforms
    root = Path(root)
    manifest = load_family(root, check_source=False)
    horizon, _ = experiment_horizons(manifest)
    transforms = {"BRIGHTNESS": [.8, 1.2], "CONTRAST": [.8, 1.2], "SATURATION": [.5, 1.5], "HUE": [-.05, .05]} if training else {}
    base = LeRobotDataset("hzxie/DOM", root=manifest["dataset_root"], episodes=episodes,
        revision=manifest["dataset_revision"],
        pin_memory=False, delta_action=True, required_features=FEATURES,
        image_transforms=ImageTransforms(None, transforms),
        delta_timestamps={"observation": [-2, 0], "action": list(range(horizon))})
    return DOMData(base, horizon=horizon, phase_config=manifest["phase_config"], phase_cache=root/"phase_cache", frozen_files=root)


def build_policy(root, variant, device="cuda"):
    from utils.helpers import get_policy_cfg
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy, rename_checkpoint_keys, standardise_state_dict
    import safetensors.torch
    root = Path(root)
    manifest = load_family(root, check_source=False)
    horizon, execution_horizon = experiment_horizons(manifest)
    initialization = Path(manifest["initialization"])
    config = get_policy_cfg(cfg_file=initialization/"config.json")
    residual, velocity, phase = VARIANTS[variant]
    for key, value in {"n_obs_steps": 2, "chunk_size": horizon, "n_action_steps": execution_horizon,
        "num_steps": 10, "resize_imgs_with_padding": (384, 384), "temporal_fusion": "attn",
        "freeze_vision_model": False, "freeze_connector": False, "freeze_text_model": False,
        "train_state_proj": True, "enable_streaming": False, "use_delta_action": True,
        "enable_residual_flow": residual, "residual_flow_formulation": "coordinate",
        "condition_on_object_velocity": velocity, "residual_lambda_type": "constant",
        "residual_phase_gate_mode": phase, "residual_phase_gate_config": manifest["phase_config"],
        "residual_drift_mode": "trajectory", "residual_drift_scale": 1.,
        "residual_action_dt": .04, "residual_axis_scale_xyz": (1., 1., 1.),
        "residual_normalize_drift": True, "residual_center_drift": True,
        "residual_estimate_drift_mean": False, "use_checkpoint_normalization": True,
        "velocity_input_mean": manifest["velocity_mean"], "velocity_input_std": manifest["velocity_std"]}.items():
        setattr(config, key, value)
    config.validate_residual_formulation()
    # Environment fields are available to the analytic/gate path, not the
    # neural prefix. Preserve original pretrained normalization module keys.
    policy = DynamicVLAPolicy(config)
    weights = safetensors.torch.load_file(str(initialization/"model.safetensors"))
    weights = rename_checkpoint_keys(weights, "model._orig_mod.//model.")
    weights, _ = standardise_state_dict(weights, set(policy.state_dict()))
    policy.load_state_dict(weights, strict=True)
    if not all(p.requires_grad for name, p in policy.named_parameters() if name.startswith("model.")):
        raise ValueError("Full fine-tuning unexpectedly contains frozen model parameters")
    std = policy.get_action_normalization_std(torch.device("cpu"), torch.float32)
    if std is None or std.numel() < 3 or not torch.isfinite(std).all() or (std[:3] <= 0).any():
        raise ValueError("Missing or invalid pretrained action normalization")
    drift = torch.zeros(horizon, config.max_action_dim)
    drift[:, :3] = torch.arange(horizon)[:, None]*.04*torch.tensor(manifest["velocity_mean"])[None, :]/std[:3]
    config.residual_drift_mean = drift.tolist()
    return policy.to(device)


def worker_init(_):
    torch.set_num_threads(1)


def make_loader(data, runtime, epoch=0, cursor=0, sampling="frames", indices=None):
    kwargs = dict(num_workers=runtime["workers"], pin_memory=runtime["pin_memory"],
                  generator=torch.Generator().manual_seed(42+epoch), worker_init_fn=worker_init)
    if runtime["workers"]:
        kwargs.update(prefetch_factor=runtime["prefetch"], persistent_workers=True, multiprocessing_context="spawn")
    if indices is not None:
        return torch.utils.data.DataLoader(torch.utils.data.Subset(data, indices), batch_size=runtime["microbatch"], shuffle=False, **kwargs)
    sampler = EpochBatchSampler(data.lengths, runtime["microbatch"], epoch, cursor, sampling=sampling)
    return torch.utils.data.DataLoader(data, batch_sampler=sampler, **kwargs)


def to_device(batch, device="cuda"):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def amp_context(precision, device="cuda"):
    if precision not in ("fp32", "bf16"):
        raise ValueError("Unsupported precision")
    return torch.autocast(device_type=str(device).split(":")[0], dtype=torch.bfloat16, enabled=precision=="bf16")


def make_optimizer(policy, lr, fused=False):
    return torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=lr,
                            eps=1e-8, weight_decay=1e-10, betas=(.9, .95), fused=fused)


def make_scheduler(optimizer, total_updates, warmup=1000):
    def multiplier(update):
        if update < warmup:
            return update/max(1, warmup)
        progress = min(1., (update-warmup)/max(1, total_updates-warmup))
        return .5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def fixed_noise(batch, policy, device):
    return torch.stack([torch.randn(policy.config.chunk_size, policy.config.max_action_dim,
        generator=torch.Generator().manual_seed(42+int(idx))) for idx in batch["sample_index"].cpu()]).to(device)


@torch.no_grad()
def evaluate(policy, loader, precision, device="cuda", should_stop=None):
    policy.eval()
    sums, counts = {}, {}
    prediction_horizon = policy.config.chunk_size
    execution_horizon = getattr(policy.config, "n_action_steps", min(20, prediction_horizon))
    horizons = sorted({execution_horizon, prediction_horizon})
    for batch in loader:
        if should_stop is not None and should_stop():
            raise InterruptedError("Validation interrupted at a batch boundary")
        batch = to_device(batch, device)
        policy.reset()
        with amp_context(precision, device):
            # Evaluate the full predicted chunk before online queue selection.
            prediction = policy.predict_action_chunk(batch, noise=fixed_noise(batch, policy, device))
        if prediction.shape[1] != prediction_horizon or not torch.isfinite(prediction).all():
            raise ValueError(f"Evaluation must return a finite full {prediction_horizon}-action chunk")
        if batch["action"].shape[1] != prediction_horizon or batch["action_is_pad"].shape[1] != prediction_horizon:
            raise ValueError("Validation targets and padding must match the prediction horizon")
        valid = ~batch["action_is_pad"].bool()
        error = (prediction.float()-batch["action"].float()).square()
        for horizon in horizons:
            mask = valid[:, :horizon]
            for label, channels in (("action_mse", slice(0, 7)), ("translation_mse", slice(0, 3)),
                                    ("orientation_mse", slice(3, 6)), ("gripper_mse", slice(6, 7))):
                values = error[:, :horizon, channels]
                key = f"h{horizon}/{label}"
                sums[key] = sums.get(key, 0)+float((values*mask[..., None]).sum())
                counts[key] = counts.get(key, 0)+int(mask.sum())*values.shape[-1]
            grip = (prediction[:, :horizon, 6] <= 0) == (batch["action"][:, :horizon, 6] <= 0)
            key = f"h{horizon}/gripper_accuracy"
            sums[key] = sums.get(key, 0)+float((grip*mask).sum())
            counts[key] = counts.get(key, 0)+int(mask.sum())
        for phase in torch.unique(batch["dynamic_object.phase_id"]).tolist():
            select = batch["dynamic_object.phase_id"] == phase
            for horizon in horizons:
                mask = valid[select, :horizon]
                for label, channels in (("action_mse", slice(0, 7)), ("translation_mse", slice(0, 3)),
                                        ("orientation_mse", slice(3, 6)), ("gripper_mse", slice(6, 7))):
                    values = error[select, :horizon, channels]
                    key = f"phase{phase}/h{horizon}/{label}"
                    sums[key] = sums.get(key, 0)+float((values*mask[..., None]).sum())
                    counts[key] = counts.get(key, 0)+int(mask.sum())*values.shape[-1]
    policy.train()
    return {key: sums[key]/counts[key] for key in sums if counts[key]}


def train(root, variant, runtime, recipe, pilot=None):
    """pilot={lr,sampling,max_updates,output}; production has no update cap."""
    root = Path(root)
    manifest = load_family(root)
    split = json.loads((root/"split.json").read_text())["splits"]
    seed_all()
    production = pilot is None
    if production:
        simulator_evaluation_mode(root, recipe)
    if runtime["precision"] != recipe["precision"]:
        raise ValueError("GPU profile precision differs from the shared recipe")
    lr, sampling = (recipe["lr"], recipe["sampling"]) if production else (pilot["lr"], pilot["sampling"])
    destination = root/"runs"/variant if production else Path(pilot["output"])
    fingerprint = digest({"manifest": manifest["fingerprint"], "recipe": recipe if production else pilot,
                          "variant": variant, "precision": runtime["precision"]})
    episodes = split["train"] if production else json.loads((root/"pilot_episodes.json").read_text())
    with run_lock(destination):
        data = build_dataset(root, episodes, training=True)
        val = build_dataset(root, split["validation"])
        val_loader = make_loader(val, runtime, indices=json.loads((root/"validation_anchors.json").read_text()))
        policy = build_policy(root, variant)
        policy.config.optimizer_lr = lr
        policy.config.scheduler_warmup_steps = 1000
        policy.config.scheduler_decay_steps = 100*math.ceil(manifest["train_frames"]/352)
        optimizer = make_optimizer(policy, lr, runtime["fused"])
        scheduler = make_scheduler(optimizer, policy.config.scheduler_decay_steps)
        progress = {"epoch": 0, "cursor": 0, "updates": 0, "samples": 0, "pilot_history": []}
        best, wandb_id = None, None
        if (destination/"latest.json").exists():
            state = load_generation(destination, policy, optimizer, scheduler, fingerprint)
            progress, best, wandb_id = state["progress"], state["best"], state["wandb_id"]
        elif (destination/"provenance.json").exists():
            previous = json.loads((destination/"provenance.json").read_text())
            if previous["fingerprint"] != fingerprint:
                raise ValueError("Output directory belongs to a different experiment")
            wandb_id = previous["wandb_id"]
        if production:
            enqueue_evaluations(root, destination, variant, progress, best)
        if production and progress["epoch"] >= 100 and not progress.get("pending_validation"):
            atomic_json(destination/"training_complete.json", {**progress, "fingerprint": fingerprint, "best": best})
            return progress
        writer = None
        if production:
            import wandb
            wandb_id = wandb_id or wandb.util.generate_id()
            writer = wandb.init(project="Kinematic-Residual-VLA", entity=os.environ.get("DVLA_WANDB_ENTITY", "aparame-clemson-university"),
                name=variant, id=wandb_id, resume="allow", dir=str(destination),
                config={"manifest": manifest, "recipe": recipe, "variant": variant, "runtime": runtime})
        atomic_json(destination/"provenance.json", {"fingerprint": fingerprint, "wandb_id": wandb_id,
                    "variant": variant, "manifest": manifest["fingerprint"], "runtime": runtime, "recipe": recipe})
        stop = {"requested": False}
        def request_stop(signum, frame):
            stop["requested"] = True
        previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGUSR1, signal.SIGTERM)}
        last_save = time.monotonic()
        pilot_history = progress.setdefault("pilot_history", [])
        def finish_epoch():
            nonlocal best, last_save
            metrics = evaluate(policy, val_loader, runtime["precision"], should_stop=lambda: stop["requested"])
            atomic_json(destination/f"validation-epoch-{progress['epoch']:03d}.json", metrics)
            value = metrics["h20/action_mse"]
            if best is None or value < best["value"]:
                best = {"value": value, "epoch": progress["epoch"], "updates": progress["updates"]}
            if writer:
                writer.log({"Validation/"+k: v for k,v in metrics.items()}, step=progress["updates"])
            milestone = progress["epoch"] in (1, 5, 100) or progress["epoch"] % 10 == 0
            progress["pending_validation"] = False
            save_generation(destination, policy, optimizer, scheduler, progress.copy(), fingerprint, wandb_id, best, milestone)
            last_save = time.monotonic()
            if milestone:
                enqueue_evaluations(root, destination, variant, progress, best)
        def finish_pilot_validation():
            values = progress.setdefault("pilot_validation", {})
            update = str(progress["updates"])
            if progress["updates"] in (1500, 2000) and update not in values:
                values[update] = evaluate(policy, val_loader, runtime["precision"],
                    should_stop=lambda: stop["requested"])["h20/action_mse"]
        def pilot_report():
            values = progress["pilot_validation"]
            report = {"variant": variant, "lr": lr, "sampling": sampling, "complete": True,
                "initial_mse": progress["pilot_initial_mse"], "final_mse": float(np.mean(list(values.values()))),
                "validation_history": values, "updates": progress["updates"]}
            if set(values) != {"1500", "2000"}:
                raise ValueError("Pilot is missing its fixed validation measurements")
            atomic_json(destination/"pilot.json", report)
            return report
        try:
            if not production and "pilot_initial_mse" not in progress:
                if progress["updates"]:
                    raise ValueError("Resumed pilot lacks its pretrained validation baseline")
                progress["pilot_initial_mse"] = evaluate(policy, val_loader, runtime["precision"],
                    should_stop=lambda: stop["requested"])["h20/action_mse"]
            if not production:
                finish_pilot_validation()
                if progress["updates"] >= pilot["max_updates"]:
                    return pilot_report()
            if not (destination/"latest.json").exists():
                save_generation(destination, policy, optimizer, scheduler, progress.copy(), fingerprint, wandb_id, best)
            if production and progress.get("pending_validation"):
                finish_epoch()
            optimizer.zero_grad(set_to_none=True)
            while progress["epoch"] < 100:
                loader = make_loader(data, runtime, progress["epoch"], progress["cursor"], sampling)
                group_samples = min(352, len(data)-progress["cursor"])
                group_consumed, group_loss = 0, 0.
                update_start = time.monotonic()
                for batch in loader:
                    batch = to_device(batch)
                    n = batch["action"].shape[0]
                    with amp_context(runtime["precision"]):
                        loss, _ = policy(batch)
                    (loss*n/group_samples).backward()
                    group_loss += loss.detach()*n
                    group_consumed += n
                    progress["cursor"] += n
                    progress["samples"] += n
                    if group_consumed < group_samples:
                        continue
                    from core.dom_profiling import check_gradients
                    check_gradients(policy)
                    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                    progress["updates"] += 1
                    progress["training_seconds"] = progress.get("training_seconds", 0.)+time.monotonic()-update_start
                    if progress["updates"] % 10 == 0:
                        logging.info("%s epoch=%.5f updates=%d samples=%d loss=%.6g lr=%.6g",
                            variant, progress["epoch"]+progress["cursor"]/len(data), progress["updates"],
                            progress["samples"], float(group_loss/group_samples), optimizer.param_groups[0]["lr"])
                    if writer and progress["updates"] % 10 == 0:
                        writer.log({"Train/loss": float(group_loss/group_samples),
                            "Train/lr": optimizer.param_groups[0]["lr"], "Train/samples": progress["samples"],
                            "Train/optimizer_updates": progress["updates"],
                            "Train/epoch": progress["epoch"]+progress["cursor"]/len(data),
                            "Train/samples_per_second": group_samples/max(time.monotonic()-update_start, 1e-8)}, step=progress["updates"])
                    group_samples = min(352, len(data)-progress["cursor"])
                    group_consumed, group_loss = 0, 0.
                    update_start = time.monotonic()
                    if not production and progress["updates"] in (1500, 2000):
                        finish_pilot_validation()
                    if stop["requested"] or time.monotonic()-last_save >= 1800:
                        save_generation(destination, policy, optimizer, scheduler, progress.copy(), fingerprint, wandb_id, best)
                        last_save = time.monotonic()
                    if stop["requested"]:
                        return {**progress, "interrupted": True}
                    if not production and progress["updates"] >= pilot["max_updates"]:
                        return pilot_report()
                if progress["cursor"] != len(data):
                    raise RuntimeError("Epoch omitted training samples")
                progress["epoch"] += 1
                progress["cursor"] = 0
                if not production:
                    continue
                progress["pending_validation"] = True
                save_generation(destination, policy, optimizer, scheduler, progress.copy(), fingerprint, wandb_id, best)
                finish_epoch()
            enqueue_evaluations(root, destination, variant, progress, best)
            atomic_json(destination/"training_complete.json", {**progress, "fingerprint": fingerprint, "best": best})
            return progress
        except InterruptedError:
            save_generation(destination, policy, optimizer, scheduler, progress.copy(), fingerprint, wandb_id, best)
            return {**progress, "interrupted": True}
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
            if writer:
                writer.finish(exit_code=0 if not stop["requested"] and progress["epoch"]==100 else 1)


def enqueue_evaluations(root, destination, variant, progress, best):
    """Recover requests even if preemption fell between checkpoint and dispatch."""
    from core.dom_checkpoint import resolve_generation
    from scripts.dom_closed_loop import dispatch
    requests = []
    for pointer in sorted(destination.glob("epoch-*.json")):
        epoch = int(pointer.stem.split("-")[1])
        request = destination/f"evaluation-epoch-{epoch:03d}.json"
        item = {"checkpoint": str(resolve_generation(destination, pointer.name)), "variant": variant,
                "epoch": epoch, "split": "validation"}
        requests.append((request, item))
    if progress["epoch"] == 100 or progress.get("early_stopped"):
        final_pointer = "final.json" if (destination/"final.json").exists() else "epoch-100.json"
        for label, pointer in (("best", "best.json"), ("final", final_pointer)):
            requests.append((destination/f"evaluation-test-{label}.json",
                {"checkpoint": str(resolve_generation(destination, pointer)), "variant": variant,
                 "epoch": progress["epoch"] if label == "final" else best["epoch"], "split": "test", "selection": label}))
    for request, item in requests:
        if not request.exists():
            atomic_json(request, item)
        if simulator_evaluation_mode(root) == "after_training":
            continue
        try:
            dispatch(root, request)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            logging.error("Manipulation evaluation submission pending: %s", exc)
