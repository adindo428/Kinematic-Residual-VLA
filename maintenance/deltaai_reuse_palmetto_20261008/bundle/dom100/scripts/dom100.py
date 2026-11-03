#!/usr/bin/env python3
"""Profile, pilot, smoke-test, train, and verify the matched six-run family."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.dom_experiment import VARIANTS, atomic_json, digest, select_recipe, choose_sampling


def approved_profile(root):
    from core.dom_profiling import gpu_key
    from core.dom_training import load_family
    path = Path(root)/"profiles"/f"{gpu_key()}.json"
    profile = json.loads(path.read_text())
    copy = dict(profile); expected = copy.pop("fingerprint")
    if digest(copy) != expected:
        raise ValueError("GPU profile was modified")
    if profile["manifest"] != load_family(root)["fingerprint"] or not profile.get("all_variants_verified"):
        raise ValueError("GPU profile is not approved for this frozen family")
    if (Path(root)/"recipe.json").exists():
        recipe = json.loads((Path(root)/"recipe.json").read_text())
        profile["runtime"] = profile["runtimes"][recipe["sampling"]]
        if profile["runtime"]["precision"] != recipe["precision"]:
            raise ValueError("This GPU needs a profile at the shared production precision")
    return profile


def smoke(root, variant, profile):
    import torch
    import safetensors.torch
    from core.dom_checkpoint import resolve_generation, save_generation
    from core.dom_training import (build_dataset, build_policy, fixed_noise, load_family, make_loader,
        make_optimizer, make_scheduler, seed_all, to_device, amp_context)
    from core.dom_profiling import check_gradients
    root = Path(root)
    seed_all()
    episodes = json.loads((root/"pilot_episodes.json").read_text())[:8]
    data = build_dataset(root, episodes, training=True)
    runtime = profile["runtime"]
    batch = to_device(next(iter(make_loader(data, {**runtime, "workers": 0}))))
    model = build_policy(root, variant)
    optimizer = make_optimizer(model, 1e-5, runtime["fused"])
    scheduler = make_scheduler(optimizer, 10000)
    with amp_context(runtime["precision"]):
        loss, _ = model(batch)
    loss.backward()
    norm = check_gradients(model)
    if model.config.condition_on_object_velocity:
        if model.model.state_proj.weight.grad[:, 6:9].abs().sum() == 0:
            raise ValueError("Neural velocity projection has no gradient")
    optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
    model.eval()
    noise = fixed_noise(batch, model, "cuda")
    with torch.no_grad(), amp_context(runtime["precision"]):
        model.reset()
        prediction = model.predict_action_chunk(batch, noise=noise)
        horizon = model.config.chunk_size
        if prediction.shape[1] != horizon or batch["action"].shape[1] != horizon or batch["action_is_pad"].shape[1] != horizon:
            raise ValueError("Decoded targets, padding, and prediction must share the configured horizon")
        if batch.get("dynamic_object.phase_gate") is not None and batch["dynamic_object.phase_gate"].shape[1] != horizon:
            raise ValueError("Phase gates must match the configured action horizon")
        poisoned = {**batch, "dynamic_object.future_velocity_xyz": torch.full((len(prediction), horizon, 3), 999., device="cuda"),
                    "dynamic_object.future_position_xyz": torch.full((len(prediction), horizon, 3), -999., device="cuda")}
        model.reset()
        alternate = model.predict_action_chunk(poisoned, noise=noise)
        torch.testing.assert_close(prediction, alternate)
    if not torch.isfinite(prediction).all():
        raise ValueError("Nonfinite inference")
    temporary = root/"smoke"/variant
    fingerprint = digest({"manifest": load_family(root)["fingerprint"], "variant": variant})
    save_generation(temporary, model, optimizer, scheduler, {"epoch": 0, "cursor": 352, "updates": 1, "samples": 352}, fingerprint, None)
    path = resolve_generation(temporary)
    # Strict reload must not require two full models plus optimizer states on
    # the profiled single GPU.
    drift_mean = model.config.residual_drift_mean
    loss_value, norm_value = float(loss.detach()), float(norm)
    model.cpu()
    del optimizer, scheduler, loss
    torch.cuda.empty_cache()
    fresh = build_policy(root, variant)
    safetensors.torch.load_model(fresh, str(path/"model.safetensors"), strict=True)
    fresh.config.residual_drift_mean = drift_mean
    with torch.no_grad(), amp_context(runtime["precision"]):
        fresh.reset()
        torch.testing.assert_close(prediction, fresh.predict_action_chunk(batch, noise=noise))
    report = {"variant": variant, "manifest": load_family(root)["fingerprint"], "gpu_key": profile["gpu_key"],
        "complete": True, "finite_loss": loss_value, "gradient_norm": norm_value,
        "profile": profile["fingerprint"], "runtime": runtime,
        "strict_reload": True, "future_inputs_ignored": True, "full_finetuning": True,
        "prediction_horizon": horizon, "execution_horizon": fresh.config.n_action_steps}
    atomic_json(root/"smoke"/f"{variant}-{profile['gpu_key']}.json", report)
    del fresh, model
    torch.cuda.empty_cache()
    shutil.rmtree(temporary)
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("profile", "profile-trial", "pilots", "choose", "smoke", "train", "verify", "preflight"):
        child = subs.add_parser(name)
        child.add_argument("--family-root", type=Path, required=True)
        if name in ("profile", "profile-trial"):
            child.add_argument("--output", type=Path, required=True)
        if name == "profile":
            child.add_argument("--precision", choices=("fp32", "bf16"), help="Require the shared production precision")
        if name in ("train", "profile-trial", "verify"):
            child.add_argument("--variant", choices=VARIANTS, required=True)
        if name == "profile-trial":
            child.add_argument("--runtime", type=json.loads, required=True)
            child.add_argument("--mode", choices=("gpu_only", "end_to_end"), required=True)
        if name == "smoke":
            child.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    root = args.family_root.resolve()
    if args.command == "profile-trial":
        from core.dom_profiling import trial
        try:
            result = trial(root, args.runtime, args.variant, args.mode)
        except Exception as exc:
            result = {"complete": False, "runtime": args.runtime, "oom": "out of memory" in str(exc).lower(), "error": str(exc)}
            traceback.print_exc()
            atomic_json(args.output, result)
            return 1
        atomic_json(args.output, result)
    elif args.command == "profile":
        from core.dom_profiling import profile
        result = profile(root, args.output, args.precision)
    elif args.command == "pilots":
        from core.dom_training import train
        profile = approved_profile(root)
        runtime = profile["runtime"]
        result = []
        def run_candidates(schemes, rates):
            for sampling in schemes:
                for lr in rates:
                    for variant in ("dom100", "kr100"):
                        destination = root/"pilots"/profile["gpu_key"]/f"{variant}-{sampling}-{lr:g}"
                        if (destination/"pilot.json").exists():
                            result.append(json.loads((destination/"pilot.json").read_text()))
                            shutil.rmtree(destination/"generations", ignore_errors=True)
                            (destination/"latest.json").unlink(missing_ok=True)
                            continue
                        runtime = profile["runtimes"][sampling]
                        pilot = {"lr": lr, "sampling": sampling, "max_updates": 2000, "output": str(destination)}
                        try:
                            row = train(root, variant, runtime, {"precision": runtime["precision"]}, pilot)
                        except (RuntimeError, ValueError) as exc:
                            if not any(word in str(exc).lower() for word in ("nonfinite", "non-finite")):
                                raise
                            row = {"variant": variant, "lr": lr, "sampling": sampling, "complete": True,
                                   "stable": False, "error": str(exc)}
                            atomic_json(destination/"pilot.json", row)
                        if row.get("interrupted"):
                            raise InterruptedError("Pilot checkpointed for continuation")
                        result.append(row)
                        shutil.rmtree(destination/"generations", ignore_errors=True)
                        (destination/"latest.json").unlink(missing_ok=True)
        try:
            run_candidates(("frames", "blocks"), (3e-5,))
            run_candidates((choose_sampling(result, profile["block_speedup"]),), (1e-5, 1e-4))
        except InterruptedError:
            return 75
        atomic_json(root/"pilots"/profile["gpu_key"]/"results.json", result)
    elif args.command == "choose":
        from core.dom_training import load_family
        # Selection can run on the login node without CUDA.
        candidates = sorted((root/"pilots").glob("*/results.json"))
        if len(candidates) != 1:
            raise ValueError("Exactly one complete tuning campaign must select the shared recipe")
        key = candidates[0].parent.name
        profile = json.loads((root/"profiles"/f"{key}.json").read_text())
        result = select_recipe(json.loads(candidates[0].read_text()), profile["block_speedup"])
        result.update(precision=profile["runtime"]["precision"], manifest=load_family(root)["fingerprint"], tuning_gpu=key)
        from utils.dom_experiment import file_digest
        result["pilot_results_sha256"] = file_digest(candidates[0])
        result["tuning_profile_sha256"] = file_digest(root/"profiles"/f"{key}.json")
        result["simulator_evaluation_mode"] = "disabled_on_deltaai"
        result["evaluation_plan_sha256"] = None
        result["fingerprint"] = digest(result)
        if (root/"recipe.json").exists() and json.loads((root/"recipe.json").read_text()) != result:
            raise ValueError("Shared production recipe is already frozen")
        atomic_json(root/"recipe.json", result)
    elif args.command == "smoke":
        profile = approved_profile(root)
        result = [smoke(root, variant, profile) for variant in ([args.variant] if args.variant else VARIANTS)]
    elif args.command == "train":
        from core.dom_training import train, load_family
        profile = approved_profile(root)
        recipe = json.loads((root/"recipe.json").read_text())
        check = dict(recipe); fingerprint = check.pop("fingerprint")
        if digest(check) != fingerprint or recipe["manifest"] != load_family(root)["fingerprint"]:
            raise ValueError("Shared recipe was modified or belongs to another family")
        smoke_file = root/"smoke"/f"{args.variant}-{profile['gpu_key']}.json"
        verified = json.loads(smoke_file.read_text())
        if not verified.get("complete") or verified["manifest"] != recipe["manifest"] or verified["profile"] != profile["fingerprint"]:
            raise ValueError("A real GPU smoke/reload test must pass before production")
        result = train(root, args.variant, profile["runtime"], recipe)
        if result.get("interrupted"):
            return 75
    elif args.command == "verify":
        import torch
        from core.dom_checkpoint import resolve_generation
        from core.dom_training import load_family
        destination = root/"runs"/args.variant
        path = resolve_generation(destination)
        state = torch.load(path/"training_state.pt", map_location="cpu", weights_only=False)
        progress = state["progress"]
        manifest = load_family(root)
        if progress["epoch"] != 100 or progress["cursor"] != 0 or progress["samples"] != 100*manifest["train_frames"]:
            raise ValueError("100 complete full-data epochs have not been verified")
        if progress["updates"] != 100*math.ceil(manifest["train_frames"]/352):
            raise ValueError("Optimizer update count differs from the complete-data schedule")
        if progress.get("pending_validation") or len(list(destination.glob("validation-epoch-*.json"))) != 100:
            raise ValueError("Epoch validation remains incomplete")
        steps = [int(item["step"]) for item in state["optimizer"]["state"].values()]
        if not steps or min(steps) != progress["updates"] or max(steps) != progress["updates"] or state["scheduler"]["last_epoch"] != progress["updates"]:
            raise ValueError("Weights, optimizer, and scheduler do not confirm the same completed updates")
        cfg = json.loads((path/"config.json").read_text())
        actual = (cfg["enable_residual_flow"], cfg["condition_on_object_velocity"], cfg["residual_phase_gate_mode"])
        if actual != VARIANTS[args.variant] or cfg["completed_epochs"] != 100:
            raise ValueError("Checkpoint architecture or epoch count differs from the requested run")
        result = {"complete": True, "variant": args.variant, "checkpoint": str(path), **progress}
        result["samples_per_second"] = progress["samples"]/progress["training_seconds"]
        atomic_json(destination/"verified_training.json", result)
    else:
        from core.dom_training import load_family
        manifest = load_family(root)
        split = json.loads((root/"split.json").read_text())["splits"]
        sets = [set(split[k]) for k in ("train", "validation", "test")]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)) or len(set.union(*sets)) != manifest["episodes"]:
            raise ValueError("Invalid full-dataset partition")
        from utils.dom_experiment import file_digest
        if file_digest(Path(manifest["initialization"])/"model.safetensors") != manifest["weights_sha256"]:
            raise ValueError("Pretrained initialization differs from the frozen weights")
        result = {"manifest": manifest["fingerprint"], "counts": manifest["counts"], "prepared": True}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
