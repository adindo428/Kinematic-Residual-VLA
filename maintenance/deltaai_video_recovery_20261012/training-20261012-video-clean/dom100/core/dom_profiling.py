"""Isolated real-data GPU trials; profiles are tied to hardware and source."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from core.dom_training import (amp_context, build_dataset, build_policy, load_family,
    make_loader, make_optimizer, seed_all, to_device)
from utils.dom_experiment import MICROBATCHES, VARIANTS, atomic_json, digest


def gpu_identity():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This workflow requires exactly one allocated CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if not any(name in properties.name.upper() for name in ("A100", "H100", "H200")):
        raise RuntimeError(f"Unapproved GPU class: {properties.name}")
    return {"name": properties.name, "total_memory": properties.total_memory,
        "capability": [properties.major, properties.minor], "torch": torch.__version__, "cuda": torch.version.cuda}


def gpu_key():
    return digest(gpu_identity())[:16]


class Telemetry:
    def __init__(self):
        self.stop = threading.Event()
        self.gpu_util, self.gpu_memory, self.ram = [], [], []
        self.thread = threading.Thread(target=self._collect, daemon=True)

    def _collect(self):
        import psutil
        process = psutil.Process()
        while not self.stop.is_set():
            try:
                self.ram.append(process.memory_info().rss+sum(p.memory_info().rss for p in process.children(recursive=True)))
                device = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                result = subprocess.run(["nvidia-smi", "-i", device, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
                # CUDA_VISIBLE_DEVICES may be a numeric physical index or UUID;
                # query that device explicitly when Slurm supplies it.
                first = result.stdout.strip().splitlines()[0]
                util, memory = map(float, first.split(","))
                self.gpu_util.append(util); self.gpu_memory.append(memory*1024**2)
            except (OSError, ValueError, subprocess.SubprocessError, psutil.Error):
                pass
            self.stop.wait(.5)

    def __enter__(self):
        self.thread.start(); return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(timeout=2)


def check_gradients(policy):
    # One combined finite check; no Python scalar read per parameter.
    maximum = getattr(getattr(policy, "config", None), "optimizer_grad_clip_norm", 10.)
    return torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],
                                        maximum, error_if_nonfinite=True, foreach=True)


def trial(root, runtime, variant="krvp100", mode="end_to_end", warmup=1, updates=3):
    seed_all()
    root = Path(root)
    manifest = load_family(root)
    episodes = json.loads((root/"pilot_episodes.json").read_text())
    data = build_dataset(root, episodes, training=True)
    loader = make_loader(data, runtime, sampling=runtime["sampling"])
    policy = build_policy(root, variant)
    optimizer = make_optimizer(policy, 1e-5, runtime["fused"])
    iterator = iter(loader)
    template = to_device(next(iterator)) if mode == "gpu_only" else None
    accumulation = 352//runtime["microbatch"]
    times, data_wait, first_loss, samples = [], [], None, 0
    torch.cuda.reset_peak_memory_stats()
    with Telemetry() as telemetry:
        for update in range(warmup+updates):
            torch.cuda.synchronize()
            start, wait = time.perf_counter(), 0.
            optimizer.zero_grad(set_to_none=True)
            for _ in range(accumulation):
                tick = time.perf_counter()
                batch = template if template is not None else to_device(next(iterator))
                wait += time.perf_counter()-tick
                with amp_context(runtime["precision"]):
                    loss, _ = policy(batch)
                if first_loss is None:
                    first_loss = float(loss.detach())
                (loss/accumulation).backward()
            check_gradients(policy)
            optimizer.step()
            torch.cuda.synchronize()
            if update >= warmup:
                times.append(time.perf_counter()-start)
                data_wait.append(wait)
                samples += 352
    elapsed = sum(times)
    if not telemetry.gpu_util or not telemetry.ram:
        raise RuntimeError("GPU/RAM telemetry was unavailable; this trial cannot approve a profile")
    peak = max(torch.cuda.max_memory_reserved(), max(telemetry.gpu_memory, default=0))
    ram = max(telemetry.ram, default=0)
    limit = int(os.environ.get("SLURM_MEM_PER_NODE", "65536"))*1024**2
    rate = samples/elapsed
    return {"runtime": runtime, "variant": variant, "mode": mode, "complete": True,
        "samples_per_second": rate, "update_latency_s": float(np.median(times)),
        "data_wait_fraction": sum(data_wait)/elapsed, "peak_gpu_bytes": peak,
        "peak_ram_bytes": ram, "gpu_utilization_percent": float(np.mean(telemetry.gpu_util)) if telemetry.gpu_util else None,
        "safe": peak <= .85*gpu_identity()["total_memory"] and ram <= .85*limit,
        "first_loss": first_loss, "epoch_seconds": manifest["train_frames"]/rate,
        "training_100_epoch_seconds": 100*manifest["train_frames"]/rate,
        "timing_excludes": ["validation", "checkpoint_io", "queue_time"]}


def profile(root, output, required_precision=None):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, hardware = load_family(root), gpu_identity()
    results = []
    def run(runtime, mode="end_to_end", variant="krvp100"):
        destination = output/f"trial-{len(results):03d}.json"
        command = [sys.executable, str(Path(__file__).resolve().parents[1]/"scripts/dom100.py"),
            "profile-trial", "--family-root", str(root), "--output", str(destination),
            "--runtime", json.dumps(runtime), "--mode", mode, "--variant", variant]
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (output/f"trial-{len(results):03d}.log").write_text(completed.stdout)
        result = json.loads(destination.read_text()) if destination.exists() else {"complete": False, "runtime": runtime, "returncode": completed.returncode}
        results.append(result)
        atomic_json(output/"trials.json", results)
        print(json.dumps({"trial": len(results), "complete": result.get("complete"), "runtime": runtime,
                          "rate": result.get("samples_per_second")}), flush=True)
        return result
    defaults = {"workers": 0, "prefetch": 2, "pin_memory": True, "sampling": "frames", "fused": False}
    precision_candidates = []
    for precision in ("fp32", "bf16"):
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            continue
        candidates = []
        for microbatch in MICROBATCHES:
            result = run({**defaults, "microbatch": microbatch, "precision": precision}, "gpu_only")
            if not result.get("complete"):
                if result.get("oom"):
                    break
                raise RuntimeError("Profiling trial failed; inspect its log")
            if result["safe"]:
                candidates.append(result)
            else:
                break
        if candidates:
            precision_candidates.append(max(candidates, key=lambda r: r["samples_per_second"]))
    if not precision_candidates:
        raise RuntimeError("No safe full-finetuning microbatch")
    # BF16 must first agree with the same-batch FP32 calculation.
    fp32 = next((r for r in results if r.get("complete") and r["mode"]=="gpu_only"
                 and r["runtime"]["precision"]=="fp32" and r["runtime"]["microbatch"]==2), None)
    bf16 = next((r for r in results if r.get("complete") and r["mode"]=="gpu_only"
                 and r["runtime"]["precision"]=="bf16" and r["runtime"]["microbatch"]==2), None)
    if bf16 and fp32 and abs(bf16["first_loss"]-fp32["first_loss"])/max(fp32["first_loss"], 1e-12) > .05:
        precision_candidates = [r for r in precision_candidates if r["runtime"]["precision"]=="fp32"]
    if required_precision:
        precision_candidates = [r for r in precision_candidates if r["runtime"]["precision"]==required_precision]
    if not precision_candidates:
        raise RuntimeError("No safe and numerically verified runtime at the requested precision")
    best = max(precision_candidates, key=lambda r: r["samples_per_second"])["runtime"]
    end_to_end = []
    for workers in (0, 2, 4, 8):
        for prefetch in ((2,) if workers==0 else (2, 4)):
            for sampling in ("frames", "blocks"):
                for pin_memory in (False, True):
                    result = run({**best, "workers": workers, "prefetch": prefetch, "sampling": sampling, "pin_memory": pin_memory})
                    if result.get("complete") and result["safe"]:
                        end_to_end.append(result)
    if not end_to_end:
        raise RuntimeError("No safe real-data runtime")
    measured = {}
    for sampling in ("frames", "blocks"):
        selected = max((r for r in end_to_end if r["runtime"]["sampling"]==sampling), key=lambda r: r["samples_per_second"])
        fused = run({**selected["runtime"], "fused": True})
        if fused.get("complete") and fused["safe"] and fused["samples_per_second"] > selected["samples_per_second"]:
            selected = fused
        for variant in VARIANTS:
            verified = run(selected["runtime"], variant=variant)
            if not verified.get("complete") or not verified["safe"]:
                raise RuntimeError(f"Selected {sampling} runtime is not safe for {variant}")
        measured[sampling] = selected
    selected = max(measured.values(), key=lambda r: r["samples_per_second"])
    frame_rate, block_rate = (measured[s]["samples_per_second"] for s in ("frames", "blocks"))
    artifact = {"version": 1, "hardware": hardware, "gpu_key": gpu_key(), "manifest": manifest["fingerprint"],
        "runtime": selected["runtime"], "block_speedup": block_rate/frame_rate,
        "runtimes": {key: value["runtime"] for key,value in measured.items()},
        "all_variants_verified": True, "selected_measurement": selected}
    artifact["fingerprint"] = digest(artifact)
    atomic_json(root/"profiles"/f"{gpu_key()}.json", artifact)
    return artifact
