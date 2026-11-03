"""Atomic, generation-based checkpoints and single-writer run ownership."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch

from utils.dom_experiment import atomic_json


@contextmanager
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {root}") from exc
        yield


def rng_state(device=None):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if device is None else [torch.cuda.get_rng_state(device)]) if torch.cuda.is_available() else None,
            "cuda_device_only": device is not None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        if state.get("cuda_device_only"):
            torch.cuda.set_rng_state(state["cuda"][0].cpu(), torch.cuda.current_device())
        else:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_generation(root, policy, optimizer, scheduler, progress, fingerprint,
                    wandb_id, best=None, milestone=False, distributed_state=None, final_checkpoint=False):
    root = Path(root)
    generations = root/"generations"
    generations.mkdir(exist_ok=True, parents=True)
    name = f"u{progress['updates']:09d}-{uuid.uuid4().hex[:8]}"
    temp, final = generations/(".tmp-"+name), generations/name
    temp.mkdir()
    policy.save_pretrained(str(temp))
    config_path = temp/"config.json"
    config = json.loads(config_path.read_text())
    config.update(epoch=progress["epoch"], completed_epochs=progress["epoch"],
                  delta_timestamps={"observation": [-2, 0]}, use_delta_action=True,
                  experiment_fingerprint=fingerprint, wandb_id=wandb_id,
                  completion_reason=progress.get("completion_reason"),
                  early_stopping=progress.get("early_stopping"))
    atomic_json(config_path, config)
    provenance_path = root/"provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {
        "fingerprint": fingerprint, "wandb_id": wandb_id}
    atomic_json(temp/"provenance.json", provenance)
    torch.save({"version": 1, "fingerprint": fingerprint, "progress": progress,
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng": distributed_state["rng_by_rank"][0] if distributed_state is not None else rng_state(),
        "distributed": distributed_state, "wandb_id": wandb_id, "best": best,
        "provenance": provenance}, temp/"training_state.pt")
    # Flush complete files before publishing the pointer. A killed writer can
    # leave an orphan temp directory, never mixed weights/optimizer state.
    for path in temp.iterdir():
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
    temp.rename(final)
    pointer = {"generation": name, "progress": progress, "fingerprint": fingerprint}
    atomic_json(root/"latest.json", pointer)
    if final_checkpoint:
        atomic_json(root/"final.json", pointer)
    if milestone:
        atomic_json(root/f"epoch-{progress['epoch']:03d}.json", pointer)
    if best is not None and best.get("updates") == progress["updates"]:
        atomic_json(root/"best.json", pointer)
    keep = {json.loads(p.read_text())["generation"] for p in root.glob("*.json")
            if p.name in ("latest.json", "best.json", "final.json") or p.name.startswith("epoch-")}
    for path in generations.iterdir():
        if path.is_dir() and not path.name.startswith(".tmp-") and path.name not in keep:
            shutil.rmtree(path)
    return final


def resolve_generation(root, pointer="latest.json"):
    root = Path(root)
    manifest = json.loads((root/pointer).read_text())
    path = root/"generations"/manifest["generation"]
    for required in ("config.json", "model.safetensors", "training_state.pt"):
        if not (path/required).is_file():
            raise FileNotFoundError(path/required)
    return path


def load_generation(root, policy, optimizer, scheduler, fingerprint, restore_random=True):
    import safetensors.torch
    path = resolve_generation(root)
    state = torch.load(path/"training_state.pt", map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError("Resume fingerprint differs from frozen experiment")
    safetensors.torch.load_model(policy, str(path/"model.safetensors"), strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if restore_random:
        restore_rng(state["rng"])
    return state
