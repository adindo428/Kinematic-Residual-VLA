#!/usr/bin/env python3
"""Upload or restore complete training checkpoints from a private Hub model repo."""

import argparse
import json
import os
import shutil
from pathlib import Path


REQUIRED_FILES = ("config.json", "model.safetensors", "training_state.pt")


def checkpoint_epoch(path: Path) -> int:
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete checkpoint at {path}: {', '.join(missing)}")
    epoch = json.loads((path / "config.json").read_text())["epoch"]
    if not isinstance(epoch, int) or epoch < 1:
        raise ValueError(f"Invalid checkpoint epoch in {path / 'config.json'}: {epoch!r}")
    if (path / "model.safetensors").stat().st_size == 0 or (path / "training_state.pt").stat().st_size == 0:
        raise ValueError(f"Empty model or training state in {path}")
    return epoch


def upload(repo_id: str, variant: str, path: Path) -> None:
    from huggingface_hub import HfApi

    epoch = checkpoint_epoch(path)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(path),
        path_in_repo=f"{variant}/epoch-{epoch:04d}",
        allow_patterns=list(REQUIRED_FILES),
        commit_message=f"{variant}: epoch {epoch}",
    )
    api.upload_file(
        repo_id=repo_id,
        repo_type="model",
        path_or_fileobj=json.dumps({"epoch": epoch}).encode(),
        path_in_repo=f"{variant}/latest.json",
        commit_message=f"{variant}: latest epoch {epoch}",
    )
    print(f"Uploaded {variant} epoch {epoch} to {repo_id}")


def restore(repo_id: str, variant: str, path: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception as error:
        from huggingface_hub.utils import RepositoryNotFoundError

        if isinstance(error, RepositoryNotFoundError):
            raise RuntimeError(
                f"Cannot access Hub repository {repo_id}; create a private model repo first "
                "and verify the repo ID and token"
            ) from error
        raise
    try:
        latest = hf_hub_download(repo_id=repo_id, filename=f"{variant}/latest.json", repo_type="model")
    except Exception as error:
        from huggingface_hub.utils import EntryNotFoundError

        if isinstance(error, EntryNotFoundError):
            if all((path / name).is_file() for name in REQUIRED_FILES):
                checkpoint_epoch(path)
                print(f"No Hub {variant} checkpoint; keeping local checkpoint")
                return
            print(f"No Hub {variant} checkpoint; starting from pretrained weights")
            return
        raise
    remote_epoch = json.loads(Path(latest).read_text())["epoch"]
    if not isinstance(remote_epoch, int) or remote_epoch < 1:
        raise ValueError(f"Invalid remote epoch: {remote_epoch!r}")
    if all((path / name).is_file() for name in REQUIRED_FILES):
        try:
            if checkpoint_epoch(path) >= remote_epoch:
                print(f"Keeping local {variant} epoch {checkpoint_epoch(path)}")
                return
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    prefix = f"{variant}/epoch-{remote_epoch:04d}"
    snapshot = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            allow_patterns=[f"{prefix}/{name}" for name in REQUIRED_FILES],
            token=os.environ.get("HF_TOKEN"),
        )
    )
    source = snapshot / prefix
    checkpoint_epoch(source)
    remote_config = json.loads((source / "config.json").read_text())
    if remote_config["epoch"] != remote_epoch:
        raise ValueError(f"Hub latest marker and checkpoint disagree for {variant}")
    if any((path / name).exists() for name in REQUIRED_FILES):
        raise RuntimeError(
            f"Local checkpoint {path} is older or incomplete; refusing to overwrite it. "
            "Move it aside or use a fresh output directory before restoring."
        )
    path.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        temporary = path / f"{name}.download"
        shutil.copyfile(source / name, temporary)
        os.replace(temporary, path / name)
    print(f"Restored {variant} epoch {remote_epoch} from {repo_id} to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("upload", "restore"))
    parser.add_argument("--repo", required=True, help="Private model repository, e.g. username/dom-vla-training")
    parser.add_argument("--variant", required=True, choices=("baseline", "residual"))
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "upload":
        upload(args.repo, args.variant, args.checkpoint)
    else:
        restore(args.repo, args.variant, args.checkpoint)


if __name__ == "__main__":
    main()
