"""Explicit, private-by-default publication with revision-pinned verification."""
import json
from pathlib import Path

from .config import file_hash
from .storage import atomic_json
from .validate import validate_dataset


def upload_dataset(root, repo_id, validation_report, public=False):
    from huggingface_hub import HfApi, hf_hub_download

    root = Path(root).resolve()
    if not repo_id or len(repo_id.split("/")) != 2:
        raise ValueError("Supply --repo-id namespace/dataset")
    report = json.loads(Path(validation_report).read_text())
    local = validate_dataset(root)
    if local["smoke_only"]:
        raise ValueError("Smoke datasets cannot be published by this command")
    if not report.get("valid") or not report.get("loader_checked") or report.get("checksums_sha256") != local["checksums_sha256"]:
        raise ValueError("A successful --loader-check report for this exact dataset is required")
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=not public, exist_ok=True)
    remote = api.repo_info(repo_id, repo_type="dataset")
    if remote.private != (not public):
        raise ValueError("Existing repository visibility differs from the requested setting; choose a matching repository or explicit visibility flag")
    existing = set(api.list_repo_files(repo_id, repo_type="dataset"))
    local_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".cache" not in p.parts}
    unexpected = existing-local_files-{".gitattributes"}
    if unexpected:
        raise ValueError(f"Remote contains unrelated/stale files; use a dedicated repository: {sorted(unexpected)[:10]}")
    commit = api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(root),
                               ignore_patterns=[".cache/**"], commit_message="Publish validated Franka tracking collection")
    revision = commit.oid
    expected = json.loads((root/"checksums.json").read_text())
    remote_info = api.repo_info(repo_id, repo_type="dataset", revision=revision, files_metadata=True)
    remote_files = {f.rfilename: f for f in remote_info.siblings}
    for name in [*expected, "checksums.json"]:
        if name not in remote_files or remote_files[name].size != (root/name).stat().st_size:
            raise ValueError(f"Remote file missing or wrong size at {revision}: {name}")
    # Download an actual episode (all cameras) and its metadata at the immutable revision.
    sample = ["checksums.json", "meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl",
              "data/chunk-000/episode_000000.parquet"]
    sample += [f"videos/chunk-000/observation.images.{name}/episode_000000.mp4" for name in ("wrist_cam", "opst_cam", "side_cam")]
    for name in sample:
        downloaded = hf_hub_download(repo_id, name, repo_type="dataset", revision=revision, force_download=True)
        if file_hash(downloaded) != file_hash(root/name):
            raise ValueError(f"Downloaded sample checksum mismatch: {name}")
        if name.endswith(".mp4"):
            from .storage import video_summary
            video_summary(downloaded)
    receipt = dict(repo_id=repo_id, revision=revision, url=f"https://huggingface.co/datasets/{repo_id}/tree/{revision}",
                   checked_remote_files=len(expected)+1, checked_downloads=sample,
                   checksums_sha256=local["checksums_sha256"])
    atomic_json(root.parent/(root.name+".upload.json"), receipt)
    return receipt
