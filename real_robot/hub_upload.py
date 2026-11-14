"""Publish validated DOM episodes; existing episode files are append-only."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets

MUTABLE_METADATA={
    'README.md','export_report.json','meta/info.json','meta/episodes.jsonl',
    'meta/episodes_stats.jsonl','meta/sources.jsonl','meta/camera.jsonl','meta/tasks.jsonl',
    'meta/splits.json','meta/train_stats.json','meta/raw_hashes.json','meta/image_preprocessing.json',
}


def authenticated_repo(repo):
    from huggingface_hub import HfApi
    api=HfApi();account=api.whoami()['name']
    repo_id=repo if '/' in repo else f'{account}/{repo}'
    if repo_id.count('/')!=1 or any(not p for p in repo_id.split('/')):raise ValueError('Expected dataset repo owner/name')
    return api,repo_id


def verify_upload_access(api,repo_id,public=False):
    """Authorize a fresh LFS object without uploading bytes or making a commit."""
    from huggingface_hub.errors import HfHubHTTPError,RepositoryNotFoundError
    from huggingface_hub.lfs import post_lfs_batch_info,UploadInfo
    source='HF_TOKEN environment variable' if os.environ.get('HF_TOKEN') else (
        'HUGGING_FACE_HUB_TOKEN environment variable' if os.environ.get('HUGGING_FACE_HUB_TOKEN') else 'cached Hugging Face login')
    try:
        try:info=api.repo_info(repo_id=repo_id,repo_type='dataset')
        except RepositoryNotFoundError:
            api.create_repo(repo_id=repo_id,repo_type='dataset',private=not public)
            info=api.repo_info(repo_id=repo_id,repo_type='dataset')
        if info.private==public:raise ValueError('Repository visibility differs from requested visibility')
        # An existing blob may require no new write. A random digest exercises the
        # same permission check as new videos; returned upload URLs are not used.
        sample=secrets.token_bytes(32)
        _,errors=post_lfs_batch_info(
            [UploadInfo(sha256=hashlib.sha256(sample).digest(),size=len(sample),sample=sample)],
            token=api.token,repo_type='dataset',repo_id=repo_id,
            revision='main',endpoint=api.endpoint)
        if errors:raise PermissionError('Hugging Face refused the new-object upload permission check')
    except HfHubHTTPError as exc:
        if exc.response is None or exc.response.status_code not in (401,403):raise
        raise PermissionError(
            f'Cannot upload new dataset contents to {repo_id} using the {source}. '
            'Use a token with write access to repository contents for this dataset. '
            'HF_TOKEN overrides the cached login: unset HF_TOKEN (and remove an outdated export from .bashrc) '
            'to use a working cached login, or run .venv/bin/python -c "from huggingface_hub import login; login()" with a write-capable token. '
            'No collection has been started by this permission check.') from exc
    return info


def verify_append_base(dataset,repo):
    from huggingface_hub import hf_hub_download
    api,repo_id=authenticated_repo(repo);info=api.repo_info(repo_id=repo_id,repo_type='dataset')
    for name in ('meta/info.json','meta/sources.jsonl'):
        remote=Path(hf_hub_download(repo_id,name,repo_type='dataset',revision=info.sha))
        if remote.read_bytes()!=(Path(dataset)/name).read_bytes():
            raise ValueError('--append-to does not match the current published dataset; use the newest combined export')
    return info.sha


def find_published_base(api,repo_id,info,search_root=None):
    """Find an exact local copy of the pinned published export, or download it."""
    from huggingface_hub import hf_hub_download,snapshot_download
    from huggingface_hub.errors import EntryNotFoundError
    root=Path(search_root) if search_root is not None else Path(__file__).resolve().parent.parent/'scratch/real_robot'
    try:
        remote_info=Path(hf_hub_download(repo_id,'meta/info.json',repo_type='dataset',revision=info.sha,
                                       token=api.token,endpoint=api.endpoint)).read_bytes()
    except EntryNotFoundError:
        return None  # A new repository has no episodes to append to yet.
    remote_sources=Path(hf_hub_download(repo_id,'meta/sources.jsonl',repo_type='dataset',revision=info.sha,
                                      token=api.token,endpoint=api.endpoint)).read_bytes()
    patterns=('collections/*/dom*/meta/info.json','datasets/*/meta/info.json',
              'collections/*/stream-upload/jobs/*/*/dom*/meta/info.json',
              'pilots/*/dom*/meta/info.json','hub_snapshots/*/*/meta/info.json')
    for pattern in patterns:
        for candidate in sorted(root.glob(pattern)):
            sources=candidate.with_name('sources.jsonl')
            if candidate.read_bytes()==remote_info and sources.is_file() and sources.read_bytes()==remote_sources:
                return candidate.parent.parent.resolve()
    target=root/'hub_snapshots'/repo_id.replace('/','--')/info.sha
    print(f'Downloading published append base at {info.sha} before collection starts.',flush=True)
    return Path(snapshot_download(repo_id,repo_type='dataset',revision=info.sha,local_dir=str(target),
                                  token=api.token,endpoint=api.endpoint)).resolve()


def matches_remote(path,remote):
    if path.stat().st_size!=remote.size:return False
    if remote.lfs:
        digest=hashlib.sha256();expected=remote.lfs.sha256
    else:
        digest=hashlib.sha1();digest.update(f'blob {remote.size}\0'.encode());expected=remote.blob_id
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()==expected


def validate_append_prefix(dataset,read_remote):
    """Check existing episode IDs, label metadata and split memberships are retained."""
    local_info=json.loads((dataset/'meta/info.json').read_text());old=json.loads(read_remote('meta/info.json'))
    if local_info['total_episodes']<=old['total_episodes']:raise ValueError('Append requires new episodes beyond the published total')
    for key in ('features','fps','robot_type','data_path','video_path','chunks_size'):
        if local_info[key]!=old[key]:raise ValueError('Append changes dataset schema: '+key)
    for name in ('episodes','episodes_stats','sources','camera','tasks'):
        path=f'meta/{name}.jsonl'
        previous=[json.loads(x) for x in read_remote(path).splitlines() if x.strip()]
        current=[json.loads(x) for x in (dataset/path).read_text().splitlines() if x.strip()]
        if current[:len(previous)]!=previous:raise ValueError('Append changes existing metadata rows: '+path)
    old_splits=json.loads(read_remote('meta/splits.json'));new_splits=json.loads((dataset/'meta/splits.json').read_text())
    for name,ids in old_splits.items():
        if not set(ids)<=set(new_splits.get(name,[])):raise ValueError('Append changes existing episode splits')
    old_hashes=json.loads(read_remote('meta/raw_hashes.json'));new_hashes=json.loads((dataset/'meta/raw_hashes.json').read_text())
    if any(new_hashes.get(k)!=v for k,v in old_hashes.items()):raise ValueError('Append changes existing raw source hashes')
    return old['total_episodes']


def reconcile_append(dataset,repo_id,api,info,public,attempts):
    """Rebuild from the latest published prefix; retain only unchanged duplicates."""
    if attempts>=3:raise RuntimeError('Dataset changed repeatedly during upload; local data is saved. Retry --append when the other upload finishes.')
    from .merge_datasets import merge_datasets
    base=find_published_base(api,repo_id,info)
    if base is None:raise ValueError('Cannot reconcile: the published dataset has no episode metadata')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    output=dataset.parent/f'dom-reconciled-{info.sha[:12]}-{stamp}'
    print(f'Reconciling pending episodes with published revision {info.sha}.',flush=True)
    report=merge_datasets([base,dataset],output,append=True,link_immutable=True)
    previous=json.loads((base/'meta/info.json').read_text())['total_episodes']
    print(f'Reconciled export: {report["episodes"]} episodes, {report["frames"]} frames | {output}',flush=True)
    if report['episodes']>previous:
        return upload_dataset(output,repo_id,public,append=True,expected_parent=info.sha,_reconcile_attempts=attempts+1)
    # A retry after a successful commit must not duplicate the same source data.
    # Confirm the local base still has the actual published immutable contents.
    preserved=0
    for file in api.list_repo_tree(repo_id=repo_id,repo_type='dataset',recursive=True,revision=info.sha):
        if hasattr(file,'blob_id') and file.path.startswith(('data/','videos/','quality/')):
            if not matches_remote(base/file.path,file):raise ValueError('Published base differs from remote episode file: '+file.path)
            preserved+=1
    if api.repo_info(repo_id=repo_id,repo_type='dataset').sha!=info.sha:
        return upload_dataset(output,repo_id,public,append=True,_reconcile_attempts=attempts+1)
    result=dict(repo_id=repo_id,url=f'https://huggingface.co/datasets/{repo_id}',commit=info.sha,public=public,
                mode='already_uploaded',previous_commit=info.sha,previous_episodes=previous,new_episodes=0,
                preserved_episode_files=preserved,dataset_path=str(output),
                validation={k:report[k] for k in ('episodes','frames')},uploaded_utc=datetime.now(timezone.utc).isoformat())
    (dataset.parent/'upload.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def upload_dataset(dataset,repo,public=False,append=False,expected_parent=None,replace_existing=False,_reconcile_attempts=0):
    if replace_existing:raise ValueError('Replacement is disabled. Use --append with an export containing the old episodes followed by the new ones.')
    from huggingface_hub import hf_hub_download,CommitOperationAdd
    from .dom_dataset import validate_dataset
    dataset=Path(dataset).resolve()
    if not (dataset/'meta/info.json').is_file():raise ValueError('Dataset meta/info.json is missing')
    api,repo_id=authenticated_repo(repo)
    info=verify_upload_access(api,repo_id,public)
    report=validate_dataset(dataset)
    if expected_parent is not None and info.sha!=expected_parent and not append:
        raise ValueError('Published dataset changed during recording; reconcile the local export before appending')
    if info.private==public:raise ValueError('Repository visibility differs from requested visibility')
    remote={f.path:f for f in api.list_repo_tree(repo_id=repo_id,repo_type='dataset',recursive=True,revision=info.sha) if hasattr(f,'blob_id')}
    previous_episodes=0;preserved=0
    if 'meta/info.json' in remote:
        if not append:raise ValueError('Dataset already contains episodes; use --append with the cumulative export')
        def read_remote(name):
            return Path(hf_hub_download(repo_id,name,repo_type='dataset',revision=info.sha)).read_text()
        remote_sources=[json.loads(line) for line in read_remote('meta/sources.jsonl').splitlines() if line.strip()]
        local_sources=[json.loads(line) for line in (dataset/'meta/sources.jsonl').read_text().splitlines() if line.strip()]
        if len(local_sources)<=len(remote_sources) or local_sources[:len(remote_sources)]!=remote_sources:
            return reconcile_append(dataset,repo_id,api,info,public,_reconcile_attempts)
        previous_episodes=validate_append_prefix(dataset,read_remote)
        for name,file in remote.items():
            if name.startswith(('data/','videos/','quality/')):
                path=dataset/name
                if not path.is_file() or not matches_remote(path,file):
                    raise ValueError('Append would change or omit an existing episode file: '+name)
                preserved+=1
    card=dataset/'README.md'
    if not card.exists():
        info_local=json.loads((dataset/'meta/info.json').read_text())
        size=info_local['features']['observation.images.opst_cam']['shape']
        card.write_text('---\ntask_categories:\n- robotics\ntags:\n- lerobot\n---\n\n# KRVLA real-robot dataset\n\n'
            f"{report['episodes']} episodes, {report['frames']} frames at {info_local['fps']} FPS. "
            f'RGB video dimensions: {size[1]}x{size[0]}. Human leader/follower dynamic pick-and-place.\n\n'
            'Existing episode data, videos and quality files are retained byte-for-byte when new episodes are appended. '
            'Metadata and aggregate statistics describe the cumulative dataset. Previous revisions remain in Git history.\n\n'
            'DOM-compatible LeRobot v2.1 export. Object pose/velocity are visible-surface estimates; orientation is assumed. '
            'Operator success is not independent placement verification. Inspect quality sidecars and export_report.json. '
            'Use explicit episode IDs in meta/splits.json. Raw RGB-D archives remain local.\n')
    if repo_id=='ailab-Clemson/KRVLA-dataset' and '/tree/training' not in card.read_text():
        card.write_text(card.read_text()+'\n## KRVLA training\n\n'
            'Use the [processed training branch](https://huggingface.co/datasets/ailab-Clemson/KRVLA-dataset/tree/training) '
            'with `revision="training"`. This main branch preserves raw exports. '
            'The collection workflow refreshes the processed branch with depth-corrected labels, quality handling, explicit splits and train-only statistics after each upload.\n')
    additions=[];kept_paths=[]
    for path in sorted(dataset.rglob('*')):
        if not path.is_file():continue
        name=str(path.relative_to(dataset))
        if name=='.gitattributes':continue
        if name in remote:
            if name.startswith(('data/','videos/','quality/')):
                kept_paths.append(name);continue
            if matches_remote(path,remote[name]):continue
            if name not in MUTABLE_METADATA:raise ValueError('Append would overwrite a non-metadata file: '+name)
        additions.append(CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(path)))
    if not additions:raise ValueError('No additions to upload')
    from huggingface_hub.errors import HfHubHTTPError
    try:
        commit=api.create_commit(repo_id=repo_id,repo_type='dataset',revision='main',parent_commit=info.sha,
            operations=additions,commit_message=f'Append {report["episodes"]-previous_episodes} episodes ({report["episodes"]} total)')
    except HfHubHTTPError as exc:
        if not append or exc.response is None or exc.response.status_code not in (409,412):raise
        if _reconcile_attempts>=3:raise RuntimeError('Dataset changed repeatedly during upload; retry --append after the other uploader finishes.') from exc
        print('Another upload committed first; refreshing the published base and retrying append.',flush=True)
        return upload_dataset(dataset,repo_id,public,append=True,_reconcile_attempts=_reconcile_attempts+1)
    actual=json.loads(Path(hf_hub_download(repo_id,'meta/info.json',repo_type='dataset',revision=commit.oid)).read_text())
    if actual['total_episodes']!=report['episodes'] or actual['total_frames']!=report['frames']:
        raise RuntimeError('Remote dataset totals differ after append')
    after={f.path:f for f in api.list_repo_tree(repo_id=repo_id,repo_type='dataset',recursive=True,revision=commit.oid) if hasattr(f,'blob_id')}
    if not set(remote)<=set(after):raise RuntimeError('Existing repository files disappeared during append')
    if any(after[name].blob_id!=remote[name].blob_id for name in kept_paths):raise RuntimeError('Existing episode file changed during append')
    result=dict(repo_id=repo_id,url=f'https://huggingface.co/datasets/{repo_id}',commit=commit.oid,public=public,
        mode='append' if previous_episodes else 'initial_upload',previous_commit=info.sha,
        previous_episodes=previous_episodes,new_episodes=report['episodes']-previous_episodes,
        preserved_episode_files=preserved,dataset_path=str(dataset),validation=report,uploaded_utc=datetime.now(timezone.utc).isoformat())
    (dataset.parent/'upload.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path);p.add_argument('--repo',required=True)
    p.add_argument('--check-access',action='store_true',help='Check new-video upload authorization without uploading data or creating a commit')
    p.add_argument('--public',action='store_true');p.add_argument('--append',action='store_true',help='Append a block or cumulative export; reconcile stale bases and skip verified unchanged source episodes already published')
    a=p.parse_args(argv)
    if a.check_access:
        api,repo_id=authenticated_repo(a.repo)
        verify_upload_access(api,repo_id,a.public)
        print(f'Upload permission confirmed: {repo_id}')
        return
    if a.dataset is None:p.error('--dataset is required unless --check-access is used')
    result=upload_dataset(a.dataset,a.repo,a.public,append=a.append)
    print(json.dumps(result,indent=2))
    if result['repo_id']=='ailab-Clemson/KRVLA-dataset':
        from .training_release import sync
        sync(result['repo_id'],result['dataset_path'],result['commit'])


if __name__=='__main__':main()
