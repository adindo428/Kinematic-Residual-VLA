"""Portable, checksum-locked preparation and launch for real-data family training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
FROZEN = Path(os.environ.get('POSTTRAIN_FROZEN', PROJECT / 'post_training/frozen'))
sys.path.insert(0, str(PROJECT / 'workstation'))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(root, files):
    for name, expected in files.items():
        path = root / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError('Missing or changed locked file: ' + str(path))


def verify_protocol():
    if not (FROZEN / 'protocol-lock.json').is_file():
        raise ValueError('Private protocol bundle missing. Run post_training/export_local.py on the source computer and transfer its output; see README.')
    lock = read(FROZEN / 'protocol-lock.json')
    for name, expected in lock['files'].items():
        path = FROZEN / name.removeprefix('post_training/frozen/') if name.startswith('post_training/frozen/') else PROJECT / name
        if not path.is_file() or sha(path) != expected:raise ValueError('Changed locked file: ' + str(path))
    return sha(FROZEN / 'protocol-lock.json')


def prepare(args):
    lock = verify_protocol()
    root = args.root.resolve()
    if (root / 'campaign.json').exists():
        raise ValueError('Campaign already exists; use a fresh root rather than overwriting a run')
    source = read(FROZEN / 'campaign.v6.json')
    campaign = copy.deepcopy(source)
    names = set(args.families.split(',')) if args.families else {e['family'] for e in source['entries']}
    if not names or names - {e['family'] for e in source['entries']}:
        raise ValueError('Unknown or empty family selection')
    campaign['entries'] = [e for e in campaign['entries'] if e['family'] in names]
    # Adding a language input changes the training protocol, never the layers.
    from velocity_conditioning import VELOCITY_PROMPT
    for entry in campaign['entries']:
        if args.conditioning == 'velocity':
            entry['velocity_conditioning'] = VELOCITY_PROMPT
    root.mkdir(parents=True, exist_ok=True)
    dataset = root / 'dataset'
    dataset.mkdir(exist_ok=True)
    manifest = read(FROZEN / 'dataset-manifest.json')
    archive = FROZEN / 'dataset-labels.tar.gz'
    if sha(archive) != manifest['label_archive_sha256']:
        raise ValueError('Label archive changed')
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            # The release contains ordinary files only. Never follow an archive link.
            if not member.isfile() or member.name not in manifest['labels']:
                raise ValueError('Unexpected archive member: ' + member.name)
            target = dataset / member.name
            if not target.resolve().is_relative_to(dataset.resolve()):
                raise ValueError('Unsafe archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source_file, target.open('wb') as dest:
                shutil.copyfileobj(source_file, dest)
    verify_files(dataset, manifest['labels'])
    for name in ['real_statistics.json', 'sampling_weights.npy', 'dependency-patch-lock.json']:
        shutil.copyfile(FROZEN / name, root / name)
    recipe = campaign['training_recipe']
    recipe['local_verified_dataset'] = str(dataset)
    recipe['sampling_weights'] = str(root / 'sampling_weights.npy')
    recipe['final_evaluation_episodes'] = read(FROZEN / 'validation-10-summary.json')['episodes']
    for field in ['current_queue_dependency', 'throughput_revision', 'benchmark_manifest', 'family_performance']:
        campaign.pop(field, None)
    campaign['prior_campaign_completion'] = read(FROZEN / 'prior-campaign-completion.json')
    campaign['run_name_prefix'] = 'real-v7-velocity-' if args.conditioning == 'velocity' else 'real-v6-migrated-'
    campaign['dataset_release'] = dict(video_repo=manifest['repo_id'],video_revision=manifest['revision'],
        local_label_archive_sha256=manifest['label_archive_sha256'])
    campaign['migration'] = dict(protocol_sha256=lock, conditioning=args.conditioning,
        source_campaign_sha256=sha(FROZEN / 'campaign.v6.json'),
        metrics_protocol='Identical physical metrics, anchor indices, masks, targets and seeds to v6',
        old_benchmarks='Reference only; benchmark all assigned checkpoints on destination before training')
    campaign['benchmark_plan']['required_families'] = [e['family'] for e in campaign['entries']]
    campaign['benchmark_plan']['report_directory'] = str(root / 'benchmarks')
    campaign['source_hashes'] = {name: expected for name, expected in read(FROZEN / 'protocol-lock.json')['files'].items()
                                 if not name.startswith('post_training/frozen/')}
    campaign['source_hashes'].update({str((root / name).relative_to(PROJECT)):sha(root / name)
                                    for name in ['real_statistics.json','dependency-patch-lock.json']
                                    if (root / name).is_relative_to(PROJECT)})
    write(root / 'campaign.json', campaign)
    print('Prepared', root, 'families:', ', '.join(sorted(names)), 'conditioning:', args.conditioning)
    print('Next: download, verify, then run. Existing v6 weights are not uploaded to GitHub.')


def patch_transformers():
    import transformers
    if transformers.__version__ != '4.53.3':
        raise ValueError('Expected Transformers 4.53.3')
    installed = Path(transformers.__file__).parent.parent
    for name, expected in read(PROJECT / 'post_training/compat/dependency-patch-lock.json')['files'].items():
        relative = name.removeprefix('transformers/')
        source = PROJECT / 'post_training/compat/transformers-replacement' / relative
        data = source.read_bytes()
        if relative == 'models/siglip/check.py':
            data = data.replace(b'4.53.2', b'4.53.3')
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Replacement hash mismatch: ' + name)
        (installed / name).write_bytes(data)
    print('Applied and SHA-verified pinned OpenPI compatibility files')


def download(args):
    verify_protocol()
    from huggingface_hub import snapshot_download
    root = args.root.resolve(); campaign = read(root / 'campaign.json')
    # All native source/metadata/tokenizers are pinned; only assigned large weights download.
    patterns = ['model-index.json','files.json','SHA256SUMS','environment-*.json',
                'requirements-*.txt','tools/**','licenses/**','assets/**','code/**']
    patterns += [e['checkpoint_file'] for e in campaign['entries']]
    snapshot_download(campaign['repo'], revision=campaign['revision'], local_dir=root / 'hub',
                      allow_patterns=patterns, max_workers=4)
    reference = read(FROZEN / 'hub-files.json')['files']
    downloaded = {name: row['sha256'] for name, row in reference.items()
                  if (root / 'hub' / name).is_file() and (name.startswith(('code/','assets/','tools/'))
                  or name in ['model-index.json','environment-adapters.json'])}
    verify_files(root / 'hub', downloaded)
    downloads = dict(state='completed',revision=campaign['revision'],models={})
    for entry in campaign['entries']:
        path = root / 'hub' / entry['checkpoint_file']
        if sha(path) != entry['sha256']:
            raise ValueError('Initial checkpoint hash mismatch: ' + entry['family'])
        downloads['models'][entry['family']] = dict(state='verified',path=str(path),sha256=entry['sha256'])
    write(root / 'downloads.json', downloads)
    manifest = read(FROZEN / 'dataset-manifest.json')
    snapshot_download(manifest['repo_id'],repo_type='dataset',revision=manifest['revision'],
                      local_dir=root / 'dataset',allow_patterns=list(manifest['videos']),max_workers=4)
    verify_files(root / 'dataset', manifest['videos'])
    if root.is_relative_to(PROJECT):
        campaign['source_hashes'].update({str((root / 'hub' / name).relative_to(PROJECT)):expected
                                          for name, expected in downloaded.items()})
    else:
        # The controller's source lock is repository-relative; the full hub lock
        # is independently verified by this utility before launch.
        campaign['migration']['external_runtime_root'] = True
    write(root / 'campaign.json', campaign)
    verify(args)


def verify(args):
    lock = verify_protocol()
    root = args.root.resolve(); campaign = read(root / 'campaign.json')
    if campaign['migration']['protocol_sha256'] != lock:
        raise ValueError('Campaign uses a different protocol lock')
    manifest = read(FROZEN / 'dataset-manifest.json')
    verify_files(root / 'dataset', {**manifest['labels'], **manifest['videos']})
    for name in ['real_statistics.json','sampling_weights.npy','dependency-patch-lock.json']:
        if sha(root / name) != sha(FROZEN / name):raise ValueError('Changed frozen statistics/weights: ' + name)
    original = read(FROZEN / 'campaign.v6.json')['training_recipe']
    for field in ['action_objective','action_validation_indices','augmentation','learning_rate','weight_decay',
                  'max_epochs','early_stopping_patience','dataset_quality_policy_sha256','sampling_weights_sha256']:
        if campaign['training_recipe'][field] != original[field]:raise ValueError('Changed training/metric field: ' + field)
    if campaign['training_recipe']['final_evaluation_episodes'] != read(FROZEN / 'validation-10-summary.json')['episodes']:
        raise ValueError('Final evaluation episodes changed')
    from velocity_conditioning import VELOCITY_PROMPT
    for entry in campaign['entries']:
        expected = VELOCITY_PROMPT if campaign['migration']['conditioning']=='velocity' else None
        if entry.get('velocity_conditioning') != expected:raise ValueError('Mixed conditioning contracts')
        if sha(root / 'hub' / entry['checkpoint_file']) != entry['sha256']:raise ValueError('Changed parent checkpoint')
    reference = read(FROZEN / 'hub-files.json')['files']
    verify_files(root / 'hub', {name:row['sha256'] for name,row in reference.items()
        if name.startswith(('code/','assets/','tools/')) and (root / 'hub' / name).is_file()})
    import transformers
    installed = Path(transformers.__file__).parent.parent
    verify_files(installed, read(FROZEN / 'dependency-patch-lock.json')['files'])
    import importlib.metadata as metadata
    runtime = {name:metadata.version(name) for name in campaign['runtime_versions']}
    for name, expected in campaign['runtime_versions'].items():
        actual = runtime[name]
        # CUDA 12.8 is allowed for GPUs needing that wheel; record the actual build.
        if (actual.split('+')[0] if name=='torch' else actual) != (expected.split('+')[0] if name=='torch' else expected):
            raise ValueError(f'Runtime mismatch: {name}: {actual} != {expected}')
    from family_data import RealFamilyDataset
    for split, count in [('train',65689),('validation',7710)]:
        if len(RealFamilyDataset(campaign,campaign['entries'][0]['family'],split)) != count:
            raise ValueError('Anchor population changed')
    write(root / 'migration-verification.json',dict(passed=True,protocol_sha256=lock,runtime=runtime,
        labels=len(manifest['labels']),videos=len(manifest['videos']),train_anchors=65689,validation_anchors=7710))
    print('Verified frozen labels/videos, source, statistics, splits, metrics, runtime and conditioning')


def run(args):
    verify(args)
    root=args.root.resolve()
    env=dict(os.environ,POSTTRAIN_ROOT=str(root),POSTTRAIN_PYTHON=sys.executable)
    subprocess.run([str(PROJECT/'workstation/family-python.sh'),
        str(PROJECT/'workstation/run_benchmarked_family_campaign.py'),'--campaign',str(root/'campaign.json')],
        env=env,cwd=PROJECT,check=True)
    if read(root/'queue.json').get('state')!='completed':
        raise RuntimeError('Campaign did not complete; inspect queue.json and per-family logs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','patch-transformers','download','verify','run'])
    parser.add_argument('--root',type=Path,default=PROJECT/'scratch/real_robot/family-finetuning-v7')
    parser.add_argument('--families',help='Comma-separated job ownership; default all six')
    parser.add_argument('--conditioning',choices=['velocity','native'],default='velocity')
    args=parser.parse_args()
    if args.command=='patch-transformers':patch_transformers()
    else:globals()[args.command](args)


if __name__=='__main__':main()
