import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
SSH = ['ssh', '-S', '/tmp/krvla-palmetto.sock', '-o', 'ControlMaster=no', '-o', 'BatchMode=yes', 'aparame@slogin.palmetto.clemson.edu']
REMOTE = r'''
import hashlib, io, json, pathlib, subprocess, sys, tarfile
root = pathlib.Path(sys.argv[1])
dirs = {'core', 'policies', 'proactive_vla', 'configs'}
utils = {'average_meter.py', 'datasets.py', 'distributed.py', 'dom_data.py', 'dom_experiment.py', 'helpers.py', 'instruction_generator.py', 'memcached.py', 'summary_writer.py', '__init__.py'}
scripts = {'dom100.py', 'prepare_dom100.py', 'check_train_readiness.py', 'train_tracking_velocity.py', 'rebuild_dom_speed_split.py', 'train_full_finetune_speed_ood_pair.sh', 'train_paired_velocity_pair.sh', 'palmetto_env.sh', 'deltaai_env.sh', 'dom100_env.sh', 'run_dom100.sh', 'hub_checkpoints.py', 'fix_lerobot_videos.py', 'dom100_submit.py', 'submit_dom100.sh'}
top = {'run.py', 'requirements.txt', 'requirements_rocky8.txt', 'LICENSE', 'DOM100.md', 'DELTA_AI.md', 'RECOVERY.md'}
files = {}
excluded = []
for p in sorted(root.rglob('*')):
    if not p.is_file() or p.is_symlink(): continue
    rel = p.relative_to(root)
    if any(part.startswith('.') or part == '__pycache__' for part in rel.parts): continue
    take = (len(rel.parts) == 1 and rel.name in top)
    take |= rel.parts[0] in dirs and p.suffix in {'.py', '.yaml', '.yml', '.json', '.txt'}
    take |= rel.parts[0] == 'utils' and rel.name in utils and len(rel.parts) == 2
    take |= rel.parts[0] == 'scripts' and rel.name in scripts and len(rel.parts) == 2
    take |= rel.parts[0] == 'slurm' and len(rel.parts) == 2 and (rel.name.startswith(('train_', 'resume_tracking_', '01_dom', '02_kr', '03_domv', '04_krv', '05_krp', '06_krvp', 'prepare_dom', 'profile_dom', 'pilots_dom', 'smoke_dom', 'deltaai_')) and rel.suffix == '.sbatch')
    if take:
        data = p.read_bytes()
        if len(data) > 20_000_000: raise RuntimeError('Unexpected large code file: ' + str(p))
        files[str(rel)] = (data, p.stat().st_mode & 0o777)
    else: excluded.append(str(rel))
if sys.argv[2] == 'tracking':
    base = pathlib.Path('/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/provenance')
    for p in sorted(base.glob('*.yaml')):
        files['configs/' + p.name] = (p.read_bytes(), 0o644)
record = {'source': str(root), 'git_head': subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(), 'files': {name: {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for name,(data,mode) in files.items()}, 'excluded_files': excluded}
files['PALMETTO_SOURCE_MANIFEST.json'] = (json.dumps(record, indent=2).encode() + b'\n', 0o644)
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for name, (data, mode) in files.items():
        info = tarfile.TarInfo(name); info.size = len(data); info.mode = mode
        archive.addfile(info, io.BytesIO(data))
'''
for label, root in {
    'tracking': '/home/aparame/Research/Dynamic_VLA/third_party/Kinematic-Residual-VLA/third_party/DynamicVLA',
    'dom100': '/scratch/aparame/Dynamic_VLA/third_party/dom100-20261007',
}.items():
    archive_path = HERE / f'{label}-palmetto.tar.gz'
    with archive_path.open('wb') as output:
        result = subprocess.run(SSH + [f'python3 - {root} {label}'], input=REMOTE.encode(), stdout=output, stderr=subprocess.PIPE)
    if result.returncode: raise RuntimeError(result.stderr.decode())
    destination = HERE / 'staging' / label
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.name.startswith('/') or '..' in Path(member.name).parts or not member.isfile():
                raise RuntimeError('Unsafe archive entry')
        archive.extractall(destination)
    manifest = json.loads((destination / 'PALMETTO_SOURCE_MANIFEST.json').read_text())
    for name, info in manifest['files'].items():
        if hashlib.sha256((destination / name).read_bytes()).hexdigest() != info['sha256']:
            raise RuntimeError('Hash mismatch: ' + name)
    print(json.dumps({'source': root, 'files': len(manifest['files']), 'archive_bytes': archive_path.stat().st_size, 'label': label}))
