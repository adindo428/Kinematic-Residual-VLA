import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
deployment = json.loads((HERE/'deployment.json').read_text())
palmetto = ['ssh', '-S', '/tmp/krvla-palmetto.sock', '-o', 'ControlMaster=no', '-o', 'BatchMode=yes', 'aparame@slogin.palmetto.clemson.edu']
delta = ['ssh', '-S', '/tmp/deltaai-kinematic.sock', '-o', 'ControlMaster=no', '-o', 'BatchMode=yes', 'aparam eshwaran@gh-login01.delta.ncsa.illinois.edu'.replace(' ', '')]
VERIFY_SOURCE = r'''
import hashlib, json, pathlib, sys
request = json.load(sys.stdin)
errors = []; checked = 0
for label, manifest in request.items():
    root = pathlib.Path(manifest['source'])
    for name, info in manifest['files'].items():
        path = root/name
        if label == 'tracking' and name.startswith('configs/tracking_dom75_full20_') and not path.exists():
            path = pathlib.Path('/scratch/aparame/Dynamic_VLA/runs/tracking_dom75_full20/provenance')/path.name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            errors.append(str(path))
        checked += 1
print(json.dumps({'checked_source_files': checked, 'changed_or_missing': errors}))
sys.exit(1 if errors else 0)
'''
sources = {label: json.loads((HERE/'staging'/label/'PALMETTO_SOURCE_MANIFEST.json').read_text()) for label in ('tracking','dom100')}
source = subprocess.run(palmetto + ['python3 -c ' + shlex.quote(VERIFY_SOURCE)], input=json.dumps(sources).encode(), capture_output=True)
if source.returncode:
    raise RuntimeError('Palmetto changed since capture: ' + source.stdout.decode() + source.stderr.decode())
print(source.stdout.decode().strip(), flush=True)
DEPLOY = r'''
import datetime, hashlib, io, json, os, pathlib, sys, tarfile
destination = pathlib.Path(sys.argv[1])
expected_archive = sys.argv[2]
payload = sys.stdin.buffer.read()
if hashlib.sha256(payload).hexdigest() != expected_archive: raise RuntimeError('Archive checksum differs')
destination.parent.mkdir(parents=True, exist_ok=True)
if destination.exists(): raise RuntimeError('Refusing to overwrite existing snapshot')
with tarfile.open(fileobj=io.BytesIO(payload), mode='r:gz') as archive:
    members = archive.getmembers()
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != destination.name or not (member.isfile() or member.isdir()):
            raise RuntimeError('Unsafe archive member: ' + member.name)
    archive.extractall(destination.parent)
os.chmod(destination, 0o700)
manifest = json.loads((destination/'TRANSFER_MANIFEST.json').read_text())
checked_bytes = 0
for name, info in manifest['files'].items():
    path = destination/name
    data = path.read_bytes()
    if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
        raise RuntimeError('Deployed file differs: '+name)
    checked_bytes += len(data)
actual = {str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()}
if actual != set(manifest['files']) | {'TRANSFER_MANIFEST.json'}: raise RuntimeError('Unexpected files in deployment')
if any(p.name in {'simulations', 'assets', 'data', 'videos', 'checkpoints', '.git', '__pycache__'} for p in destination.rglob('*')):
    raise RuntimeError('Excluded directory present')
if any(p.suffix in {'.safetensors','.pt','.mp4','.parquet','.usd','.usda','.usdc','.sif'} for p in destination.rglob('*')):
    raise RuntimeError('Excluded payload present')
report = {'destination': str(destination), 'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'verified_files': len(actual), 'source_bytes': checked_bytes, 'archive_sha256': expected_archive, 'simulator_payload_present': False, 'dataset_or_weight_payload_present': False, 'training_submitted': False}
latest = destination.parent/'LATEST.json'
temporary = latest.with_suffix('.json.tmp')
temporary.write_text(json.dumps(report, indent=2)+'\n')
temporary.replace(latest)
print(json.dumps(report))
'''
payload = Path(deployment['archive']).read_bytes()
if hashlib.sha256(payload).hexdigest() != deployment['archive_sha256']:
    raise RuntimeError('Local archive changed')
cmd = 'python3 -c ' + shlex.quote(DEPLOY) + ' ' + shlex.quote(deployment['destination']) + ' ' + deployment['archive_sha256']
result = subprocess.run(delta + [cmd], input=payload, capture_output=True)
if result.returncode: raise RuntimeError(result.stdout.decode()+result.stderr.decode())
report = json.loads(result.stdout)
report['palmetto_source_verification'] = json.loads(source.stdout)
(HERE/'DEPLOYMENT_REPORT.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
