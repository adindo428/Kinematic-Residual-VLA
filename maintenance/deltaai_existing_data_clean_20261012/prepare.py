import base64,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
PAL=['ssh','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=15','aparame@slogin.palmetto.clemson.edu']
DELTA=['ssh','-S','/tmp/deltaai-kinematic.sock','-o','ControlMaster=no','-o','BatchMode=yes','-o','ConnectTimeout=15','aparam eshwaran@gh-login01.delta.ncsa.illinois.edu'.replace(' ','')]
remote="""import base64,json
from pathlib import Path
p=Path('/scratch/aparame/Dynamic_VLA/dataset-audit-20261012/final-receipt')
m=json.loads(Path('/scratch/aparame/Dynamic_VLA/dom100-audited-20261012/manifest.json').read_text())
f={x.name:base64.b64encode(x.read_bytes()).decode() for x in p.iterdir() if x.is_file()}
f['palmetto-final-manifest.json']=base64.b64encode(json.dumps(m).encode()).decode()
print(json.dumps(f))
"""
r=subprocess.run(PAL+['python3 -'],input=remote.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
files=json.loads(r.stdout)
(HERE/'audit-receipts').mkdir(exist_ok=True)
for name,data in files.items():(HERE/'audit-receipts'/name).write_bytes(base64.b64decode(data))
for name in ['freeze_family.py','verify_existing.py','verify.sbatch','coordinate.py','profile.sbatch','production.sbatch']:
 files[name]=base64.b64encode((HERE/name).read_bytes()).decode()
# Preserve measured single GPU report from the already deployed six-exclusion setup.
root='/work/nvme/bivv/aparameshwaran/dynamicvla/setup-existing-data-domv-20261012'
remote="""import base64,json
from pathlib import Path
r=Path(%r);r.mkdir(exist_ok=True)
f=json.loads(%r)
for n,b in f.items():
 p=r/n if n in %r else r/'audit-receipts'/n
 p.parent.mkdir(exist_ok=True);p.write_bytes(base64.b64decode(b))
s=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-video-clean-domv-20261012/bounded-single-gpu.json')
(r/'bounded-single-gpu.json').write_bytes(s.read_bytes())
print('Small audit receipts and workflow uploaded; no dataset transfer.')
"""%(root,json.dumps(files),['freeze_family.py','verify_existing.py','verify.sbatch','coordinate.py','profile.sbatch','production.sbatch'])
r=subprocess.run(DELTA+['python3 -'],input=remote.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
print(r.stdout.decode())
