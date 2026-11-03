"""Archive verified friction arms after evaluation; leave incomplete evidence intact."""
import fcntl
import json
from pathlib import Path
import sys
sys.path.insert(0,'/home/aparame/Research/Dynamic_VLA/maintenance/storage_relocation_20261009')
import archive_results as archive
root=Path('/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_friction/20261009')
lock=(archive.BASE/'archive.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
assert archive.API.repo_info(archive.REPO,repo_type='dataset').private
records=[]
for name in ('control','contact_mu1','contact_mu2'):
    folder=root/name
    if (folder/'verification.json').exists() and json.loads((folder/'verification.json').read_text())['verified']:
        records.append(archive.archive_one(folder,'official_dom_friction/20261009/'+name))
for path in root.iterdir():
    if path.is_file() and path.suffix in ('.json','.csv'):
        archive.API.upload_file(path_or_fileobj=path,path_in_repo='official_dom_friction/20261009/metadata/'+path.name,
                               repo_id=archive.REPO,repo_type='dataset',commit_message='Record finger contact friction diagnostic')
archive.upload_metadata()
print(json.dumps({'verified_archives':[r['path_in_repo'] for r in records]},indent=2))
