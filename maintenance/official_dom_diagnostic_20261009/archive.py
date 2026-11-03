"""Archive completed diagnostic arms using the existing verified private-HF workflow."""
import fcntl
import json
from pathlib import Path
import sys

sys.path.insert(0,'/home/aparame/Research/Dynamic_VLA/maintenance/storage_relocation_20261009')
import archive_results as archive

root=Path('/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_diagnostic/20261009')
lock=(archive.BASE/'archive.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX)
assert archive.API.repo_info(archive.REPO,repo_type='dataset').private
records=[]
for name in ('original','gripper','gripper_aligned','gripper_aligned_dual'):
    folder=root/name
    verification=folder/'verification.json'
    if verification.exists() and json.loads(verification.read_text())['verified']:
        records.append(archive.archive_one(folder,'official_dom_diagnostic/20261009/'+name))
old=Path('/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_table1/20261009/full/01_DA')
if (old/'verification.json').exists():
    records.append(archive.archive_one(old,'official_dom_table1/20261009/full/01_DA'))
for path in root.iterdir():
    if path.is_file() and path.suffix in ('.json','.csv'):
        archive.API.upload_file(path_or_fileobj=path,
            path_in_repo='official_dom_diagnostic/20261009/metadata/'+path.name,
            repo_id=archive.REPO,repo_type='dataset',commit_message='Record DOM gripper and timing diagnostic')
archive.upload_metadata()
print(json.dumps({'verified_archives':[r['path_in_repo'] for r in records]},indent=2))
