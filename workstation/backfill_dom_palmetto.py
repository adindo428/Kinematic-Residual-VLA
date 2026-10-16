#!/usr/bin/env python3
"""Complete and verify all DOM episodes/two training cameras on Palmetto."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import time

from transfer_dom_palmetto import WORK,REPO,STATE,DATA,USER,HOST,SOURCE,TARGET,SOCKET,SSH,remote,run_child

REVISION='bbf6baa15f7253746d271a6e958e037afa537b1e'
CAMERAS=['observation.images.opst_cam','observation.images.wrist_cam']
SCOPE="(path NOT LIKE 'videos/%' OR path LIKE 'videos/%/observation.images.opst_cam/%' OR path LIKE 'videos/%/observation.images.wrist_cam/%')"
STATUS=STATE/'palmetto_backfill_status.json'
ROOT=STATE/'palmetto-backfill'
REMOTE=SOURCE+'/.krvla-verification/'+REVISION


def status(phase,**details):
    value={'phase':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'revision':REVISION,'selected_cameras':CAMERAS,'host':HOST,'username':USER,
           'destination':SOURCE,**details}
    temporary=STATUS.with_suffix('.json.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(STATUS)
    print(json.dumps(value),flush=True)


def rsync(arguments,pull=False):
    partial=str(ROOT/'partial') if pull else REMOTE+'/partial'
    code=run_child(['rsync','--archive','--no-owner','--no-group','--no-perms','--omit-dir-times',
                    '--partial',f'--partial-dir={partial}','--protect-args',
                    '--info=progress2,stats2','--human-readable','--timeout=180',
                    '-e',shlex.join(SSH),*arguments])
    if code:raise RuntimeError(f'Palmetto backfill rsync exited with {code}')


def verify_remote():
    command=['python3',REMOTE+'/verify_dom_files.py','--root',SOURCE,
             '--manifest',REMOTE+'/expected.jsonl','--report',REMOTE+'/report.json',
             '--missing',REMOTE+'/missing.list0','--revision',REVISION]
    code=run_child([*SSH,TARGET,shlex.join(command)])
    if code not in (0,2):raise RuntimeError('Palmetto checksum verifier failed')
    result=json.loads(remote('cat -- '+shlex.quote(REMOTE+'/report.json')))
    expected=hashlib.sha256((ROOT/'expected.jsonl').read_bytes()).hexdigest()
    if result['manifest_sha256']!=expected or result['revision']!=REVISION or result['selected_cameras']!=CAMERAS:
        raise ValueError('Remote verification provenance differs from the pinned local dataset')
    return result


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    status('waiting_for_local_two_camera_dataset')
    while True:
        downloaded=json.loads((STATE/'download_status.json').read_text())
        if (downloaded.get('complete') and downloaded.get('revision')==REVISION
                and downloaded.get('completion_scope')=='all_episodes_training_cameras'
                and downloaded.get('selected_cameras')==CAMERAS and downloaded.get('checksum_verification')):break
        time.sleep(15)
    status('waiting_for_palmetto_connection')
    while subprocess.run(['ssh','-S',str(SOCKET),'-O','check',TARGET],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10).returncode:
        time.sleep(15)
    if remote('id -un').strip()!=USER:raise ValueError('SSH connection belongs to a different user')
    if not SOURCE.startswith(f'/scratch/{USER}/'):raise ValueError('Unexpected Palmetto destination')
    info=json.loads(remote('cat -- '+shlex.quote(SOURCE+'/meta/info.json')))
    if info['total_episodes']!=downloaded['episodes']:raise ValueError('Remote dataset episode count differs')
    db=sqlite3.connect('file:'+str(STATE/'inventory.sqlite')+'?mode=ro',uri=True)
    if db.execute(f'SELECT COUNT(*) FROM files WHERE {SCOPE} AND verified=0').fetchone()[0]:
        raise ValueError('Local pinned dataset is not complete')
    count=size=0
    with (ROOT/'files.list0').open('wb') as listing,(ROOT/'expected.jsonl').open('w') as manifest:
        for row in db.execute(f'SELECT path,size,checksum,git_blob FROM files WHERE {SCOPE} ORDER BY path'):
            name,length,checksum,git_blob=row
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe manifest path')
            listing.write(name.encode()+b'\0');manifest.write(json.dumps(list(row))+'\n')
            count+=1;size+=length
    db.close()
    remote('mkdir -p -- '+shlex.quote(REMOTE))
    status('copying_missing_files_to_palmetto',required_files=count,required_bytes=size)
    rsync(['--size-only','--relative','--from0',f'--files-from={ROOT}/files.list0',str(DATA)+'/',f'{TARGET}:{SOURCE}/'])
    rsync([str(ROOT/'expected.jsonl'),str(REPO/'workstation/verify_dom_files.py'),f'{TARGET}:{REMOTE}/'])
    status('verifying_palmetto_checksums',required_files=count,required_bytes=size)
    result=verify_remote()
    if not result['complete']:
        # Equal-sized old/corrupt files cannot be trusted by size-only rsync.
        # Repair the exact mismatches and require a fresh full verification.
        rsync([f'{TARGET}:{REMOTE}/missing.list0',str(ROOT/'repair.list0')],pull=True)
        status('repairing_checksum_mismatches',mismatches=result['files']-result['verified_files'])
        rsync(['--ignore-times','--relative','--from0',f'--files-from={ROOT}/repair.list0',str(DATA)+'/',f'{TARGET}:{SOURCE}/'])
        status('verifying_palmetto_checksums_after_repair')
        result=verify_remote()
    if not result['complete'] or result['files']!=count or result['bytes']!=size:
        raise ValueError('Palmetto still lacks pinned two-camera files')
    (ROOT/'remote_verified.json').write_text(json.dumps(result,indent=2)+'\n')
    status('complete',remote_checksum_verified=True,verified_files=result['verified_files'],verified_bytes=result['verified_bytes'])


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('failed',error_type=type(error).__name__)
        raise
