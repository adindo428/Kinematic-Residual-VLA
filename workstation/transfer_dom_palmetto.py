#!/usr/bin/env python3
"""Import pinned DOM files from a user's authenticated Palmetto DTN connection."""
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import signal
import sqlite3
import subprocess
import time

WORK = Path(os.environ['KRVLA_WORK']).resolve()
REPO = Path(os.environ['KRVLA_REPO']).resolve()
STATE = WORK/'training/dom-full'
DATA = WORK/'cache/huggingface/lerobot/hzxie/DOM'
USER = os.environ.get('PALMETTO_DOM_USER','aparame')
HOST = os.environ.get('PALMETTO_DOM_HOST','hpcdtn01.rcd.clemson.edu')
SOURCE = os.environ.get('PALMETTO_DOM_SOURCE',f'/scratch/{USER}/.cache/huggingface/lerobot/hzxie/DOM')
TARGET = f'{USER}@{HOST}'
SOCKET = WORK/'ssh/palmetto'
SSH = ['ssh','-S',str(SOCKET),'-o','ControlMaster=no','-o','BatchMode=yes',
       '-o',f'UserKnownHostsFile={WORK}/ssh/known_hosts','-o','StrictHostKeyChecking=yes',
       '-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=4']
DOWNLOAD_UNIT = 'krvla-dom-full-download-20261010.service'
STATUS = STATE/'palmetto_transfer_status.json'


def status(phase, **details):
    row = {'phase':phase,'updated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'host':HOST,'username':USER,'source':SOURCE,'destination':str(DATA),**details}
    temporary = STATUS.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(row,indent=2)+'\n')
    temporary.replace(STATUS)
    print(json.dumps(row),flush=True)


def remote(command):
    result = subprocess.run([*SSH,TARGET,command],stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            text=True,timeout=60)
    if result.returncode:
        raise RuntimeError(f'Palmetto preflight failed: {result.stderr[-2000:]}')
    return result.stdout


def run_child(command):
    process = subprocess.Popen(command,cwd=REPO,start_new_session=True)
    try:
        return process.wait()
    finally:
        try:
            os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def resume_download():
    # A stopped transient unit can be garbage-collected; recreate if necessary.
    if subprocess.run(['systemctl','--user','start',DOWNLOAD_UNIT],stderr=subprocess.DEVNULL).returncode:
        subprocess.run(['systemd-run','--user','--unit',DOWNLOAD_UNIT.removesuffix('.service'),
                        '--description=Resume checksum-verified full DOM download after Palmetto import',
                        f'--working-directory={REPO}','--property=Restart=on-failure','--property=RestartSec=60',
                        '--property=KillMode=control-group',f'--property=StandardOutput=append:{WORK}/logs/dom-dataset-download.log',
                        '--property=StandardError=inherit','/bin/bash',str(REPO/'workstation/download-dom-full.sh')],check=True)


def main():
    if not re.fullmatch('[a-zA-Z0-9_-]+',USER) or HOST!='hpcdtn01.rcd.clemson.edu':
        raise ValueError('Use the verified Clemson data-transfer node and a plain username')
    if not SOURCE.startswith(f'/scratch/{USER}/') or '\n' in SOURCE or '..' in Path(SOURCE).parts:
        raise ValueError('Source must be a dataset path under your Palmetto scratch directory')
    STATE.mkdir(parents=True,exist_ok=True)
    # Keep HF downloads working until the user has authenticated locally.
    status('waiting_for_palmetto_login',command='bash workstation/palmetto-dom.sh connect')
    while subprocess.run(['ssh','-S',str(SOCKET),'-O','check',TARGET],
                         stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10).returncode:
        time.sleep(15)
    status('checking_remote_dataset')
    info = json.loads(remote('cat -- '+shlex.quote(SOURCE+'/meta/info.json')))
    local_info = json.loads((DATA/'meta/info.json').read_text())
    for field in ('total_episodes','total_frames','total_videos','fps','video_path','data_path'):
        if info.get(field)!=local_info.get(field):
            raise ValueError(f'Palmetto dataset metadata differs from the pinned release: {field}')
    remote('rsync --version >/dev/null')
    # Validate connection account and source before stopping the other writer.
    identity = remote('id -un').strip()
    if identity!=USER:
        raise ValueError('Authenticated SSH master belongs to a different Palmetto user')
    db = sqlite3.connect(STATE/'inventory.sqlite')
    config = dict(db.execute('SELECT key,value FROM configuration'))
    if config['revision']!='bbf6baa15f7253746d271a6e958e037afa537b1e' or config['dataset_root']!=str(DATA):
        raise ValueError('DOM inventory does not match the authorized release/destination')
    if db.execute('SELECT COUNT(*) FROM directories WHERE listed=0').fetchone()[0]:
        raise ValueError('Pinned file inventory is incomplete')
    subprocess.run(['systemctl','--user','stop',DOWNLOAD_UNIT],check=True)
    try:
        remaining = db.execute('SELECT COALESCE(SUM(size),0) FROM files WHERE verified=0').fetchone()[0]
        import shutil
        if shutil.disk_usage(DATA).free < remaining+64*1024**3:
            raise RuntimeError('SSD lacks sufficient free space for the transfer')
        listing = STATE/'palmetto_pending_files.list0'
        count = 0
        with listing.open('wb') as stream:
            for (name,) in db.execute('SELECT path FROM files WHERE verified=0 ORDER BY path'):
                path = Path(name)
                if path.is_absolute() or '..' in path.parts:
                    raise ValueError('Unsafe inventory path')
                stream.write(name.encode()+b'\0');count+=1
        partial = STATE/'rsync-partial';partial.mkdir(exist_ok=True)
        status('transferring',pending_files=count,pending_bytes=remaining)
        command = ['rsync','--archive','--copy-links','--no-owner','--no-group','--omit-dir-times',
                   '--size-only','--partial',f'--partial-dir={partial}',
                   '--from0',f'--files-from={listing}','--relative','--protect-args',
                   '--info=progress2,stats2','--human-readable','--timeout=180',
                   '-e',shlex.join(SSH),f'{TARGET}:{SOURCE.rstrip("/")}/',str(DATA)+'/']
        code = run_child(command)
        # Partial old cache/vanished source files are still useful; the exact
        # pinned hashes decide which bytes can be used, never rsync's exit alone.
        if code not in (0,23,24):
            raise RuntimeError(f'Palmetto rsync exited with {code}; transfer can be resumed')
        db.close();db=None
        status('verifying_copied_files',rsync_exit_code=code)
        code = run_child(['/bin/bash',str(REPO/'workstation/download-dom-full.sh'),'--verify-only'])
        if code==0:
            status('complete',all_pinned_files_verified=True)
        elif code==2:
            status('finishing_missing_files_from_hub',all_pinned_files_verified=False)
            resume_download()
        else:
            raise RuntimeError('Offline DOM checksum verification failed')
    except BaseException:
        resume_download()
        raise
    finally:
        if db is not None:db.close()


if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        status('failed',error_type=type(error).__name__)
        raise
