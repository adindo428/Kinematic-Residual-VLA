"""Checksum-verified cross-filesystem relocation, retaining the original path."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

BASE=Path('/scratch/aparame/storage_relocation_20261009')


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def relocate(source,destination):
    source=Path(source);destination=Path(destination)
    if source.is_symlink():
        if source.resolve()!=destination.resolve():raise ValueError('Source points to another destination')
        if not destination.is_dir():raise ValueError('Destination missing')
        return {'source':str(source),'destination':str(destination),'already_relocated':True}
    if not source.is_dir():raise ValueError(f'Missing directory: {source}')
    if not str(source).startswith('/home/aparame/'):raise ValueError('Source outside approved home')
    if not str(destination).startswith(str(BASE)+'/'):raise ValueError('Destination outside relocation root')
    destination.mkdir(parents=True,exist_ok=True)
    subprocess.run(['rsync','-aH','--',str(source)+'/',str(destination)+'/'],check=True)
    check=subprocess.run(['rsync','-aHnc','--delete','--itemize-changes','--',str(source)+'/',str(destination)+'/'],capture_output=True,text=True,check=True)
    if check.stdout.strip():raise ValueError('Source/destination changed or mismatched; original retained: '+check.stdout[:300])
    size=int(subprocess.check_output(['du','-sb',str(source)],text=True).split()[0])
    backup=source.with_name(source.name+'.relocation_backup_20261009')
    if backup.exists():raise ValueError('A previous relocation backup needs inspection')
    source.rename(backup)
    try:source.symlink_to(destination,target_is_directory=True)
    except BaseException:
        backup.rename(source);raise
    # Only the checksum-verified former copy is removed, with destination still intact.
    shutil.rmtree(backup)
    result={'source':str(source),'destination':str(destination),'checksum_verified':True,
            'original_directory_replaced_with_symlink':True,'logical_bytes':size,
            'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    key=hashlib.sha256(str(source).encode()).hexdigest()[:16]
    save(BASE/'journal'/f'{key}.json',result)
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cache',choices=('pip','uv'),required=True);a=p.parse_args()
    relocate(Path('/home/aparame/.cache')/a.cache,BASE/'caches'/a.cache)
