#!/usr/bin/env python3
"""Archive completed result trees to private HF, then move verified working copies."""
import argparse
import datetime
import hashlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import tarfile
import time
from huggingface_hub import HfApi
from relocate import BASE,save,relocate

REPO='adi2440/krvla-evaluation-results'
RESULTS=Path('/home/aparame/Research/Dynamic_VLA/RESULTS')
JOURNAL=BASE/'cloud_journal'
UPLOAD=BASE/'hf_upload'
API=HfApi()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


class HashedReader:
    def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256()
    def read(self,size=-1):
        data=self.stream.read(size);self.hash.update(data);return data


def create_archive(source,archive):
    archive.parent.mkdir(parents=True,exist_ok=True)
    temporary=archive.with_name(archive.name+'.partial')
    manifest={'source':str(source),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{}}
    with tarfile.open(temporary,'w:gz',compresslevel=1,dereference=True) as tar:
        # Results inspected before selection contain no external directory links.
        for path in sorted(source.rglob('*')):
            if path.is_symlink():
                resolved=path.resolve()
                if not str(resolved).startswith(str(RESULTS)+'/') and not str(resolved).startswith(str(BASE)+'/'):
                    raise ValueError('External result link must be inspected: '+str(path))
                if path.is_dir():raise ValueError('Directory link needs explicit materialization: '+str(path))
            if not path.is_file():continue
            name=str(path.relative_to(source))
            if path.name in ('.env','token','stored_tokens','id_rsa','id_ed25519') or path.name.startswith('.env.'):
                raise ValueError('Credential-like file requires exclusion: '+name)
            if path.suffix in ('.json','.md','.txt','.log','.py','.sh','.yaml') and path.stat().st_size<10*1024**2:
                content=path.read_text(errors='replace')
                if re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}',content):
                    raise ValueError('Credential pattern requires exclusion: '+name)
            info=tar.gettarinfo(str(path),arcname=name)
            before=path.stat()
            with path.open('rb') as f:
                reader=HashedReader(f);tar.addfile(info,reader)
            after=path.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('File changed during archive: '+str(path))
            manifest['files'][name]={'size':after.st_size,'sha256':reader.hash.hexdigest()}
        data=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('ARCHIVE_MANIFEST.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
    # Read each stored member back; truncated/corrupted archive data cannot pass.
    checked={}
    with tarfile.open(temporary,'r:gz') as tar:
        for member in tar:
            if member.name=='ARCHIVE_MANIFEST.json':continue
            if not member.isfile():raise ValueError('Unexpected archive member')
            f=tar.extractfile(member);h=hashlib.sha256();count=0
            for block in iter(lambda:f.read(4*1024**2),b''):h.update(block);count+=len(block)
            checked[member.name]={'size':count,'sha256':h.hexdigest()}
    if checked!=manifest['files']:raise ValueError('Archive verification failed')
    temporary.replace(archive)
    save(archive.with_name(archive.name+'.manifest.json'),manifest)
    return manifest


def remote_verified(path,expected_hash,size,revision=None):
    info=API.get_paths_info(REPO,[path],repo_type='dataset',revision=revision)
    if len(info)!=1:return False
    entry=info[0]
    lfs=getattr(entry,'lfs',None)
    digest=(lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)) if lfs else None
    return entry.size==size and digest==expected_hash


def archive_one(source,prefix):
    source=Path(source);key=hashlib.sha256(prefix.encode()).hexdigest()[:16];record=JOURNAL/f'{key}.json'
    archive=UPLOAD/(prefix+'.tar.gz')
    if record.is_file():
        previous=json.loads(record.read_text())
        if previous.get('relocated') and source.is_symlink():return previous
    print('ARCHIVE_START',prefix,flush=True)
    manifest=create_archive(source,archive)
    digest=sha(archive);remote_path=prefix+'.tar.gz'
    if not remote_verified(remote_path,digest,archive.stat().st_size):
        commit=API.upload_file(path_or_fileobj=archive,path_in_repo=remote_path,repo_id=REPO,repo_type='dataset',commit_message='Archive verified completed results: '+prefix)
        if not remote_verified(remote_path,digest,archive.stat().st_size,revision=commit.oid):
            raise ValueError('Cloud archive checksum/size verification failed')
    API.upload_file(path_or_fileobj=archive.with_name(archive.name+'.manifest.json'),path_in_repo=prefix+'.manifest.json',repo_id=REPO,repo_type='dataset',commit_message='Record per-file integrity manifest')
    result={'source':str(source),'path_in_repo':remote_path,'repo_id':REPO,'archive_sha256':digest,
            'archive_bytes':archive.stat().st_size,'files':len(manifest['files']),
            'uncompressed_bytes':sum(r['size'] for r in manifest['files'].values()),
            'archive_readback_verified':True,'cloud_sha256_and_size_verified':True,
            'archived_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'relocated':False}
    save(record,result)
    # Check the current source content still equals the uploaded archive before relocation.
    current={str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()}
    if current!=set(manifest['files']):raise ValueError('Source inventory changed after archive')
    for name,item in manifest['files'].items():
        path=source/name
        if path.stat().st_size!=item['size'] or sha(path)!=item['sha256']:raise ValueError('Source changed after upload')
    destination=BASE/'results'/source.relative_to(RESULTS)
    result['relocation']=relocate(source,destination);result['relocated']=True
    save(record,result);print('ARCHIVED_AND_RELOCATED',prefix,flush=True)
    return result


def running_ids():
    lines=subprocess.check_output(['squeue','-h','-u','aparame','-o','%i|%T'],text=True).splitlines()
    return {line.split('|')[0] for line in lines if line.split('|')[1] in ('RUNNING','COMPLETING')}


def candidates():
    active=running_ids();selected=[]
    coord=RESULTS/'tracking_comparison_four_75epoch/Coordinate_Conditioned_Velocity/16787490'
    if coord.is_dir():
        summary=coord/'comparison_with_existing/summary.json'
        if summary.is_file() and json.loads(summary.read_text()).get('complete'):
            selected.append((coord,'coordinate_velocity/16787490'))
    track=RESULTS/'tracking_table1_75epoch/20261008'
    for mode in ('pilot','full'):
        for folder in sorted((track/mode).glob('*')):
            if not folder.is_dir() or not folder.name.isdigit():continue
            if f'16830026_{int(folder.name)}' in active:continue
            verify=folder/'verification.json'
            if verify.is_file() and json.loads(verify.read_text()).get('verified'):
                selected.append((folder,f'tracking_table1/20261008/{mode}/{folder.name}'))
    dom=RESULTS/'official_dom_table1/20261009'
    if (dom/'pilot_verified.json').is_file():selected.append((dom/'pilot','official_dom_table1/20261009/pilot'))
    for folder in sorted((dom/'full').glob('*')):
        if not folder.is_dir():continue
        if f'16868047_{int(folder.name[:2])}' in active:continue
        verify=folder/'verification.json'
        if verify.is_file() and json.loads(verify.read_text()).get('verified'):
            selected.append((folder,f'official_dom_table1/20261009/full/{folder.name}'))
    return selected


def upload_metadata():
    stage=BASE/'metadata_upload';stage.mkdir(parents=True,exist_ok=True)
    roots=[(RESULTS/'tracking_comparison_four_75epoch/Coordinate_Conditioned_Velocity/16787490/comparison_with_existing','coordinate_velocity/16787490/browse'),
           (RESULTS/'tracking_table1_75epoch/20261008','tracking_table1/20261008/metadata'),
           (RESULTS/'tracking_table1_75epoch/20261008/partial_CR_DA_LS','tracking_table1/20261008/partial_CR_DA_LS'),
           (RESULTS/'official_dom_table1/20261009','official_dom_table1/20261009/metadata')]
    for root,prefix in roots:
        if not root.is_dir():continue
        files=[p for p in root.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md','.png') and p.stat().st_size<20*1024**2]
        for p in files:
            dst=stage/prefix/p.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    records=[json.loads(p.read_text()) for p in sorted(JOURNAL.glob('*.json'))]
    save(stage/'archive_index.json',{'repo_id':REPO,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'scope':'Completed verified shards only; evaluations may still be running. Active trial folders are not moved.',
          'archives':records,'total_uncompressed_bytes':sum(r['uncompressed_bytes'] for r in records),
          'total_archive_bytes':sum(r['archive_bytes'] for r in records)})
    (stage/'README.md').write_text('''---
pretty_name: KRVLA evaluation archives
---
# KRVLA evaluation results (private archive)

Completed closed-loop evaluation results from Palmetto. This is a research-results archive, not training data or a claim of benchmark completion. Consult each run protocol and final summary; partial tables remain labelled partial.

- `coordinate_velocity/16787490`: Coordinate+Velocity comparison, reusing prior models; browse contains plots and metric tables.
- `tracking_table1/20261008`: four-model tracking adaptation. Success is mean per-episode time within 3 cm, not manipulation success. One room, ten seeded setups per dimension.
- `official_dom_table1/20261009`: released DynamicVLA checkpoint on actual DOM manipulation tests; task completion success and simulation time, with upstream streaming execution.

Each `.tar.gz` contains all files from one completed run/shard, including videos, raw telemetry, action chunks, manifests, logs and verification. An internal `ARCHIVE_MANIFEST.json` records per-file SHA256/size; its external copy can be read without downloading the full archive. `archive_index.json` records archive hashes, source paths, provenance and relocation status.

Recover an archive with an authenticated Hugging Face account:

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download("adi2440/krvla-evaluation-results", "coordinate_velocity/16787490.tar.gz", repo_type="dataset")
print(p)
```

Extract into a fresh local directory with `tar -xzf ARCHIVE.tar.gz -C DESTINATION`. Verify its archive SHA256 against `archive_index.json` and its per-file hashes against `ARCHIVE_MANIFEST.json`. No trained weights are intentionally uploaded here except any verification weights already embedded in result provenance.

Working copies are relocated to scratch only after archive readback and cloud hash/size verification; original home paths become symlinks. Scratch is temporary and unbacked: the cloud archive is the durable copy. Do not rely on a scratch symlink surviving purge indefinitely. Future completed results are added by the final archival job after the benchmark/summary jobs end.
''')
    API.upload_folder(repo_id=REPO,repo_type='dataset',folder_path=stage,commit_message='Update browseable metrics and archive inventory')
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--only-first',action='store_true');p.add_argument('--wait-for-summary',action='store_true');a=p.parse_args()
    BASE.mkdir(parents=True,exist_ok=True);JOURNAL.mkdir(exist_ok=True)
    import fcntl
    lock=(BASE/'archive.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    info=API.repo_info(REPO,repo_type='dataset')
    if not info.private:raise ValueError('The result archive must remain private')
    failures=[]
    for source,prefix in candidates():
        try:archive_one(source,prefix)
        except Exception as error:
            print('ARCHIVE_ERROR',prefix,type(error).__name__,str(error)[:300],flush=True)
            failures.append({'source':str(source),'prefix':prefix,'type':type(error).__name__,'error':str(error)[:300]})
        if a.only_first:break
    if a.wait_for_summary:
        deadline=time.monotonic()+7200
        while time.monotonic()<deadline:
            q=subprocess.run(['squeue','-h','-j','16830027,16868048','-o','%T|%R'],capture_output=True,text=True)
            lines=q.stdout.strip().splitlines()
            if not lines or all('DependencyNeverSatisfied' in line for line in lines):break
            time.sleep(30)
    records=upload_metadata()
    save(BASE/'latest_archive_status.json',{'verified_archives':len(records),'relocated':sum(r.get('relocated',False) for r in records),
         'failures':failures,'final_summaries_present':{
             'tracking':(RESULTS/'tracking_table1_75epoch/20261008/summary.json').is_file(),
             'official_dom':(RESULTS/'official_dom_table1/20261009/summary.json').is_file()},
         'remaining_active_jobs':sorted(running_ids()),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
