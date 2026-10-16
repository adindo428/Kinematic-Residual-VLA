#!/usr/bin/env python3
"""Offline pinned-file verifier used for the authorized Palmetto backfill."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--missing',type=Path,required=True)
    parser.add_argument('--revision',required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    manifest_sha=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    count=verified=size=verified_size=0
    start=last=time.monotonic()

    def check(row):
        name,length,expected,git_blob=row
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe verification path')
        path=root/relative
        try:
            if not path.is_file() or path.stat().st_size!=length:return False
            value=hashlib.sha1() if git_blob else hashlib.sha256()
            if git_blob:value.update(f'blob {length}\0'.encode())
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(2*1024**2),b''):value.update(block)
            return value.hexdigest()==expected
        except (FileNotFoundError,PermissionError):
            return False

    with args.manifest.open() as manifest,args.missing.open('wb') as missing,ThreadPoolExecutor(max_workers=2) as pool:
        rows=(json.loads(line) for line in manifest)
        while True:
            batch=list(itertools.islice(rows,64))
            if not batch:break
            for row,valid in zip(batch,pool.map(check,batch)):
                count+=1;size+=row[1]
                if valid:verified+=1;verified_size+=row[1]
                else:missing.write(row[0].encode()+b'\0')
            if time.monotonic()-last>=30:
                print(json.dumps({'phase':'remote_checksum_verification','verified_files':verified,'files_checked':count}),flush=True)
                last=time.monotonic()
    report={'complete':count==verified,'revision':args.revision,'files':count,'verified_files':verified,
            'bytes':size,'verified_bytes':verified_size,'manifest_sha256':manifest_sha,
            'elapsed_seconds':time.monotonic()-start,
            'selected_cameras':['observation.images.opst_cam','observation.images.wrist_cam']}
    temporary=args.report.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(args.report)
    print(json.dumps(report),flush=True)
    return 0 if report['complete'] else 2


if __name__=='__main__':raise SystemExit(main())
