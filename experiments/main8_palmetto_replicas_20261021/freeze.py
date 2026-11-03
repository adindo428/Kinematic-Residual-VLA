"""Freeze each site's preparation after environments and transfers validate."""
import argparse
import json
from pathlib import Path
from common import atomic, sha, fingerprint
from subset import require


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();r=a.root.resolve()
    lock=r/('implementation-lock-'+a.site+'.json')
    require(lock.exists(),'Environment source lock missing')
    files=['split.json','subset-report.json','statistics.json','data-provenance.json','checkpoints.json',lock.name,'site.json','parquet_hashes.json','video_hashes.json','role.sh']
    files += [str(f.relative_to(r)) for folder in ['recipes','runtimes'] for f in sorted((r/folder).glob('*.json'))]
    files += [str(f.relative_to(r)) for f in sorted((r/'parents').glob('*/migration.json'))]
    files += [str(f.relative_to(r)) for f in sorted((r/'trackers').glob('*.json'))]
    files += ['replica-config.json','node_role.py','allocation.py']
    source={p.name:sha(p) for p in sorted(r.glob('*.py')) if not p.name.startswith('._')}
    atomic(r/'source-manifest.json',{'files':source,'fingerprint':fingerprint(source)})
    atomic(r/('prepared-'+a.site+'.json'),{'complete':True,'files':{f:sha(r/f) for f in files},'source_sha256':sha(r/'source-manifest.json')})
    print(json.dumps({'frozen_site':a.site,'source':fingerprint(source),'artifacts':len(files)}),flush=True)


if __name__=='__main__':main()
