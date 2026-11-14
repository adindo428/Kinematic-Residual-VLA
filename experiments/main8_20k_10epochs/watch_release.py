"""Release this task's held DeltaAI job after the shared CPU staging gate.

Delta and DeltaAI have separate schedulers. Never use a Delta job ID as a
DeltaAI afterok dependency. This bounded helper releases only its recorded
held job after source/data/checkpoint preparation receipts verify.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from common import atomic, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--job',required=True);p.add_argument('--site',choices=['delta','deltaai'],default='deltaai')
    a=p.parse_args();r=a.root.resolve();status=r/(a.site+'-release-status.json');started=time.time()
    try:
        while time.time()-started<4*3600:
            marker=r/'staging-delta.json'
            ready=marker.exists() and json.loads(marker.read_text()).get('ready')
            if a.site=='delta':
                sizes={
                    'parents/pi05_baseline/latest.pt':40456772509,'parents/pi05_kr/latest.pt':40456772509,
                    'parents/vlash_baseline/latest.pt':20946855289,'parents/vlash_kr/latest.pt':20946855289}
                sizes.update(json.loads((r/'asset-sizes.json').read_text()) if (r/'asset-sizes.json').exists() else {'MISSING_ASSET_SIZES':1})
                ready=all((r/name).is_file() and (r/name).stat().st_size==size for name,size in sizes.items())
                ready=ready and all((r/name).exists() for name in ['source-checkpoint-hashes.json','implementation-lock-deltaai.json','parent-preparation/checkpoints.json','parent-preparation/implementation-lock.json'])
            if ready:
                state=json.loads((r/'submissions.json').read_text())[a.site]
                assert state['job_id']==a.job and state['held_for_shared_staging']
                assert sha(Path(state['script']))==state['script_sha256']
                if a.site=='deltaai':
                    prepared=json.loads((r/'prepared-deltaai.json').read_text())
                    assert prepared['complete'] and sha(r/'source-manifest.json')==prepared['source_sha256']
                    for name,value in prepared['files'].items():assert sha(r/name)==value,name
                subprocess.check_call(['scontrol','release',a.job])
                atomic(status,{'released':True,'job_id':a.job,'shared_staging_verified':True})
                return
            atomic(status,{'released':False,'job_id':a.job,'waiting_for':'shared CPU transfer/normalization/source verification','seconds':time.time()-started})
            time.sleep(30)
        raise RuntimeError('Shared staging did not finish within4h; held job retained')
    except Exception as exc:
        atomic(status,{'released':False,'job_id':a.job,'error':str(exc)})
        raise


if __name__=='__main__':main()
