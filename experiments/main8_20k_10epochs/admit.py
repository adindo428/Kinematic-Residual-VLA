"""CPU staging gate: validate finished transfers before allocating GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from common import atomic, sha
from subset import require


def main():
    p=argparse.ArgumentParser();p.add_argument('--site',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--source-hashes',action='store_true')
    a=p.parse_args();r=a.root.resolve();pal=a.site=='palmetto'
    if a.source_hashes:
        require(pal,'Source digest audit runs on Palmetto')
        result={
            'pi05_baseline':{'bytes':40456772509,'sha256':'ec522d6f1c9243e37575b61cd9068e04384830e36e74f4399a999cef2639ad75'},
            'pi05_kr':{'bytes':40456772509,'sha256':'09c0466b8809bc8e5f0dc248da135bcb186c64bb0759557def0bfdaee9119e6c'}}
        for name in ['vlash_baseline','vlash_kr']:
            file=r/'parents'/name/'latest.pt'
            result[name]={'bytes':file.stat().st_size,'sha256':sha(file)}
            print(json.dumps({name:result[name]}),flush=True)
        atomic(r/'source-checkpoint-hashes.json',result)
        return
    expected={'parents/domv100/model.safetensors':1118678424,
              'parents/domv100/training_state.pt':2200710501} if pal else {
              'parents/pi05_baseline/latest.pt':40456772509,'parents/pi05_kr/latest.pt':40456772509,
              'parents/vlash_baseline/latest.pt':20946855289,'parents/vlash_kr/latest.pt':20946855289}
    required=['delta-domv-f991015a/core/dom_training.py'] if pal else [
        'source-checkpoint-hashes.json','parent-preparation/checkpoints.json','parent-preparation/implementation-lock.json',
        'implementation-lock-deltaai.json',
        'assets/models--lerobot--pi05_base/snapshots/22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0/model.safetensors',
        'assets/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/tokenizer.json']
    started=time.monotonic()
    while True:
        missing=[name for name,n in expected.items() if not (r/name).is_file() or (r/name).stat().st_size!=n]
        missing += [name for name in required if not (r/name).is_file()]
        if not missing:break
        atomic(r/('staging-'+a.site+'.json'),{'ready':False,'waiting_for':missing,'seconds':time.monotonic()-started})
        print(json.dumps({'waiting_for':missing,'seconds':int(time.monotonic()-started)}),flush=True)
        require(time.monotonic()-started<3.5*3600,'Transfer wait exceeded3.5h; refusing GPU admission')
        time.sleep(30)
    if pal:
        known={'parents/domv100/model.safetensors':'6ba0df746da2f910066955df052620fc0083220e87f27ebcbd25990e51281630',
               'parents/domv100/training_state.pt':'9be33e781e81a7ab74b44de3cdaf2e0bb7432c7673e141091c2eab7fcf6e5921'}
        for name,value in known.items():require(sha(r/name)==value,'DOM source/destination digest differs: '+name)
        python='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python'
    else:
        originals=json.loads((r/'source-checkpoint-hashes.json').read_text())
        for name,spec in originals.items():
            file=r/'parents'/name/'latest.pt'
            require(file.stat().st_size==spec['bytes'] and sha(file)==spec['sha256'],'Transferred checkpoint differs: '+name)
        python=str(r/'env-delta-x86/bin/python')
    subprocess.check_call([python,str(r/'setup.py'),'--root',str(r),'--site',a.site])
    subprocess.check_call([python,str(r/'setup.py'),'--root',str(r),'--site',a.site,'--lock-only'])
    subprocess.check_call([python,str(r/'freeze.py'),'--root',str(r),'--site',a.site])
    if not pal:
        subprocess.check_call([python,str(r/'freeze.py'),'--root',str(r),'--site','deltaai'])
    atomic(r/('staging-'+a.site+'.json'),{'ready':True,'all_transfer_digests_verified':True,'seconds':time.monotonic()-started})
    print(json.dumps({'staging_complete':a.site}),flush=True)


if __name__=='__main__':main()
