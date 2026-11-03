"""Freeze and validate terminal production checkpoints for explicit migration."""
from pathlib import Path
import json
import subprocess
import time
import torch
from common import atomic, fingerprint, sha

ROOT=Path(__file__).resolve().parent
OLD=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-a100-r2-20261019')
CONTROL=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-opt-control-20261019')


def main():
    stop=json.loads((CONTROL/'stop-state.json').read_text())
    if not stop.get('complete') or len(stop.get('old_requeued_jobs_cancelled',[]))!=3:
        raise ValueError('Old checkpointed continuations have not all stopped')
    state=json.loads((OLD/'submissions.json').read_text())
    jobs=[v['job_id'] for k,v in state.items() if k.endswith('_train')]
    deadline=time.time()+120
    while subprocess.check_output(['squeue','-j',','.join(jobs),'-h','-o','%i'],text=True).strip():
        if time.time()>deadline:raise ValueError('An old A100 production is still active or queued')
        time.sleep(1)
    prepared=json.loads((ROOT/'prepared.json').read_text())
    receipts={}
    for recipe_path in sorted((ROOT/'recipes').glob('*.json')):
        name=recipe_path.stem
        recipe=json.loads(recipe_path.read_text())
        if recipe_path.read_bytes()!=(OLD/'recipes'/recipe_path.name).read_bytes():
            raise ValueError('Recipe changed')
        parent=OLD/'runs'/name
        path=parent/'latest.pt'
        if not path.exists():
            receipts[name]={'fresh_start':True,'reason':'No verified production checkpoint; never resume readiness weights'}
            continue
        latest=json.loads((parent/'latest.json').read_text())
        provenance=json.loads((parent/'provenance.json').read_text())
        if provenance['recipe']!=recipe or provenance['preparation']!=prepared:
            raise ValueError('Parent recipe or frozen preparation differs')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        if checkpoint['identity']!=latest['identity'] or checkpoint['progress']!=latest['progress']:
            raise ValueError('Parent checkpoint and saved progress disagree')
        if len(checkpoint['rng'])!=2 or not checkpoint['optimizer']['state']:
            raise ValueError('Incomplete parent optimizer/per-rank RNG state')
        receipt={'checkpoint':str(path),'checkpoint_sha256':sha(path),
            'identity':checkpoint['identity'],'recipe_fingerprint':fingerprint(recipe),
            'preparation':prepared,'hardware':provenance['hardware'],'world_size':2,
            'progress':checkpoint['progress'],'parent_job_id':provenance['job_id'],
            'parent_source':provenance['source'],'explicit_runtime_migration':True}
        del checkpoint
        atomic(ROOT/'resume'/f'{name}.json',receipt)
        target=ROOT/'runs'/name
        target.mkdir(parents=True,exist_ok=True)
        if (parent/'wandb.json').exists():
            atomic(target/'wandb.json',json.loads((parent/'wandb.json').read_text()))
        receipts[name]=receipt
    atomic(ROOT/'resume-prepared.json',{'complete':True,'runs':receipts,'stop':stop})
    print(json.dumps({name:value.get('progress',value) for name,value in receipts.items()}),flush=True)


if __name__=='__main__':main()
