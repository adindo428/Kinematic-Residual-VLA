#!/usr/bin/env python3
"""Freeze supplied checkpoints and identical custom Table-1 scenarios on Palmetto."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/scratch/aparame/Dynamic_VLA')


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    import torch
    if (ROOT/'catalog.json').exists():
        print('Existing pinned catalog retained',flush=True);return
    entries={
        'pi05_baseline':('π0.5','main8-replicas-h200-20261021/runs/pi05_baseline/latest.pt'),
        'pi05_kr':('π0.5 + KR','main8-replicas-h200-20261021/runs/pi05_kr/latest.pt'),
        'vlash_baseline':('VLASH','h200-microbatch-20261022/vlash_baseline/runs/vlash_baseline/latest.pt'),
        'vlash_kr':('VLASH + KR','h200-microbatch-20261022/vlash_kr/runs/vlash_kr/latest.pt'),
        'smolvla_baseline':('SmolVLA','main8-h200-only-20261022/runs/smolvla_baseline/latest.pt'),
        'smolvla_kr':('SmolVLA + KR','main8-h200-only-20261022/runs/smolvla_kr/latest.pt'),
        'domv100':('DOM + velocity','main8-h200-only-20261022/runs/domv100/latest.pt')}
    pins=ROOT/'pinned';pins.mkdir(exist_ok=True)
    # Hard links retain immutable atomic checkpoint generations across source
    # replacement/cleanup. Never chmod a hard link to the user's training file.
    for name,(_,rel) in entries.items():
        dst=pins/(name+'.pt')
        if not dst.exists():os.link((BASE/rel).resolve(),dst)
    original_source=BASE/'runs/official_dom_table1_20261009/source'
    source=ROOT/'source'
    if not source.exists():
        shutil.copytree(original_source,source,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc'))
    native=BASE/'main8-20k-10epochs-20261021-r2/delta-domv-f991015a'
    native_dst=ROOT/'dom_source'
    if not native_dst.exists():
        native_dst.mkdir()
        for path in native.rglob('*.py'):
            if '__pycache__' in path.parts:continue
            target=native_dst/path.relative_to(native);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    upstream=BASE/'cross-vla-krfm-20261019/upstream-vlash/vlash'
    vlash=ROOT/'vlash_source/vlash'
    if not vlash.exists():shutil.copytree(upstream,vlash,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    suite=read(ROOT/'suite.json');models={}
    for name,(label,rel) in entries.items():
        print('Freezing '+name,flush=True)
        original=(BASE/rel).resolve();family=original.parents[2];code=ROOT/'model_code'/name;code.mkdir(parents=True,exist_ok=True)
        for item in ('policy.py','native_dom.py','common.py','vlash_runtime.py','statistics.json',
                     'data-provenance.json','checkpoints.json'):
            shutil.copy2(family/item,code/item)
        recipe=read(family/'recipes'/f'{name}.json')
        write(code/'recipes'/f'{name}.json',recipe)
        if name=='domv100':
            parent=code/'parents/domv100';parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(family/'parents/domv100/config.json',parent/'config.json')
        checkpoint=pins/(name+'.pt');state=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        models[name]={'label':label,'original_checkpoint':str(BASE/rel),'resolved_checkpoint':str(original),
            'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
            'checkpoint_mtime_ns':checkpoint.stat().st_mtime_ns,'checkpoint_sha256':sha(checkpoint),
            'code_root':str(code),'recipe':str(code/'recipes'/f'{name}.json'),
            'method':recipe['method'],'kr':recipe['kr'],'identity':state['identity'],'progress':state['progress']}
        del state
        root=ROOT/'results'/name;lists=root/'test_lists';lists.mkdir(parents=True,exist_ok=True)
        tests=[]
        for test in suite['cases']:
            path=ROOT/'configs'/Path(test['path']).name
            if sha(path)!=test['config_sha256']:raise ValueError('Transferred scenario differs from local suite')
            tests.append({**test,'path':str(path)})
        for i,dim in enumerate(suite['dimension_order']):
            paths=[r['path'] for r in tests if r['dimension']==dim]
            (lists/f'{i:02d}_{dim}.txt').write_text('\n'.join(paths)+'\n')
        pilot=[next(r['path'] for r in tests if Path(r['path']).name==name)
               for name in suite['pilot_names']]
        (lists/'pilot.txt').write_text('\n'.join(pilot)+'\n')
        write(root/'protocol.json',{'benchmark':'Custom task-aligned DOM mini-suite',
            'dimension_order':suite['dimension_order'],'tests':tests,'expected_trials':360,
            'trials_per_configuration':20,'pilot_trials':9,'requested_gpu':'l40s:2',
            'checkpoint':models[name],'source_template_sha256':suite['source_request_sha256'],
            'physical_actions':'Current-reference Cartesian XYZ/Euler deltas reconstructed exactly once; gripper absolute.',
            'execution':'Asynchronous, index-aligned 20-step action chunks; stale actions discarded.',
            'scope':'Same 18 scenarios as local old KRV100, not official 1800-trial Table 1.',
            'runtime_caveat':'Local old KRV100 used two A6000 and Isaac Lab 2.2.1. Remote L40S/container version recorded per model.'})
        write(root/'suite_status.json',{'state':'prepared','expected_trials':360})
    files={}
    for directory in ('workstation','evaluation','model_code','dom_source','vlash_source','source','configs'):
        for path in (ROOT/directory).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json'):files[str(path)]=sha(path)
    catalog={'models':models,'model_order':list(entries),'dimension_order':suite['dimension_order'],
        'official_source':str(source),'dom_source':str(native_dst),'vlash_source':str(vlash.parent),
        'assets':str(BASE/'dom_benchmark'),'simulator_container':'/scratch/aparame/isaaclab.sif',
        'python':str(BASE/'cross-vla-krfm-five-epochs-20261020/env/bin/python'),
        'apptainer':'/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/apptainer-1.1.9-q6riruwvgo2gsiwyeebasmym7f5a3hgq/bin/apptainer',
        'files_sha256':files,'upstream_commit':subprocess.check_output(['git','-C',str(original_source),'rev-parse','HEAD'],text=True).strip()}
    write(ROOT/'catalog.json',catalog);print('Pinned seven models; catalog complete',flush=True)


if __name__=='__main__':main()
