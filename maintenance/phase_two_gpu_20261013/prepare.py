import json,shutil,sys
from pathlib import Path
base=Path('/scratch/aparame/Dynamic_VLA')
oldsource=base/'third_party/dom100-ddp-video-clean-20261012';source=base/'third_party/dom100-phase-two-gpu-20261013'
old=base/'dom100-audited-20261012';root=base/'dom100-phase-two-gpu-20261013'
assert not source.exists() and not root.exists(),'Refuse to overwrite prior snapshot'
shutil.copytree(oldsource,source,ignore=shutil.ignore_patterns('__pycache__','.git','*.out','*.err'))
p=source/'scripts/dom100_distributed.py';s=p.read_text();assert 'world not in (4,8)' in s;s=s.replace('world not in (4,8)','world not in (2,4,8)').replace('four/eight GPUs','two/four/eight GPUs').replace('four or eight GPUs','two, four or eight GPUs');p.write_text(s);compile(s,str(p),'exec')
p=source/'scripts/run_dom100_distributed.sh';s=p.read_text();assert '4|8)' in s;s=s.replace('4|8)','2|4|8)').replace('four or eight GPUs','two, four or eight GPUs');p.write_text(s)
sys.path.insert(0,str(source));from utils.dom_experiment import digest,file_digest,source_digest,atomic_json,VARIANTS
assert VARIANTS['krvp100']==(True,True,'causal_rules')
m=json.loads((old/'manifest.json').read_text());root.mkdir()
for name,h in m['artifacts'].items():
 assert file_digest(old/name)==h,name
 target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to((old/name).resolve())
for name in ['phase_cache','evaluation_plan.json']:
 if not (root/name).exists():(root/name).symlink_to((old/name).resolve())
p=root/'initialization/model.safetensors';p.symlink_to((old/'initialization/model.safetensors').resolve());assert file_digest(p)==m['weights_sha256']
parent=m.pop('fingerprint');m.update(initialization=str(root/'initialization'),source_sha256=source_digest(source),runtime_adaptation_parent=parent);m['fingerprint']=digest(m);atomic_json(root/'manifest.json',m)
r=json.loads((old/'recipe.json').read_text());parentrecipe=r.pop('fingerprint');r.update(manifest=m['fingerprint'],supported_world_sizes=[2,4,8]);r['fingerprint']=digest(r);atomic_json(root/'recipe.json',r)
(root/'logs').mkdir();(root/'submissions').mkdir()
atomic_json(root/'provenance/two_gpu.json',{'parent_family':str(old),'parent_manifest':parent,'parent_recipe':parentrecipe,'source':str(source),'source_sha256':m['source_sha256'],'manifest':m['fingerprint'],'recipe':r['fingerprint'],'variant':'krvp100','change':'Allow two-GPU launch; identical model and data; batch352; phase gate causal_rules','split_sha256':m['split_sha256']})
for label,gpu,feature,mb in [('h2x2','h200','gpu_h200',44),('h2','h100','gpu_h100',44),('a2','a100','gpu_a100_40gb',22)]:
 script=f'''#!/bin/bash -l
#SBATCH --job-name=krvp100-{label}
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus={gpu}:2
#SBATCH --constraint={feature}
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --requeue
#SBATCH --output={root}/logs/krvp100-{label}-%j.out
#SBATCH --error={root}/logs/krvp100-{label}-%j.err
set -euo pipefail
export DVLA_PROJECT_DIR={source}
export DOM100_ROOT={root}
export DOM100_DDP_GPUS=2 DOM100_DDP_MICROBATCH={mb} DOM100_DDP_WORKERS=1
cd "$DVLA_PROJECT_DIR"
exec bash scripts/run_dom100_allocation_candidate.sh krvp100
'''
 (root/'submissions'/f'krvp100_{label}.sbatch').write_text(script)
print((root/'provenance/two_gpu.json').read_text())
