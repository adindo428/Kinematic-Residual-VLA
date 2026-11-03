"""Adapt the exact Palmetto distributed trainer for training-only DeltaAI."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

here = Path(__file__).resolve().parent
source = here/'source'
target = here/'training-20261009T165212Z'/'dom100'
assert not target.exists()
shutil.copytree(source,target)
changes=[]
def replace(name,old,new):
    p=target/name; t=p.read_text()
    assert old in t,(name,old)
    p.write_text(t.replace(old,new))
    changes.append({'file':name,'old':old,'new':new})

shutil.rmtree(target/'slurm')
for name in ('dom100_submit.py','submit_dom100.sh','train_paired_velocity_pair.sh','run_dom100.sh'):
    (target/'scripts'/name).unlink(missing_ok=True)
replace('core/dom_training.py', ', source_digest, simulator_evaluation_mode', ', source_digest')
replace('core/dom_training.py','    if production:\n        simulator_evaluation_mode(root, recipe)\n','')
replace('core/dom_training.py','            if milestone:\n                enqueue_evaluations(root, destination, variant, progress, best)\n','')
replace('core/dom_training.py','        if production:\n            enqueue_evaluations(root, destination, variant, progress, best)\n','')
replace('core/dom_training.py','            enqueue_evaluations(root, destination, variant, progress, best)\n','')
p=target/'core/dom_training.py';t=p.read_text();p.write_text(t[:t.index('\n\ndef enqueue_evaluations(')]+'\n')
replace('core/dom_distributed_training.py','build_policy, enqueue_evaluations,','build_policy,')
replace('core/dom_distributed_training.py',', simulator_evaluation_mode','')
replace('core/dom_distributed_training.py','    simulator_evaluation_mode(root,recipe)\n','')
replace('core/dom_distributed_training.py','            if milestone:\n                leader_call(lambda:enqueue_evaluations(root,destination,variant,progress,best))\n','')
replace('core/dom_distributed_training.py','            leader_call(lambda:enqueue_evaluations(root,destination,variant,progress,best))\n','')
replace('utils/dom_experiment.py','"scripts", "simulations"','"scripts"')
p=target/'utils/dom_experiment.py';t=p.read_text();start=t.index('\n\ndef simulator_evaluation_mode(');end=t.index('\n\ndef grouped_stratified_split(',start);p.write_text(t[:start]+t[end:])
replace('scripts/dom100.py',', simulator_evaluation_mode','')
replace('scripts/dom100.py','        from scripts.dom_closed_loop import validate_integration\n','')
replace('scripts/dom100.py','        if simulator_evaluation_mode(root, recipe) == "during_training":\n            validate_integration(root)\n','')
replace('scripts/dom100.py','result["simulator_evaluation_mode"] = simulator_evaluation_mode(root)','result["simulator_evaluation_mode"] = "disabled_on_deltaai"')
replace('scripts/dom100.py','result["evaluation_plan_sha256"] = file_digest(root/"evaluation_plan.json") if (root/"evaluation_plan.json").exists() else None','result["evaluation_plan_sha256"] = None')
(target/'scripts/dom100_env.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
: "${DVLA_PROJECT_DIR:?Set the frozen DeltaAI training project}"
: "${DOM100_ROOT:?Set the frozen DeltaAI training family}"
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MALLOC_ARENA_MAX=4 PYTHONUNBUFFERED=1
''')
# This runner preserves the already configured native/ARM environment.
for p in target.rglob('*.py'):
    tree=ast.parse(p.read_text(),filename=str(p))
    for node in ast.walk(tree):
        names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        assert not any(n.startswith(('isaac','omni','simulations','scripts.dom_closed_loop')) for n in names),str(p)

record={'palmetto_source':'/scratch/aparame/Dynamic_VLA/third_party/dom100-ddp-es-20261009',
        'variant':'domv100','simulator_payload_present':False,'changes':changes,
        'files':{str(p.relative_to(target)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in target.rglob('*') if p.is_file()}}
(here/'SOURCE_ADAPTATION.json').write_text(json.dumps(record,indent=2)+'\n')
archive=here/'deltaai-training.tar.gz'
with tarfile.open(archive,'w:gz') as out:out.add(target.parent,arcname=target.parent.name)
print(json.dumps({'files':len(record['files']),'archive_bytes':archive.stat().st_size,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
