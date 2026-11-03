from pathlib import Path
import ast, hashlib, importlib, json, os, shutil, sys, tarfile
stage=Path('/scratch/aparame/Dynamic_VLA/runs/tracking_table1_20261008')
source=stage/'source'
root=Path('/home/aparame/Research/Dynamic_VLA/RESULTS/tracking_table1_75epoch/20261008')
update=stage/'scoring_update_3cm'
update.mkdir(exist_ok=False)
with tarfile.open(stage/'scoring_3cm.tgz') as t:t.extractall(update)
files=['scripts/verify_benchmark.py','scripts/tracking_benchmark.py','scripts/summarize_benchmark.py',
       'scripts/rescore_previous.py','scripts/prepare_benchmark.py','tests/test_benchmark.py']
backup=root/'provenance'/'scoring_v3'
backup.mkdir(exist_ok=False)
for name in files:
    dest=backup/name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source/name,dest)
for name in ('protocol.json','launch.json','README.md'):shutil.copy2(root/name,backup/name)
# Prove that this update changes no trajectories or RGB perturbations.
a=ast.parse((source/'scripts/tracking_benchmark.py').read_text())
b=ast.parse((update/'scripts/tracking_benchmark.py').read_text())
for name in ('BenchmarkPath','ObservationCondition'):
    old=next(n for n in a.body if isinstance(n,ast.ClassDef) and n.name==name)
    new=next(n for n in b.body if isinstance(n,ast.ClassDef) and n.name==name)
    assert ast.dump(old)==ast.dump(new),name
completed=sorted(p.parent for kind in ('pilot','full') for p in (root/kind).glob('*/verification.json'))
(root/'provenance/completed_shards_before_scoring4.json').write_text(json.dumps([str(p) for p in completed],indent=2))
# Install verifier first; it can rescore old worker outputs after the live update.
for name in files:
    destination=source/name
    tmp=destination.with_name(destination.name+'.scoring4.tmp')
    shutil.copyfile(update/name,tmp);os.replace(tmp,destination)
sys.path.insert(0,str(source/'scripts'))
scoring=importlib.import_module('tracking_benchmark')
assert scoring.SCORING_VERSION==4
p=json.loads((root/'protocol.json').read_text())
p['scoring_history']=p.get('scoring_history',[])+[{'version':p['scoring_version'],
    'success_definition':p['success_definition'],'time_definition':p['time_definition']}]
p.update(scoring_version=4,success_definition=scoring.SUCCESS_DEFINITION,time_definition=scoring.TIME_DEFINITION,
    success_tolerance_m=.03,success_aggregation='arithmetic mean of per-episode time percentages',
    binary_success_threshold=None,initial_second_counts_for_success=True,
    scoring_change_reason='User requested mean episode coverage within 3 cm, removing the binary 25% rule.')
p.pop('required_tracking_fraction_at_least',None)
(root/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
launch=json.loads((root/'launch.json').read_text())
launch.update(scoring_version=4,success_tolerance_m=.03,binary_success_threshold=None,
    success_aggregation=p['success_aggregation'],initial_second_counts_for_success=True)
launch.pop('required_tracking_fraction_at_least',None)
(root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
(root/'README.md').write_text('DynamicVLA Table 1-style tracking adaptation\n\n'+scoring.SUCCESS_DEFINITION+
    '\n\nEach CR–DR column and the overall Success (%) are equal-weight means of episode percentages. '
    'The first second counts; the expert warmup does not. No 25% binary threshold remains. '
    'Policy terminations receive no credit for the unexecuted horizon; Completion (%) is separate in CSV/JSON.\n\n'+
    'Acquisition time: '+scoring.TIME_DEFINITION+'\n\n'+p['scope']+'\n\n'+
    'Four 75-epoch checkpoints, 1800 trials/model, 7200 total. Pilot 16830025 passed; full array 16830026; summary 16830027. '
    'All trials retain telemetry/action chunks; trial 0 of each setup retains three camera videos. '
    'Missing or infrastructure-failed trials block publication. Scoring version 4; previous protocols and source are retained in provenance. '
    'Current scores can be reproduced from saved telemetry without rerunning models.\n')
audit={'scoring_version':4,'previous_scoring_version':3,'success_definition':scoring.SUCCESS_DEFINITION,
    'physics_restarted':False,'policy_changed':False,'trajectory_classes_unchanged':True,
    'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in files}}
(root/'provenance/scoring_revision_3cm.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({'installed':4,'completed_shards_to_rescore':len(completed),'audit':audit},indent=2))
