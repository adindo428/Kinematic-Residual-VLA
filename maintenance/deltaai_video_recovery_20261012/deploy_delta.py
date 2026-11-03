import json,subprocess
from pathlib import Path
work=Path('/work/nvme/bivv/aparameshwaran/dynamicvla');setup=work/'setup-video-clean-domv-20261012'
snapshots=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots')
source=snapshots/'training-20261012-video-clean/dom100';family=work/'domv-video-clean-ddp-20261012';old=work/'domv-bnfix-ddp-20261011'
assert not source.exists()
subprocess.check_call(['tar','-xzf',str(setup/'training.tar.gz'),'-C',str(snapshots)])
report=json.loads(subprocess.check_output(['python3',str(setup/'freeze_clean_family.py'),str(old),str(family),str(source)],universal_newlines=True))
print(json.dumps(report),flush=True)
(setup/'bounded-single-gpu.json').write_bytes((work/'setup-bnfix-domv-20261011/bounded-single-gpu.json').read_bytes())
status={'stage':'ready_for_corrected_readiness','production_submitted':False,'family':str(family),'source':str(source),
 'previous_training_job':'3360409','previous_wandb_id':'xf5q6j0h','previous_training_failure':'Empty MP4 for episode 195603; 80 updates logged but only initialization checkpoint persisted',
 'readiness_candidate_limit_minutes':15,'readiness_total_limit_minutes':20,'stop_after_first_safe':True,
 'change':'User-approved identical six-episode exclusion on DeltaAI/Palmetto, coordinated loader failure reporting, checkpoints every 25 updates through warmup'}
(setup/'status.json').write_text(json.dumps(status,indent=2)+'\n')
log=(setup/'coordinator.log').open('a');p=subprocess.Popen(['python3',str(setup/'coordinate.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
report['coordinator_pid']=p.pid
(setup/'deployment.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'coordinator_pid':p.pid}),flush=True)
