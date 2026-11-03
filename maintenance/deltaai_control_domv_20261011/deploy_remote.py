from pathlib import Path
import subprocess,json,time
work=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
setup=work/'setup-control-domv-20261011'
snapshot=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots')
assert not (snapshot/'training-20261011-control-gloo').exists()
subprocess.check_call(['tar','-xzf',str(setup/'training.tar.gz'),'-C',str(snapshot)])
result=json.loads(subprocess.check_output(['python3',str(setup/'freeze_remote.py')],universal_newlines=True))
print(json.dumps(result),flush=True)
(setup/'bounded-single-gpu.json').write_bytes((work/'setup-lean-domv-20261010/bounded-single-gpu.json').read_bytes())
status=dict(stage='ready_for_bounded_check',production_submitted=False,family=result['family'],source=result['source'],
    readiness_candidate_limit_minutes=15,readiness_total_limit_minutes=20,stop_after_first_safe=True,
    previous_readiness_jobs=['3349092','3350039','3354039'],
    previous_failure='Two smoke updates and checkpoint files complete; synchronization stalled during post-reload GPU memory telemetry',
    change='CPU Gloo group for checkpoint metadata and readiness telemetry; one measured 44-microbatch two-worker configuration')
(setup/'status.json').write_text(json.dumps(status,indent=2)+'\n')
log=(setup/'coordinator.log').open('a')
p=subprocess.Popen(['python3',str(setup/'coordinate.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(setup/'coordinator.pid').write_text(str(p.pid)+'\n')
result['coordinator_pid']=p.pid
(setup/'deployment.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'coordinator_pid':p.pid}),flush=True)
