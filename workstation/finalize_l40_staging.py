"""Submit one L40 only after the exact 4.5 runtime passes CPU preflight."""
import json
from pathlib import Path
import subprocess
import time

import palmetto_allocated_transport as transport

ROOT = Path(__file__).resolve().parents[1]/'scratch/dom45/evaluations/pi05-kr-l40-20261111'
REMOTE = '/scratch/aparame/Dynamic_VLA/pi05-kr-l40-lowclutter-20261111'


def main():
    receipt = ROOT/'stage/transfer-receipt.json'
    deadline = time.monotonic()+7200
    while not receipt.exists():
        if time.monotonic() > deadline: raise TimeoutError('Exact runtime transfer did not finish; no GPU requested')
        time.sleep(15)
    assert json.loads(receipt.read_text())['complete']
    expected = json.loads((ROOT/'inference_checkpoint_receipt.json').read_text())['inference_checkpoint_sha256']
    script = '''import hashlib,json,os,runpy,subprocess
from pathlib import Path
r=Path(REMOTE)
assert os.environ['SLURM_JOB_ID']=='20066627' and os.environ.get('SLURMD_NODENAME')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
assert sha(r/'inference_checkpoint.pt')==EXPECTED
runpy.run_path(str(r/'workstation/initialize_l40_deployment.py'),run_name='__main__')
sim=r/'envs/sim45/bin/python'
probe=subprocess.run(['/usr/bin/apptainer','exec','--cleanenv','--bind','/scratch/aparame:/scratch/aparame,/home/aparame:/home/aparame',
 '/scratch/aparame/isaaclab.sif','env','-u','PYTHONPATH','-u','PYTHONHOME',str(sim),'-c',
 "import importlib.metadata as m,json; assert m.version('isaacsim')=='4.5.0'; print(json.dumps({'isaacsim':m.version('isaacsim')}))"],capture_output=True,text=True,timeout=90)
assert probe.returncode==0,probe.stderr[-2000:]
metadata={'passed':True,'simulator':'4.5.0','isaaclab':'2.2.1','cpu_host':os.uname().nodename,'cpu_job':os.environ['SLURM_JOB_ID'],
 'inference_checkpoint_sha256':EXPECTED,'sim_archive_sha256':sha(r/'sim45.tar.gz')}
(r/'cpu_preflight.json').write_text(json.dumps(metadata,indent=2)+chr(10))
# Capture failures even if Python fails before the evaluation controller loads.
launcher=r/'workstation/run_l40_low_clutter.py';text=launcher.read_text()
replacement=chr(10).join(["if __name__ == '__main__':", "    try: main()", "    except BaseException as error:",
 "        (ROOT/'deployment_failure.json').write_text(json.dumps({'error':str(error)},indent=2))", "        raise"])
text=text.replace("if __name__ == '__main__': main()", replacement)
launcher.write_text(text)
control=r/'scratch/dom45/evaluations/low-clutter-full-table1-20261102/l40_suite';p=control/'plan.json';plan=json.loads(p.read_text())
plan['files_sha256'][str(launcher)]=sha(launcher);p.write_text(json.dumps(plan,indent=2)+chr(10))
submitted=subprocess.check_output(['sbatch','--parsable',str(r/'evaluate.sbatch')],text=True).strip().split(';')[0]
job={'job_id':submitted,'account':'omniverse_dndawso_manual_users','partition':'omniverse','gpus':1,'gpu_type':'l40',
 'cpus':6,'memory_gib':48,'walltime':'3-00:00:00','required_simulator':'4.5.0','latency_gate_required_before_scoring':True,
 'submitted_from_cpu_job':os.environ['SLURM_JOB_ID']}
(r/'job.json').write_text(json.dumps(job,indent=2)+chr(10))
print(json.dumps({'cpu_preflight':metadata,'job':job},indent=2))
'''.replace('REMOTE', repr(REMOTE)).replace('EXPECTED', repr(expected))
    result = transport.run(['python3','-'], input_text=script, timeout=1800)
    parsed = json.loads(result)
    (ROOT/'submission.json').write_text(json.dumps(parsed, indent=2)+'\n')
    print(result, flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        (ROOT/'staging_failure.json').write_text(json.dumps({'error': str(error),
            'action': 'No simulator-version substitution; retain local evaluation fallback'}, indent=2)+'\n')
        raise
