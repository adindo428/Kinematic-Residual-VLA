from pathlib import Path
import sys,json,subprocess,hashlib,shlex
H=Path(__file__).resolve().parent;R='/scratch/aparame/Dynamic_VLA/main8-smol-closedloop-pilot-r2-20261027';T='/scratch/aparame/Dynamic_VLA/main8-h200-only-20261022';S='/scratch/aparame/Dynamic_VLA/runs/official_dom_table1_20261009/source';P='/scratch/aparame/Dynamic_VLA/cross-vla-krfm-five-epochs-20261020/env/bin/python'
SSH=['ssh','-F','/dev/null','-S','/private/tmp/vlcm-palmetto-k1vtw3i7/socket','-o','BatchMode=yes','-o','ConnectTimeout=5','aparame@slogin.palmetto.clemson.edu']
def remote(cmd,data=None):return subprocess.run(SSH+[cmd],input=data,capture_output=True,timeout=180,check=True).stdout.decode()
def main():
 assert not (H/'submission.json').exists(),'Reconcile recorded job; do not retry'
 assert 'smol-closedloop-pilot' not in remote('squeue -u aparame -h -o "%j"')
 remote('test ! -e '+R+' && mkdir -p '+R+'/scripts '+R+'/results/test_lists '+R+'/logs')
 for f in ('client.py','run.py','test_client.py'):remote('cat > '+R+'/'+f,(H/f).read_bytes())
 for f in ('prepare.py','report.py','server_observer.py'):remote('cp /scratch/aparame/Dynamic_VLA/runs/official_dom_table1_20261009/scripts/'+f+' '+R+'/scripts/'+f)
 remote('ln -s '+S+' '+R+'/source')
 code="""from pathlib import Path
import json,hashlib,zipfile
R=Path('"""+R+"""');T=Path('"""+T+"""');S=Path('"""+S+"""');assets=Path('/scratch/aparame/Dynamic_VLA/dom_benchmark');old=Path('/home/aparame/Research/Dynamic_VLA/RESULTS/official_dom_table1/20261009')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
protocol=json.loads((old/'protocol.json').read_text());names=(old/'test_lists/pilot.txt').read_text();assert len(names.splitlines())==9
with zipfile.ZipFile(assets/'DOM-Test.zip') as z:
 members={Path(n).name:n for n in z.namelist() if n.endswith('.json')}
 for name in names.splitlines():assert Path(name).read_bytes()==z.read(members[Path(name).name])
(R/'results/test_lists/pilot.txt').write_text(names)
protocol.update(benchmark='SmolVLA completed 20k10 checkpoint closed-loop integration pilot',checkpoint=str(T/'runs/smolvla_baseline/latest.pt'),streaming=False,observation_offsets=[0],requested_trials=9,training_model='SmolVLA baseline',limitations=['Nine trials verify integration only; not a success-rate comparison or exact paper reproduction.','Official newer Isaac 5.1/Lab2.3.2 simulator and physics retained. Known gripper/environment reproduction limitations remain.','Native SmolVLA20-action open-loop queue, one current observation; separate control protocol from DynamicVLA streaming.','KR is excluded until current simulator velocity is verified; no fake zero velocity in KR.'])
(R/'results/protocol.json').write_text(json.dumps(protocol,indent=2))
files={str(p):sha(p) for p in list(T.glob('*.py'))+[T/'recipes/smolvla_baseline.json',T/'statistics.json',T/'data-provenance.json',T/'checkpoints.json',T/'runs/smolvla_baseline/latest.json',T/'runs/smolvla_baseline/completed.json',T/'runs/smolvla_baseline/provenance.json',S/'scripts/inference.py']}
config={'training_root':str(T),'official_source':str(S),'files':files,'checkpoint_sha256':sha(T/'runs/smolvla_baseline/latest.pt')}
(R/'config.json').write_text(json.dumps(config,indent=2));print(json.dumps(config))
"""
 cfg=remote('python3 -',code.encode());(H/'config.json').write_text(cfg)
 tests=remote('export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}; cd '+R+' && '+P+' -m unittest -v 2>&1 && '+P+' -c '+shlex.quote('from torchcodec.decoders import VideoDecoder; print("Native TorchCodec import passed")'));(H/'cpu-contracts.txt').write_text(tests);print(tests,flush=True)
 # Freeze commands and code in a separate evaluation directory.
 body='''#!/usr/bin/env bash
#SBATCH --job-name=smol-closedloop-r2
#SBATCH --account=cuuser_yue6_vae_verification
#SBATCH --partition=work1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output='''+R+'''/logs/pilot-%j.out
#SBATCH --error='''+R+'''/logs/pilot-%j.err
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export LD_LIBRARY_PATH=/software/slurm/spackages/linux-rocky8-x86_64/gcc-12.3.0/cuda-12.3.0-p2hoh7xwcu52zilqglv3nnc5bwnritue/lib64:/home/aparame/.conda/envs/dynamicvla/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH='''+R+'''/scripts
cd '''+R+'''
'''+P+' '+R+'/run.py --stage '+R+' --root '+R+'/results --mode pilot\n'
 (H/'pilot.sbatch').write_text(body);remote('cat > '+R+'/pilot.sbatch',body.encode())
 pre=remote('bash -n '+R+'/pilot.sbatch && sbatch --test-only '+R+'/pilot.sbatch 2>&1');(H/'scheduler-preflight.txt').write_text(pre)
 state={'state':'intent','script':R+'/pilot.sbatch','script_sha256':hashlib.sha256(body.encode()).hexdigest(),'root':R,'checkpoint':T+'/runs/smolvla_baseline/latest.pt','trials':9,'gpu':'L40S','gpu_count':1,'max_hours':2};(H/'submission.json').write_text(json.dumps(state,indent=2))
 remote('cat > '+R+'/submission.json',json.dumps(state).encode())
 result=remote('sbatch --parsable '+R+'/pilot.sbatch').strip();job=result.split(';')[0];assert job.isdigit(),result
 state.update(state='submitted',job_id=job);(H/'submission.json').write_text(json.dumps(state,indent=2));remote('cat > '+R+'/submission.json',json.dumps(state).encode());print(json.dumps(state),flush=True)
if __name__=='__main__':main()
