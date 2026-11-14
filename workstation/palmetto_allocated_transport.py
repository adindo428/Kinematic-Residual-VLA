"""Transport commands to a verified Palmetto Slurm CPU step, never the login shell."""
import json
from pathlib import Path
import shlex
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]
ALLOCATION=REPO/'scratch/dom45/training/palmetto-completed-models-20261110/cpu_allocation.json'


def ssh_base():
    a=json.loads(ALLOCATION.read_text())
    return ['ssh','-S',a['socket'],'-o','BatchMode=yes','-o','ConnectTimeout=15',a['user']+'@'+a['host']]


def scheduler(command):
    # The only login-side operations are minimal scheduler/transport operations.
    assert command[0] in {'squeue','scontrol'}
    return subprocess.run([*ssh_base(),shlex.join(command)],capture_output=True,text=True,timeout=45,check=True).stdout


def allocated_command(args):
    a=json.loads(ALLOCATION.read_text());job=str(a['job_id'])
    guard=('test "${SLURM_JOB_ID:-}" = '+shlex.quote(job)+' && test -n "${SLURMD_NODENAME:-}" || exit 91; '
           'case "$(hostname -s)" in *login*) exit 92;; esac; exec "$@"')
    return [*ssh_base(),shlex.join(['srun','--jobid='+job,'--overlap','--exact','--nodes=1','--ntasks=1',
            '--cpus-per-task=1','--mem=1G','bash','-c',guard,'bash',*args])]


def run(args,input_text=None,timeout=300):
    return subprocess.run(allocated_command(args),input=input_text,capture_output=True,text=True,timeout=timeout,check=True).stdout


if __name__=='__main__':
    # rsync invokes its transport as HOST rsync --server ... . The remote rsync
    # executable itself runs inside the allocation, not on the login node.
    argv=sys.argv[1:]
    if argv[:1]==['-l']:
        assert argv[1]=='aparame';argv=argv[2:]
    assert len(argv)>1
    target,*args=argv
    assert target in {'aparame@slogin.palmetto.clemson.edu','slogin.palmetto.clemson.edu'}
    raise SystemExit(subprocess.call(allocated_command(args)))
