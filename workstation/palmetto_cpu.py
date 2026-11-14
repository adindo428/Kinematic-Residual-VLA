"""Transport-only login hop; execute the whole script in an existing CPU job."""
import argparse
from pathlib import Path
import shlex
import subprocess


def command(script,*,node='node1347',job='20145158',socket='/home/ailab/.ssh/palmetto-slogin.sock',cpus=1):
    if len(script.encode())>100000:raise ValueError('Control scripts only; use a DTN for data transfers')
    if not job.isdigit() or not node.startswith('node') or not node[4:].isdigit():raise ValueError('Explicit CPU node and numeric existing job required')
    if not 1<=cpus<=8:raise ValueError('CPU step exceeds the verified 8-core allocation')
    guard='set -eu\nexec </dev/null\n[ "$(hostname -s)" = '+shlex.quote(node)+' ] || exit 90\n[ "${SLURM_JOB_ID:-}" = '+shlex.quote(job)+' ] || exit 91\n'
    step=['srun','--jobid='+job,'--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task='+str(cpus),'--nodelist='+node,'bash','-lc',guard+script]
    return ['ssh','-S',str(Path(socket).expanduser()),'-o','ControlMaster=no','-o','BatchMode=yes','-T',
        'aparame@slogin.palmetto.clemson.edu','exec '+shlex.join(step)]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--script',type=Path,required=True)
    parser.add_argument('--node',default='node1347');parser.add_argument('--job',default='20145158');parser.add_argument('--socket',default='/home/ailab/.ssh/palmetto-slogin.sock')
    parser.add_argument('--cpus',type=int,default=1);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    cmd=command(args.script.read_text(),node=args.node,job=args.job,socket=args.socket,cpus=args.cpus)
    if args.check:print(shlex.join(cmd))
    else:raise SystemExit(subprocess.call(cmd,stdin=subprocess.DEVNULL))
