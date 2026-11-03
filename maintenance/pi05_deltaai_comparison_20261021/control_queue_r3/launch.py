"""One reconciled queued interactive request using unchanged benchmark code."""
from pathlib import Path
import subprocess,json,re,datetime,hashlib
HERE=Path(__file__).resolve().parent;C=json.loads((HERE/'config.json').read_text());ROOT=Path(C['root'])
def atomic(name,obj):
 p=HERE/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2)+'\n');t.replace(p)
def main():
 assert not (HERE/'requested.json').exists(),'Reconcile existing request; never duplicate'
 user=__import__('os').environ['USER']
 queued=subprocess.check_output(['squeue','-h','-u',user,'-o','%i'],text=True).split()
 assert C['previous_job'] not in queued,'Previous diagnostic request remains active'
 accounting=subprocess.check_output(['sacct','-j',C['previous_job'],'-X','-nP','-o','JobID,State,ExitCode'],text=True)
 assert C['previous_job']+'|CANCELLED' in accounting,'Previous allocation outcome uncertain'
 assert not subprocess.check_output(['squeue','-h','-u',__import__('os').environ['USER'],'-n',C['job_name'],'-o','%i'],text=True).strip(),'Existing duplicate request'
 assert hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()==C['manifest_sha256']
 assert hashlib.sha256((ROOT/'sweep.sh').read_bytes()).hexdigest()==C['sweep_sha256']
 assert (ROOT/'checkpoint.pt').stat().st_size==40456772509
 assert not (ROOT/'control.log').exists(),'Unexpected prior GPU execution; reconcile'
 request={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supersedes_definitively_timed_out_request':C['previous_job'],'partition':'ghx4-interactive','gpus':2,'cpus':32,'memory_gib':256,'allocation_limit_hours':2,'queue_wait_limit_minutes':None,'production_changed':False}
 with (HERE/'requested.json').open('x') as f:json.dump(request,f,indent=2)
 command=['salloc','-A','bivv-dtai-gh','-p','ghx4-interactive','--nodes=1','--ntasks=1','--gpus-per-node=2','--cpus-per-task=32','--mem=256G','--time=02:00:00','--job-name='+C['job_name'],'bash','-c',f'cd {ROOT}; printf "%s\\n" "$SLURM_JOB_ID" > allocation-id; exec srun --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus-per-task=2 --cpu-bind=cores bash sweep.sh']
 with (HERE/'allocation.log').open('x') as log:
  process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  for line in process.stdout:
   log.write(line);log.flush()
   match=re.search(r'(?:Pending job allocation|Granted job allocation) (\d+)',line)
   if match:
    request.update(job_id=match.group(1),last_scheduler_line=line.strip());atomic('submission.json',request)
  request['controller_exit_code']=process.wait();request['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic('controller-complete.json',request)
if __name__=='__main__':main()
