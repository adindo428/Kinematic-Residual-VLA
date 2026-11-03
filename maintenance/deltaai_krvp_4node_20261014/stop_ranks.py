"""Forward wall-limit signals only to training workers in this allocation."""
import os,signal,psutil
job=os.environ['SLURM_JOB_ID'];count=0
for p in psutil.process_iter(['pid','uids','cmdline']):
    try:
        if p.info['uids'].real!=os.getuid():continue
        command=p.info['cmdline'] or []
        # torchrun's command line also contains the worker entry path.
        # Signal its leaf workers; killing the launcher would abort checkpointing.
        if 'torch.distributed.run' in command:continue
        if not any(str(arg).endswith('/multi_entry.py') for arg in command):continue
        if p.environ().get('SLURM_JOB_ID')!=job:continue
        os.kill(p.pid,signal.SIGUSR1);count+=1
    except (psutil.Error,ProcessLookupError):pass
print('Training workers signalled:',count,flush=True)
