"""Runtime tuning changes execution, never the fixed recipe or data order."""
import os
import threading
import time
import subprocess
import resource

import torch
import torch.distributed as dist


def validate_h200(name, memory, world):
    if world != 2 or 'H200' not in name or memory < 120 * 1024**3:
        raise ValueError('This fresh attempt requires exactly two actual H200 GPUs with at least120GiB each')


def synchronized_flags(stop, checkpoint_due, device):
    flags = torch.tensor([int(stop),int(checkpoint_due)],device=device)
    dist.all_reduce(flags,op=dist.ReduceOp.MAX)
    return bool(flags[0]), bool(flags[1])


def validate_runtime(runtime, global_batch, world):
    micro = runtime['microbatch']
    if micro < 1 or global_batch % (micro*world):
        raise ValueError('Microbatch must exactly divide the unchanged effective batch')
    if not 0 <= runtime['workers'] <= 12 or runtime['main_threads'] < 1:
        raise ValueError('Invalid bounded CPU runtime')
    if world*(runtime['workers']+runtime['main_threads']) > int(os.environ.get('SLURM_CPUS_PER_TASK','32')):
        raise ValueError('Runtime exceeds the CPU allocation')


class Resources:
    """Conservative summed process/worker RSS, and sampled GPU activity."""
    def __init__(self, gpu=False):
        self.peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        self.util = []
        self.gpu = gpu
        self.started = time.monotonic()
        self.cpu_seconds = 0.
        self.cpu_elapsed = 0.
        self.cpu_last = {}
        self.cpu_tick = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._sample,daemon=True)

    def _sample(self):
        import psutil
        process = psutil.Process()
        tick = 0
        while not self.done.is_set():
            try:
                children = [p for p in [process,*process.children(recursive=True)] if p.is_running()]
                rss = sum(p.memory_info().rss for p in children)
                self.peak_rss = max(self.peak_rss,rss)
                now = time.monotonic()
                current = {}
                for p in children:
                    cpu = p.cpu_times()
                    current[(p.pid,p.create_time())] = cpu.user + cpu.system
                if self.cpu_tick is not None:
                    self.cpu_seconds += sum(max(0.,v-self.cpu_last.get(k,v)) for k,v in current.items())
                    self.cpu_elapsed += now-self.cpu_tick
                self.cpu_last,self.cpu_tick = current,now
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
            if self.gpu and tick%4==0:
                try:
                    # Sample only this allocation's visible device IDs.
                    visible = os.environ.get('CUDA_VISIBLE_DEVICES','0,1')
                    result = subprocess.run(['nvidia-smi','--id='+visible,
                        '--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],
                        text=True,capture_output=True,timeout=2)
                    if result.returncode==0:self.util.extend(float(v) for v in result.stdout.split())
                except (ValueError,subprocess.SubprocessError):
                    pass
            tick += 1
            self.done.wait(.5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self,*_):
        self.done.set()
        self.thread.join(timeout=3)

    @property
    def mean_cpu_busy_cores(self):
        return self.cpu_seconds/self.cpu_elapsed if self.cpu_elapsed else None

    @property
    def mean_gpu_utilization(self):
        return sum(self.util)/len(self.util) if self.util else None
