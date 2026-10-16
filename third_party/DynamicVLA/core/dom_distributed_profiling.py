"""Measure real two-GPU training updates and distinguish loader/compute costs."""
import contextlib
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time

import psutil
import pynvml
import torch
import torch.distributed as dist

from core.dom_distributed import leader_call
from core.dom_distributed_training import distributed_loader, updates, wrap_policy
from core.dom_training import build_dataset, build_policy, load_family, make_optimizer, make_scheduler, seed_all, to_device
from utils.dom_experiment import atomic_json, digest


class Telemetry:
    def __init__(self, local_rank):
        pynvml.nvmlInit()
        mask = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
        physical = mask[local_rank]
        self.handle = (pynvml.nvmlDeviceGetHandleByIndex(int(physical)) if physical.isdigit()
                       else pynvml.nvmlDeviceGetHandleByUUID(physical))
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.rows, self.peak_ram, self.peak_gpu, self.errors = [], 0, 0, []
        self.processes = {os.getpid():psutil.Process()}
        self.thread = threading.Thread(target=self.collect, daemon=True)

    def collect(self):
        parent = psutil.Process()
        while not self.stop.is_set():
            try:
                processes = [parent, *parent.children(recursive=True)]
                ram, cpu = 0, 0.
                for process in processes:
                    try:
                        stable = self.processes.setdefault(process.pid, process)
                        ram += stable.memory_info().rss
                        cpu += stable.cpu_percent()
                    except psutil.Error:
                        pass
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                row = {'time_s':time.time(), 'gpu_percent':util.gpu,
                       'gpu_memory_percent':util.memory, 'gpu_used_bytes':memory.used,
                       'ram_bytes':ram, 'cpu_percent':cpu,
                       'power_w':pynvml.nvmlDeviceGetPowerUsage(self.handle)/1000}
                with self.lock:
                    self.rows.append(row)
                    self.peak_ram = max(self.peak_ram, ram)
                    self.peak_gpu = max(self.peak_gpu, memory.used)
            except (psutil.Error, pynvml.NVMLError) as error:
                self.errors.append(type(error).__name__)
            self.stop.wait(.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)

    def summary(self, start):
        with self.lock:
            rows = [row for row in self.rows if row['time_s'] >= start]
            if len(rows) < 3:
                raise RuntimeError('Insufficient GPU/CPU telemetry for performance approval')
            return {'samples':len(rows), 'mean_gpu_utilization_percent':statistics.mean(row['gpu_percent'] for row in rows),
                    'mean_memory_utilization_percent':statistics.mean(row['gpu_memory_percent'] for row in rows),
                    'mean_cpu_percent':statistics.mean(row['cpu_percent'] for row in rows),
                    'mean_power_w':statistics.mean(row['power_w'] for row in rows),
                    'peak_gpu_bytes':self.peak_gpu, 'peak_ram_bytes':self.peak_ram,
                    'telemetry_errors':list(set(self.errors))}


def gpu_batches(template, microbatch, world, count):
    """Reuse an actual decoded CUDA batch only for the compute-capacity sweep."""
    local = 352//world
    for _ in range(count):
        for offset in range(0, local, microbatch):
            size = min(microbatch, local-offset)
            batch = {key:(value[:size] if isinstance(value, (list, tuple)) or
                         torch.is_tensor(value) and value.ndim else value) for key,value in template.items()}
            device = batch['action'].device
            batch.update({'ddp.valid':torch.ones(size, dtype=torch.int64, device=device),
                          'ddp.update_samples':torch.full((size,),352,dtype=torch.int64,device=device),
                          'ddp.update_end':torch.full((size,),int(offset+size==local),dtype=torch.int64,device=device)})
            yield batch


def profile_trial(root, runtime, mode, device, output, warmup=2, measured_updates=5, trace=False, epoch=0):
    root, output = Path(root), Path(output)
    manifest = load_family(root)
    world, rank = dist.get_world_size(), dist.get_rank()
    if warmup < 1 or measured_updates < 3:
        raise ValueError('Use warmup and at least three measured optimizer updates')
    seed_all()
    properties = torch.cuda.get_device_properties(device)
    hardware = {'name':properties.name,'total_memory':properties.total_memory,
                'torch':torch.__version__,'cuda':torch.version.cuda,'capability':[properties.major,properties.minor]}
    torch.cuda.reset_peak_memory_stats(device)
    with Telemetry(device.index) as telemetry:
        # Profile the actual complete training split and uniform sampler, not a
        # small episode set whose cache hit rate could overstate throughput.
        split = json.loads((root/'split.json').read_text())['splits']['train']
        data = build_dataset(root, split, training=True)
        loader = distributed_loader(data, runtime, epoch=epoch, sampling='frames')
        if len(data) != manifest['train_frames']:
            raise ValueError('Profiling dataset differs from the frozen training split')
        if mode == 'gpu_only':
            template = to_device(next(iter(loader)), device)
            loader = gpu_batches(template, runtime['microbatch'], world, warmup+measured_updates)
        policy = build_policy(root, 'krv100', device)
        model = wrap_policy(policy, device)
        optimizer = make_optimizer(policy, 3e-5, runtime['fused'])
        scheduler = make_scheduler(optimizer, 10000, warmup=0)
        optimizer.zero_grad(set_to_none=True)
        # Match fresh production's rank-specific flow-matching noise.
        torch.manual_seed(42+rank)
        torch.cuda.manual_seed(42+rank)
        progress = {'epoch':epoch,'cursor':0,'updates':0,'samples':0}
        iterator = iter(updates(model,optimizer,scheduler,loader,progress,runtime['precision'],device))
        latencies, waits, losses = [], [], []
        trace_dir = output.parent/'traces'
        profiler = contextlib.nullcontext()
        if trace:
            trace_dir.mkdir(parents=True, exist_ok=True)
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0,warmup=warmup,active=1,repeat=1),
                on_trace_ready=lambda profile:profile.export_chrome_trace(str(trace_dir/f'rank-{rank}.trace.json')),
                record_shapes=True,profile_memory=True,with_stack=False)
        start_measurement = None
        with profiler as timeline:
            for index in range(warmup+measured_updates):
                torch.cuda.synchronize(device)
                if index == warmup:
                    dist.barrier()
                    start_measurement = time.time()
                begin = time.perf_counter()
                record = next(iterator)
                torch.cuda.synchronize(device)
                elapsed = torch.tensor(time.perf_counter()-begin,device=device,dtype=torch.float64)
                dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
                if record['samples'] != 352 or not math.isfinite(record['loss']):
                    raise ValueError('Invalid full-batch profiling update')
                if index >= warmup:
                    latencies.append(float(elapsed))
                    waits.append(record['data_wait_s'])
                    losses.append(record['loss'])
                if trace:
                    timeline.step()
        measurements = telemetry.summary(start_measurement)
        measurements['peak_gpu_bytes'] = max(measurements['peak_gpu_bytes'],torch.cuda.max_memory_reserved(device))
        host_limit = int(os.environ['SLURM_MEM_PER_NODE'])*1024**2
        measurement = {'rank':rank,'hardware':hardware,'telemetry':measurements,
                       'data_wait_fraction':sum(waits)/sum(latencies), 'losses':losses}
        ranks = [None]*world
        dist.all_gather_object(ranks, measurement)
        safe = (all(row['telemetry']['peak_gpu_bytes'] <= .85*row['hardware']['total_memory'] for row in ranks)
                and sum(row['telemetry']['peak_ram_bytes'] for row in ranks) <= .85*host_limit)
        result = {'complete':True,'safe':safe,'manifest':manifest['fingerprint'],'variant':'krv100',
                  'world_size':world,'runtime':runtime,'mode':mode,'trace':trace,
                  'sampling_epoch':epoch,
                  'warmup_updates':warmup,'measured_updates':measured_updates,
                  'global_effective_batch':352,'samples_per_second':352*measured_updates/sum(latencies),
                  'median_update_latency_s':statistics.median(latencies),'update_latencies_s':latencies,
                  'data_wait_fraction':max(row['data_wait_fraction'] for row in ranks),
                  'hardware_by_rank':[row['hardware'] for row in ranks], 'ranks':ranks,
                  'host_limit_bytes':host_limit,'scope':'Warm optimizer updates; excludes validation and checkpoint I/O'}
        result['fingerprint'] = digest(result)
        leader_call(lambda:atomic_json(output,result))
        return result
