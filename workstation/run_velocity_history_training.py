"""Local distance-gated KRV: bounded profiling, strict readiness, then production."""
import argparse
from datetime import timedelta, datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'scratch/dom45/training/krv-distance-gate-history-20261113'
sys.path.insert(0, str(STAGE/'source'))

def worker(args):
    import torch
    import torch.distributed as dist
    import psutil
    # Local compatibility for the shared profiler; no scheduler allocation is implied.
    os.environ.setdefault("SLURM_MEM_PER_NODE", str(psutil.virtual_memory().total // 1024**2))
    from core.dom_distributed import initialize_control_group, leader_call
    from core.dom_training import load_family
    from utils.dom_experiment import atomic_json, digest
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    local = int(os.environ['LOCAL_RANK'])
    torch.set_num_threads(2); torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30)); initialize_control_group()
    family = STAGE/'family'; manifest = load_family(family)
    runtime = json.loads(args.runtime) if args.runtime else json.loads((STAGE/'selected_runtime.json').read_text())['runtime']
    stop = threading.Event()
    def monitor():
        process = psutil.Process(); peak = 0
        while not stop.wait(1):
            try:
                rss = process.memory_info().rss + sum(c.memory_info().rss for c in process.children(recursive=True))
                peak = max(peak, rss); os.environ['KRV_PEAK_RSS'] = str(peak)
                if args.mode == 'train' and (rss > 95*1024**3 or psutil.virtual_memory().available < 20*1024**3):
                    logging.error('Host memory guard requesting checkpointed stop')
                    os.kill(os.getpid(), signal.SIGUSR1); return
            except psutil.Error: pass
    thread = threading.Thread(target=monitor, daemon=True)
    # Train signals are installed only after dataset/model construction.
    if args.mode != 'train': thread.start()
    try:
        if args.mode == 'profile':
            from core.dom_distributed_profiling import profile_trial
            profile_trial(family, runtime, 'end_to_end', device, args.output, warmup=2, measured_updates=5)
        elif args.mode == 'smoke':
            from core.dom_distributed_training import distributed_smoke
            report = distributed_smoke(family, 'krv100', runtime, device)
            leader_call(lambda: atomic_json(STAGE/'readiness.json', report))
        else:
            from core import dom_distributed_training as training
            readiness = json.loads((STAGE/'readiness.json').read_text())
            assert readiness['safe'] and readiness['strict_reload'] and readiness['runtime'] == runtime
            assert readiness['manifest'] == manifest['fingerprint']
            import wandb
            if not callable(getattr(wandb.util, 'generate_id', None)):
                from wandb.sdk.lib.runid import generate_id
                wandb.util.generate_id = generate_id
            original = training.updates
            def monitored_updates(*pos, **kw):
                if not thread.is_alive() and not stop.is_set(): thread.start()
                for record in original(*pos, **kw):
                    free, total = torch.cuda.mem_get_info(device)
                    over = torch.tensor(int(free < .10*total), device=device)
                    dist.all_reduce(over, op=dist.ReduceOp.MAX)
                    if int(over): os.kill(os.getpid(), signal.SIGUSR1)
                    yield record
            training.updates = monitored_updates
            recipe = json.loads((family/'recipe.json').read_text())
            body = dict(recipe); claimed = body.pop('fingerprint'); assert digest(body) == claimed
            result = training.train_distributed(family, 'krv100', runtime, recipe, device)
            leader_call(lambda: atomic_json(STAGE/'training_result.json', result))
            return 75 if result.get('interrupted') else 0
    finally:
        stop.set()
        if thread.ident: thread.join(timeout=2)
        dist.destroy_process_group()
    return 0

def coordinate():
    from utils.dom_experiment import atomic_json
    plan = json.loads((STAGE/'plan.json').read_text())
    def status(phase, **kw):
        atomic_json(STAGE/'status.json', dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **kw))
    lock = open(ROOT/'scratch/dom45/evaluations/low-clutter-full-table1-20261102/queue.lock', 'a')
    status('waiting_for_local_gpu_lock')
    fcntl.flock(lock, fcntl.LOCK_EX)
    busy = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
    if busy: raise RuntimeError('Unrelated GPU processes present; will not interrupt them')
    base = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(Path(__file__).resolve())]
    def execute(mode, runtime=None, output=None, label=None):
        cmd = base+['--mode',mode]
        if runtime: cmd += ['--runtime',json.dumps(runtime)]
        if output: cmd += ['--output',str(output)]
        with (STAGE/f'{label or mode}.log').open('a') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(STAGE/'child.json', dict(pid=proc.pid, mode=mode))
            return proc.wait()
    try:
        profiles = STAGE/'profiling'; profiles.mkdir(exist_ok=True)
        selected = STAGE/'selected_runtime.json'
        if not selected.exists():
            measured = []
            for runtime in plan['candidates']:
                label = f'profile-m{runtime["microbatch"]}-w{runtime["workers"]}'
                output = profiles/f'{label}.json'
                status('profiling', runtime=runtime)
                code = execute('profile', runtime, output, label)
                if code:
                    if 'out of memory' in (STAGE/f'{label}.log').read_text().lower(): continue
                    raise RuntimeError(f'{label} failed; see log')
                result = json.loads(output.read_text())
                if result['safe']: measured.append(result)
            if not measured: raise RuntimeError('No measured runtime meets memory limits')
            winner = max(measured, key=lambda p:p['samples_per_second'])
            if winner['data_wait_fraction'] > .05:
                runtime = {**winner['runtime'], 'workers':12}
                label = f'profile-m{runtime["microbatch"]}-w12'
                output = profiles/f'{label}.json'
                status('profiling', runtime=runtime)
                if execute('profile', runtime, output, label):
                    raise RuntimeError(f'{label} failed; see log')
                result = json.loads(output.read_text())
                if result['safe']:
                    measured.append(result)
                    winner = max(measured, key=lambda p:p['samples_per_second'])
            atomic_json(selected, dict(runtime=winner['runtime'], samples_per_second=winner['samples_per_second'],
                measured_profiles=[p['runtime'] for p in measured], manifest=plan['manifest'],
                scope='Fastest of bounded local candidates; short full-data calibration, not sustained stability proof'))
        runtime = json.loads(selected.read_text())['runtime']
        status('calibrating_history', runtime=runtime)
        output = profiles/'history-calibration.json'
        if execute('profile', runtime, output, 'history-calibration'):
            raise RuntimeError('History calibration failed')
        if not json.loads(output.read_text())['safe']:
            raise RuntimeError('History calibration exceeded memory limits')
        status('readiness', runtime=json.loads(selected.read_text())['runtime'])
        if execute('smoke'): raise RuntimeError('Readiness failed; production not launched')
        status('training', runtime=json.loads(selected.read_text())['runtime'])
        code = execute('train')
        status('complete' if code == 0 else 'stopped', exit_code=code)
    except BaseException as exc:
        status('failed', error=str(exc)); raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', default='coordinate', choices=['coordinate','profile','smoke','train'])
    parser.add_argument('--runtime'); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    sys.exit(coordinate() if args.mode == 'coordinate' else worker(args))
