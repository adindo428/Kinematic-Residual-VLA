"""Resident cross-VLA workers using the previously evaluated adapter unchanged."""
import hashlib
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time

from resident_table1_worker import ResidentPool, session_loop
from smolvla_table1_client import Adapter


def heaviest_set(active, keys, receipts, capacity):
    others = sorted((k for k in keys if k != active), key=lambda k: receipts[k]['gpu_allocated'], reverse=True)
    return [active, *others[:capacity-1]]


def shared_worker_specs(specs):
    """Merge only explicitly opted-in, identical checkpoint implementations."""
    physical = []; aliases = {}; groups = {}
    for spec in specs:
        entry = spec['entry']
        identity = (entry['checkpoint_sha256'], entry['code_root'], entry['recipe'],
                    bool(spec.get('corrected_prompt_contract')), spec.get('gate_source'))
        canonical = groups.get(identity) if spec.get('share_checkpoint_worker') else None
        if canonical is None:
            canonical = dict(spec); physical.append(canonical)
            if spec.get('share_checkpoint_worker'): groups[identity] = canonical
        else:
            assert canonical['entry']['identity'] == entry['identity']
            canonical['grasp_gate'] = bool(canonical.get('grasp_gate') or spec.get('grasp_gate'))
        aliases[spec['key']] = canonical['key']
    return physical, aliases


def worker(spec, inbox, output, control, log_path):
    with open(log_path, 'a', buffering=1) as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        import torch
        from transformers import AutoTokenizer
        from completed_vla_table1_client import pinned_tokenizer
        from capture_table1_runtime import stats
        logging.basicConfig(level=logging.INFO, force=True)
        torch.set_num_threads(2)
        entry = spec['entry']; catalog = spec['catalog']
        sys.path.insert(0, entry['code_root']); sys.path.insert(1, catalog['vlash_source'])
        from policy_inference import Policy
        path = Path(entry['checkpoint']); stat = path.stat()
        assert (stat.st_size, stat.st_mtime_ns) == (entry['checkpoint_bytes'], entry['checkpoint_mtime_ns'])
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert saved['identity'] == entry['identity'] and saved['progress'] == entry['progress']
        recipe = json.loads(Path(entry['recipe']).read_text())
        refs = json.loads((Path(entry['code_root'])/'checkpoints.local.json').read_text())
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        with pinned_tokenizer(AutoTokenizer, refs['google/paligemma-3b-pt-224']['path']):
            model = Policy(entry['code_root'], recipe, 'cuda')
        model.load_state_dict(saved['model'], strict=True); del saved
        model.eval().requires_grad_(False)
        if spec.get('corrected_prompt_contract'):
            from adapter_grasp_gating import install_corrected_prediction
            install_corrected_prediction(model, gated=spec.get('grasp_gate', False))
        adapter = Adapter(model, recipe, torch)
        cfg = adapter.config
        cfg.adapter_grasp_gate = bool(spec.get('grasp_gate', False))
        cfg.adapter_gate_capable = cfg.adapter_grasp_gate
        cfg.adapter_gate_source = spec.get('gate_source')
        def dummy():
            obs = {k: torch.zeros(1, cfg.n_obs_steps, *f.shape) for k, f in cfg.input_features.items()}
            obs.update(task='Pick up the moving object.', index=0)
            obs['observation.environment_state'] = torch.zeros(1, cfg.n_obs_steps, 9)
            obs['observation.environment_state'][..., 6:9] = torch.tensor([.1, .02, -.01])
            if cfg.adapter_grasp_gate:
                obs['evaluation.phase_gate'] = torch.ones(1, 1, 20, 1)
            return obs
        def infer(obs, generation):
            if cfg.adapter_grasp_gate:
                gate = obs['evaluation.phase_gate'][:, -1].to(model.velocity_mean.device)
                assert gate.shape == (1, 20, 1) and torch.isfinite(gate).all()
                assert ((gate >= 0) & (gate <= 1)).all()
                model._evaluation_gate = gate
            return adapter.infer(obs, generation)
        infer(dummy(), 0)
        output.put({'initialized': True, 'config': cfg, 'gpu_allocated': torch.cuda.memory_allocated(),
                    'checkpoint_sha256': entry['checkpoint_sha256'], 'strict_checkpoint_loaded': True,
                    'corrected_prompt_contract': bool(spec.get('corrected_prompt_contract')),
                    'causal_grasp_gate': cfg.adapter_grasp_gate})
        handler = None
        def activate(path):
            nonlocal handler
            logger = logging.getLogger()
            if handler is not None: logger.removeHandler(handler); handler.close()
            handler = logging.FileHandler(path); logger.addHandler(handler)
        def compute(obs, generation):
            result = infer(obs, generation)
            logging.info('Adapter physical chunk %.4fs', result[3])
            return result
        def probe(command):
            # Reseeding makes device round trips comparable without changing scored inference.
            torch.manual_seed(42); torch.cuda.manual_seed_all(42)
            result = infer(dummy(), 0)[2]
            digest = hashlib.sha256(result.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            gate_proof = {}
            if cfg.adapter_grasp_gate:
                zero = dummy(); zero['evaluation.phase_gate'].zero_()
                torch.manual_seed(42); torch.cuda.manual_seed_all(42)
                zero_actions = infer(zero, 0)[2]
                from common import drift
                velocity = zero['observation.environment_state'][:, -1, 6:9]
                expected = drift(velocity, model.velocity_mean.cpu(), 20, torch.zeros(1, dtype=torch.long))
                torch.testing.assert_close(result-zero_actions, expected.transpose(0, 1), rtol=1e-4, atol=1e-5)
                gate_proof = {'same_noise_gate_zero_only_removes_centered_xyz': True}
            for _ in range(3): infer(dummy(), 0)
            samples = []
            for _ in range(command.get('samples', 20)):
                tick = time.perf_counter(); infer(dummy(), 0)
                samples.append(1000*(time.perf_counter()-tick))
            return {'passed': True, 'action_sha256': digest, 'measurements_ms': samples,
                    **gate_proof, **stats(samples), 'gpu_allocated': torch.cuda.memory_allocated(),
                    'gpu_reserved': torch.cuda.memory_reserved()}
        def place(device):
            torch.cuda.synchronize(); model.to(device)
            if device == 'cpu': torch.cuda.empty_cache()
            else: torch.cuda.synchronize()
            return {'ready': True, 'device': device, 'gpu_allocated': torch.cuda.memory_allocated()}
        try: session_loop(inbox, output, control, compute, activate, probe, place)
        finally: adapter.executor.shutdown(wait=True)


class PendingChunk:
    """A synchronous manager queue avoids executor/idle acknowledgement races."""
    def __init__(self, output): self.output = output; self.value = None
    def done(self):
        if self.value is None:
            try: self.value = self.output.get_nowait()
            except queue.Empty: return False
        return True
    def result(self):
        assert self.value is not None
        return self.value


class Submitter:
    def __init__(self, front): self.front = front
    def submit(self, fn, obs, generation):
        if getattr(self.front.config, 'adapter_gate_capable', False) and not self.front.config.adapter_grasp_gate:
            # Ungated and gated variants share the KR network. The ungated
            # variant explicitly requests a unit gate for every chunk.
            import torch
            obs = {**obs, 'evaluation.phase_gate': torch.ones(1, 1, 20, 1)}
        self.front.inbox['obs'] = (obs, generation)
        return PendingChunk(self.front.output)


class Front(Adapter):
    def __init__(self, cfg, inbox, output, control):
        self.config = cfg; self.inbox = inbox; self.output = output; self.control = control
        self.future = None; self.generation = 0; self.log = None
        self.executor = Submitter(self); Adapter.reset(self)
    def command(self, kind, **kw):
        self.control.send({'kind': kind, **kw})
        if not self.control.poll(300): raise TimeoutError('Adapter worker control timeout')
        return self.control.recv()
    def idle(self): return self.command('idle')
    def reset(self):
        self.idle(); self.future = None; self._adapter_gate_tracker = None; Adapter.reset(self)
        self.command('activate', log=self.log)
    def infer(self, *args): raise AssertionError('GPU computation belongs to the worker')


class Pool(ResidentPool):
    def __init__(self, specs, directory):
        self.variant_gates = {s['key']: bool(s.get('grasp_gate')) for s in specs}
        specs, self.variant_to_worker = shared_worker_specs(specs)
        self.ctx = mp.get_context('spawn'); self.manager = self.ctx.Manager()
        self.fronts = {}; self.processes = []; self.receipts = {}; self.devices = {}; self.lru = []
        self.capacity = 1; self.profile = {}; directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        try:
            # Initialize one GPU model at a time, parking it before loading the next.
            # This permits a smaller resident set if four models exceed capacity.
            for spec in specs:
                inbox = self.manager.dict(); output = self.manager.Queue(maxsize=1); parent, child = self.ctx.Pipe()
                proc = self.ctx.Process(target=worker, args=(spec, inbox, output, child, str(directory/(spec['key']+'.log'))), daemon=True)
                proc.start(); self.processes.append(proc)
                deadline = time.monotonic()+1200
                while True:
                    try: initialized = output.get(timeout=1); break
                    except queue.Empty:
                        if not proc.is_alive(): raise RuntimeError('Resident worker failed to initialize: '+spec['key'])
                        if time.monotonic() > deadline: raise TimeoutError('Resident worker initialization timed out')
                assert initialized['initialized']
                cfg = initialized.pop('config')
                front = Front(cfg, inbox, output, parent); key = spec['key']
                self.fronts[key] = front; self.receipts[key] = initialized
                self.devices[key] = 'cuda'; self.lru.append(key)
                front.command('place', device='cpu'); self.devices[key] = 'cpu'
                (directory/'loading.json').write_text(json.dumps({'loaded': list(self.fronts), 'target': len(specs), 'receipts': self.receipts}, indent=2)+'\n')
        except BaseException:
            self.close(); raise

    def select(self, key):
        front = super().select(self.variant_to_worker[key])
        front.config.adapter_grasp_gate = self.variant_gates[key]
        return front

    def qualify_all(self, destination):
        from capture_table1_runtime import stats
        import subprocess
        trials = []; keys = list(self.fronts); start_capacity = len(keys)
        prior = Path(destination).parent/'failed_four_model_qualification.json'
        if prior.exists():
            previous = json.loads(prior.read_text())
            assert not previous['passed'] and previous['gpu_residency_capacity'] == len(keys)
            for key in keys:
                assert previous['worker_receipts'][key]['checkpoint_sha256'] == self.receipts[key]['checkpoint_sha256']
            trials.extend(previous['paired_trials']); start_capacity -= 1
        for capacity in range(start_capacity, 0, -1):
            # Large idle models require no compute; nevertheless measure each backbone.
            model_proofs = {}; ok = True
            for key in keys:
                groups = {'single': [], 'candidate': []}; hashes = set()
                for mode in ['single', 'candidate', 'candidate', 'single']:
                    self.idle()
                    wanted = [key] if mode == 'single' else heaviest_set(key, keys, self.receipts, capacity)
                    # Evict before loading, so a failed four-model attempt can safely fall back.
                    for k in keys:
                        if k not in wanted and self.devices[k] != 'cpu':
                            self.fronts[k].command('place', device='cpu'); self.devices[k] = 'cpu'
                    for k in wanted:
                        if self.devices[k] != 'cuda':
                            self.fronts[k].command('place', device='cuda'); self.devices[k] = 'cuda'
                    p = self.fronts[key].command('probe', samples=20)
                    hashes.add(p['action_sha256']); groups[mode] += p['measurements_ms']
                assert len(hashes) == 1, 'Model placement changed seeded actions'
                a, b = stats(groups['single']), stats(groups['candidate'])
                if capacity == 1:
                    # There is no residency comparison at capacity one. Avoid
                    # treating noise between identical isolated samples as contention.
                    a = b = stats(groups['single'] + groups['candidate'])
                passed = b['p50_ms'] <= a['p50_ms']/.9 and b['p95_ms'] <= a['p95_ms']/.9
                model_proofs[key] = {'single': a, 'candidate': b, 'passed': passed,
                                     'cpu_gpu_roundtrip_actions_exact': True, 'action_sha256': next(iter(hashes)),
                                     'tested_resident_set': heaviest_set(key, keys, self.receipts, capacity)}
                ok &= passed
                (Path(destination)/'qualification_progress.json').write_text(json.dumps({'capacity': capacity, 'models': model_proofs}, indent=2)+'\n')
            self.capacity = capacity
            # Keep the qualified number loaded at return; the LRU swaps only outside episodes.
            self.set_capacity(capacity)
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True)
            gpu = next([int(x.strip()) for x in line.split(',')] for line in memory.splitlines() if int(line.split(',')[0]) == 1)
            headroom = gpu[1] < .9*gpu[2]
            trials.append({'capacity': capacity, 'models': model_proofs, 'passed': bool(ok and headroom),
                           'driver_memory_mib': {'used': gpu[1], 'total': gpu[2]}})
            self.profile = {'passed': bool(ok and headroom), 'gpu_residency_capacity': capacity,
                            'minimum_latency_retention': .9, 'paired_trials': trials, 'worker_receipts': self.receipts,
                            'physical_networks': len(keys), 'variant_to_worker': self.variant_to_worker}
            if ok and headroom:
                for variant, physical in self.variant_to_worker.items():
                    self.profile['paired_trials'][-1]['models'][variant] = model_proofs[physical]
            (Path(destination)/'resident_profile.json').write_text(json.dumps(self.profile, indent=2)+'\n')
            if ok and headroom: return self.profile
        raise RuntimeError('Even isolated adapter qualification failed')
