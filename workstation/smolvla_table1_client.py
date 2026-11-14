#!/usr/bin/env python3
"""Run pinned SmolVLA inference with an explicitly batched RPC instruction."""
import importlib.util
import argparse
import json
import logging
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
FROZEN = REPO / 'scratch/dom45/evaluations/multivla20k-20261029/workstation'
sys.path.insert(0, str(FROZEN))
spec = importlib.util.spec_from_file_location('frozen_multivla_client', FROZEN / 'multivla_table1_client.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


def batch_instruction(task, batch_size):
    tasks = [task] if isinstance(task, str) else task
    if not isinstance(tasks, (tuple, list)) or len(tasks) != batch_size or not all(isinstance(t, str) for t in tasks):
        raise ValueError('RPC instruction batch does not match the observation batch')
    return list(tasks)


class Adapter(client.Adapter):
    def infer(self, obs, generation):
        # The RPC supplies one string, while training/tokenization expect one
        # string per environment. Otherwise the wrapper tokenizes characters
        # as separate examples and concatenates incompatible batch dimensions.
        obs = {**obs, 'task': batch_instruction(obs['task'], obs['observation.state'].shape[0])}
        return super().infer(obs, generation)


def install_waiting_rpc(rpc):
    """Wait in libzmq between observations so the model worker can run Python."""
    original = rpc.get_latest_observation

    def latest(socket):
        if not socket.poll(timeout=10):
            return None
        return original(socket)

    rpc.get_latest_observation = latest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=27000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    entry = catalog['models'][args.model]
    import torch
    torch.set_num_threads(2)
    if catalog['expected_gpu'] not in torch.cuda.get_device_name(0):
        raise ValueError('Unexpected inference GPU')
    if entry['method'] != 'smolvla':
        raise ValueError('This client is scoped to the two SmolVLA checkpoints')
    sys.path.insert(0, entry['code_root'])
    from policy_inference import Policy
    recipe = json.loads(Path(entry['recipe']).read_text())
    checkpoint = Path(entry['checkpoint'])
    stat = checkpoint.stat()
    if stat.st_size != entry['checkpoint_bytes'] or stat.st_mtime_ns != entry['checkpoint_mtime_ns']:
        raise ValueError('Pinned checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['identity'] != entry['identity'] or saved['progress'] != entry['progress']:
        raise ValueError('Checkpoint identity/progress changed')
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = Policy(entry['code_root'], recipe, 'cuda')
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().requires_grad_(False)
    adapter = Adapter(model, recipe, torch)
    dummy = {k: torch.zeros((1, adapter.config.n_obs_steps, *v.shape)) for k, v in adapter.config.input_features.items()}
    dummy.update(task='Pick up the moving object.', **{'observation.environment_state': torch.zeros(1, 1, 9)}, index=0, dt_scale=1.)
    _, _, prediction, elapsed = adapter.infer(dummy, adapter.generation)
    if args.probe:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'policy_probe.json').write_text(json.dumps({
            'strict_checkpoint_loaded': True, 'model': args.model, 'progress': entry['progress'],
            'identity': entry['identity'], 'checkpoint_sha256': entry['checkpoint_sha256'],
            'finite_physical_shape': list(prediction.shape), 'gpu': torch.cuda.get_device_name(0),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'probe_chunk_s': elapsed,
            'rpc_wait': 'libzmq poll, timeout 10 ms; no busy spin'}, indent=2) + '\n')
        adapter.executor.shutdown()
        return
    source = Path(catalog['official_source'])
    sys.path.append(str(source))
    spec = importlib.util.spec_from_file_location('official_rpc', source / 'scripts/inference.py')
    rpc = importlib.util.module_from_spec(spec)
    rpc_text = (source / 'scripts/inference.py').read_text().replace('import utils.helpers', 'import rpc_geometry as helpers').replace('utils.helpers.', 'helpers.')
    exec(compile(rpc_text, str(source / 'scripts/inference.py'), 'exec'), rpc.__dict__)
    transform = rpc._get_transformed_observation

    def with_object_state(observation, rotation, features, device='cpu'):
        import numpy as np
        value = transform(observation, rotation, features, device)
        obj = observation['observation.environment_state']['object']
        environment = np.concatenate([obj['pos'], rpc.helpers.get_rotation_vector(obj['quat'], rotation), obj['velocity']], axis=-1).astype(np.float32)
        value['observation.environment_state'] = torch.from_numpy(environment).to(device)
        return value

    rpc._get_transformed_observation = with_object_state
    install_waiting_rpc(rpc)
    sockets = None
    start = time.monotonic()
    while sockets is None:
        try:
            sockets = rpc.get_zmq_sockets('127.0.0.1', args.port, args.port + 1)
        except RuntimeError:
            if time.monotonic() - start > 900:
                raise
            time.sleep(5)
    sockets[1].send_pyobj({'vla': args.model, 'epoch': entry['progress']['epoch']})
    try:
        n, results = rpc.run_tests(*sockets, {'delta_timestamps': [0]}, adapter, str(checkpoint),
                                  entry['progress']['epoch'], 'euler', True, str(args.output))
        logging.info('Evaluation completed. Total tests run: %d', n)
    finally:
        adapter.executor.shutdown(wait=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s')
    main()
