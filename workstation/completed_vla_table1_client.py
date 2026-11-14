#!/usr/bin/env python3
"""Evaluate transferred PI05/VLASH snapshots using pinned local tokenizer files."""
import argparse
from contextlib import contextmanager
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch

from smolvla_table1_client import Adapter, install_waiting_rpc
from evaluation_training_policy import require_eligible


@contextmanager
def pinned_tokenizer(tokenizer_class, local_path):
    """Resolve VLASH's internal tokenizer ID to its existing frozen snapshot."""
    original = tokenizer_class.from_pretrained

    def load(name, *args, **kwargs):
        if str(name) == 'google/paligemma-3b-pt-224':
            name = str(local_path)
            kwargs['local_files_only'] = True
        return original(name, *args, **kwargs)

    with patch.object(tokenizer_class, 'from_pretrained', side_effect=load):
        yield


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
    require_eligible(args.model, entry)
    if args.model not in ('vlash_kr', 'pi05_baseline'):
        raise ValueError('Incomplete VLASH baseline and PI05+KR are held out of evaluation')
    if entry['progress']['epoch'] < 10 or entry['progress']['cursor'] != 0:
        raise ValueError('Expected the completed epoch-10 checkpoint')
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    if catalog['expected_gpu'] not in torch.cuda.get_device_name(0):
        raise ValueError('Unexpected inference GPU')
    sys.path.insert(0, entry['code_root'])
    sys.path.insert(1, catalog['vlash_source'])
    from policy_inference import Policy
    recipe = json.loads(Path(entry['recipe']).read_text())
    checkpoint = Path(entry['checkpoint'])
    stat = checkpoint.stat()
    if stat.st_size != entry['checkpoint_bytes'] or stat.st_mtime_ns != entry['checkpoint_mtime_ns']:
        raise ValueError('Pinned checkpoint changed')
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    if saved['identity'] != entry['identity'] or saved['progress'] != entry['progress']:
        raise ValueError('Checkpoint identity/progress changed')
    refs = json.loads((Path(entry['code_root']) / 'checkpoints.local.json').read_text())
    tokenizer_path = Path(refs['google/paligemma-3b-pt-224']['path'])
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    with pinned_tokenizer(AutoTokenizer, tokenizer_path):
        model = Policy(entry['code_root'], recipe, 'cuda')
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().requires_grad_(False)
    adapter = Adapter(model, recipe, torch)
    dummy = {key: torch.zeros((1, 1, *feature.shape)) for key, feature in adapter.config.input_features.items()}
    env = torch.zeros(1, 1, 9)
    env[..., 6] = .1
    dummy.update(task='Pick up the moving object.', **{'observation.environment_state': env}, index=0, dt_scale=1.)
    _, _, prediction, elapsed = adapter.infer(dummy, adapter.generation)
    if args.probe:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'policy_probe.json').write_text(json.dumps({
            'strict_checkpoint_loaded': True, 'model': args.model, 'progress': entry['progress'],
            'identity': entry['identity'], 'checkpoint_sha256': entry['checkpoint_sha256'],
            'finite_physical_shape': list(prediction.shape), 'gpu': torch.cuda.get_device_name(0),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'probe_chunk_s': elapsed,
            'tokenizer_snapshot': str(tokenizer_path), 'rpc_wait': 'libzmq poll, timeout 10 ms'}, indent=2) + '\n')
        adapter.executor.shutdown()
        return
    source = Path(catalog['official_source'])
    sys.path.append(str(source))
    spec = importlib.util.spec_from_file_location('official_rpc', source / 'scripts/inference.py')
    rpc = importlib.util.module_from_spec(spec)
    text = (source / 'scripts/inference.py').read_text().replace('import utils.helpers', 'import rpc_geometry as helpers').replace('utils.helpers.', 'helpers.')
    exec(compile(text, str(source / 'scripts/inference.py'), 'exec'), rpc.__dict__)
    transform = rpc._get_transformed_observation

    def with_object_state(observation, rotation, features, device='cpu'):
        import numpy as np
        value = transform(observation, rotation, features, device)
        obj = observation['observation.environment_state']['object']
        env = np.concatenate([obj['pos'], rpc.helpers.get_rotation_vector(obj['quat'], rotation), obj['velocity']], axis=-1).astype(np.float32)
        value['observation.environment_state'] = torch.from_numpy(env).to(device)
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
        count, results = rpc.run_tests(*sockets, {'delta_timestamps': [0]}, adapter,
                                      str(checkpoint), entry['progress']['epoch'], 'euler', True, str(args.output))
        logging.info('Evaluation completed. Total tests run: %d', count)
    finally:
        adapter.executor.shutdown(wait=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s')
    main()
