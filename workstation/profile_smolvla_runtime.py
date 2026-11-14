#!/usr/bin/env python3
"""Measure SmolVLA chunk latency with idle, busy, and yielding RPC threads."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--model', default='smolvla_baseline')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    entry = catalog['models'][args.model]
    import torch
    torch.set_num_threads(2)
    from smolvla_table1_client import Adapter
    sys.path.insert(0, entry['code_root'])
    from policy_inference import Policy
    recipe = json.loads(Path(entry['recipe']).read_text())
    saved = torch.load(entry['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    model = Policy(entry['code_root'], recipe, 'cuda')
    model.load_state_dict(saved['model'], strict=True)
    del saved
    model.eval().requires_grad_(False)
    adapter = Adapter(model, recipe, torch)
    obs = {k: torch.zeros((1, 1, *v.shape)) for k, v in model.policy.config.input_features.items()}
    obs.update(task='Catch the moving drink can.', **{'observation.environment_state': torch.zeros(1, 1, 9)}, index=10, dt_scale=1.)
    torch.manual_seed(42)
    for _ in range(2):
        adapter.infer(obs, adapter.generation)
    report = {'model': args.model, 'gpu': torch.cuda.get_device_name(0), 'measurements': {}}
    with ThreadPoolExecutor(max_workers=1) as executor:
        for mode in ('direct', 'busy_rpc_thread', 'yielding_rpc_thread'):
            measurements = []
            for _ in range(3):
                start = time.perf_counter()
                if mode == 'direct':
                    result = adapter.infer(obs, adapter.generation)
                else:
                    future = executor.submit(adapter.infer, obs, adapter.generation)
                    while not future.done():
                        # Bound the deliberately starving case: an unchecked
                        # Python spin can delay thousands of GPU launches for
                        # minutes. Yield after two seconds to complete safely.
                        if mode == 'yielding_rpc_thread' or time.perf_counter() - start > 2:
                            time.sleep(.001)
                    result = future.result()
                measurements.append({'wall_s': time.perf_counter() - start, 'worker_inference_s': result[3]})
            report['measurements'][mode] = measurements
            print(mode, 'median_s', statistics.median(m['worker_inference_s'] for m in measurements), flush=True)
    adapter.executor.shutdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
