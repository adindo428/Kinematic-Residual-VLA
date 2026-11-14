"""Sequential adapter checkpoint blocks in a persistent low-clutter simulator."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import logging
from pathlib import Path
import signal
import sys
import threading
import time

from resident_adapter_table1 import Pool
from resident_table1_client import patch_run_tests, write
from smolvla_table1_client import install_waiting_rpc

STOP = threading.Event()


def verify_pilots(plan):
    import numpy as np
    snapshots = {}; instances = {}; prompts = {}
    for job in plan['jobs']:
        if job['scored']: continue
        folder = Path(job['folder'])
        assert json.loads((folder/'resident_client_completion.json').read_text())['trials'] == 1
        rows = [json.loads(s) for s in (folder/'reset_states.jsonl').read_text().splitlines()]
        assert len(rows) == 1
        state = rows[0]['state']; scene = job['scene']
        instances.setdefault(scene, set()).add(rows[0]['scene_instance'])
        if scene in snapshots:
            base = snapshots[scene]
            for key in ('robot_joint_pos', 'robot_joint_vel'): np.testing.assert_allclose(state[key], base[key], atol=1e-5, rtol=0)
            assert state['objects'].keys() == base['objects'].keys()
            for key in state['objects']: np.testing.assert_allclose(state['objects'][key], base['objects'][key], atol=1e-5, rtol=0)
        else: snapshots[scene] = state
        row = json.loads((folder/'prompt_trace.jsonl').read_text().splitlines()[0])
        assert prompts.setdefault(scene, row['text']) == row['text']
    assert all(len(v) == 1 for v in instances.values())
    return {'passed': True, 'all_pilots_excluded': True, 'scene_reused': True,
            'matched_reset_states_atol': 1e-5, 'same_frozen_prompts': True}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); plan = json.loads(args.plan.read_text()); args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s %(message)s', force=True)
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    pool = None; handler = None; qualified_scene = False; slow_blocks = {}; live_health = []
    try:
        write(args.output/'status.json', {'stage': 'loading_resident_policies', 'updated_utc': datetime.now(timezone.utc).isoformat()})
        pool = Pool(plan['models'], args.output/'workers')
        pool.qualify_all(args.output)
        import torch
        source = Path(plan['catalog']['official_source']); sys.path.append(str(source))
        script = source/'scripts/inference.py'
        spec = importlib.util.spec_from_file_location('resident_adapter_rpc', script)
        rpc = importlib.util.module_from_spec(spec)
        text = script.read_text().replace('import utils.helpers', 'import rpc_geometry as helpers').replace('utils.helpers.', 'helpers.')
        exec(compile(text, str(script), 'exec'), rpc.__dict__)
        run_text = patch_run_tests(text)
        needle = '        observations.append(observation)'
        assert run_text.count(needle) == 1
        run_text = run_text.replace(needle, '        _adapter_update_phase(vla_model, observation)\n'+needle)
        from adapter_grasp_gating import update_live_gate
        rpc._adapter_update_phase = lambda front, obs: update_live_gate(front, obs, rpc.helpers.get_rotation_vector)
        exec(compile(run_text, str(script), 'exec'), rpc.__dict__)
        transform = rpc._get_transformed_observation
        def with_object_state(observation, rotation, features, device='cpu'):
            import numpy as np
            value = transform(observation, rotation, features, device)
            obj = observation['observation.environment_state']['object']
            env = np.concatenate([obj['pos'], rpc.helpers.get_rotation_vector(obj['quat'], rotation), obj['velocity']], axis=-1).astype(np.float32)
            value['observation.environment_state'] = torch.from_numpy(env).to(device)
            if 'evaluation.phase_gate' in observation:
                value['evaluation.phase_gate'] = torch.from_numpy(observation['evaluation.phase_gate']).to(device)
            return value
        rpc._get_transformed_observation = with_object_state; install_waiting_rpc(rpc)
        get = rpc.get_latest_observation
        def latest(socket):
            if STOP.is_set(): raise InterruptedError('Resident adapter evaluation stopped')
            return get(socket)
        rpc.get_latest_observation = latest
        obs, act = rpc.get_zmq_sockets('127.0.0.1', 27600, 27601)
        models = {m['key']: m['entry'] for m in plan['models']}
        for job in plan['jobs']:
            if STOP.is_set(): raise InterruptedError('Resident adapter evaluation stopped')
            if job['scored'] and not qualified_scene:
                write(args.output/'pilot_verification.json', verify_pilots(plan)); qualified_scene = True
            entry = models[job['model']]; folder = Path(job['folder']); folder.mkdir(parents=True, exist_ok=True)
            assert not (folder/'trial_results.jsonl').exists(), 'Never overwrite saved trials'
            front = pool.select(job['model']); front.log = str(folder/'client.log')
            logger = logging.getLogger()
            if handler is not None: logger.removeHandler(handler); handler.close()
            handler = logging.FileHandler(folder/'client.log')
            handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s %(message)s')); logger.addHandler(handler)
            front.reset(); rpc._resident_fence = front.idle
            write(folder/'runtime.json', {'checkpoint_sha256': entry['checkpoint_sha256'], 'identity': entry['identity'],
                  'progress': entry['progress'], 'scored': job['scored'], 'repeat_offset': job['repeat_offset'],
                  'resident_checkpoints': len(pool.fronts), 'gpu_policy_capacity': pool.capacity,
                  'comparison_variants': len(plan['models']),
                  'physical_worker': pool.variant_to_worker[job['model']],
                  'parallel_inference': False, 'qualification': str(args.output/'resident_profile.json')})
            while rpc.get_latest_observation(obs) is not None: pass
            act.send_pyobj({'vla': job['model'], 'epoch': entry['progress']['epoch'], 'resident_job': job['id']})
            write(args.output/'status.json', {'stage': 'evaluating', 'job': job, 'updated_utc': datetime.now(timezone.utc).isoformat()})
            tick = time.perf_counter()
            n, results = rpc.run_tests(obs, act, {'delta_timestamps': [0]}, front, entry['checkpoint'], entry['progress']['epoch'], 'euler', True, str(folder/'actions'))
            front.idle(); assert n == job['repeats']
            logging.info('Evaluation completed. Total tests run: %d', n)
            from run_resident_adapter_table1 import validate_rows, rows
            protocol = json.loads((args.plan.parent.parent/'protocol.json').read_text())
            test = next(t for t in protocol['tests'] if t['name'] == Path(job['scene']).stem)
            validate_rows(rows(folder/'trial_results.jsonl'), rows(folder/'prompt_trace.jsonl'),
                          rows(folder/'trajectory_metrics.jsonl'), test, job['repeat_offset'])
            write(folder/'resident_client_completion.json', {'complete': True, 'trials': n, 'job': job,
                  'wall_seconds': time.perf_counter()-tick, 'checkpoint_sha256': entry['checkpoint_sha256'],
                  'post_episode_wait': 'worker completion fence', 'original_inference_dt_scale_sleep_preserved': True})
            if job['scored']:
                from monitor_table1_latency import log_samples, coverage
                from capture_table1_runtime import stats
                samples = log_samples(folder/'client.log')
                measured = stats([r['ms'] for r in samples])
                reference = pool.profile['paired_trials'][-1]['models'][job['model']]['single']
                enough = len(samples) >= 100
                ok = not enough or (measured['p50_ms'] <= reference['p50_ms']/.9 and measured['p95_ms'] <= reference['p95_ms']/.9)
                slow_blocks[job['model']] = slow_blocks.get(job['model'], 0)+1 if not ok else 0
                health = {'job': job['id'], 'model': job['model'], 'measured': measured, 'isolated_reference': reference,
                          'action_coverage': coverage(folder/'client.log'), 'minimum_retention': .9,
                          'enough_samples': enough, 'within_reference': ok, 'gpu_capacity': pool.capacity,
                          'updated_utc': datetime.now(timezone.utc).isoformat()}
                if slow_blocks[job['model']] >= 2 and pool.capacity > 1:
                    pool.set_capacity(pool.capacity-1)
                    health['capacity_reduced_at_episode_boundary'] = pool.capacity
                    slow_blocks[job['model']] = 0
                live_health.append(health); write(args.output/'live_health.json', {'checks': live_health})
        act.send_pyobj({'resident_stop': True})
        write(args.output/'status.json', {'stage': 'complete', 'updated_utc': datetime.now(timezone.utc).isoformat()})
    finally:
        if pool is not None: pool.close()


if __name__ == '__main__': main()
