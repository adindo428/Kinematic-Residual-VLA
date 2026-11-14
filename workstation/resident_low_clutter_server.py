"""Frozen low-clutter prompts and metrics with the scene-caching resident server."""
import json
import os
from pathlib import Path
import sys

import low_clutter_table1_server as low
import resident_table1_server as resident


def prompt(self, cfg):
    name = cfg['evaluation_case_id']
    index = resident.COUNTS[name] + resident.CURRENT.get('repeat_offset', 0)
    assert 0 <= index < 20
    resident.LAST_PROMPT = cfg['evaluation_prompts'][index]
    return resident.LAST_PROMPT


def instrument(main, app, args, namespace):
    import numpy as np
    original = namespace['simulate']
    def simulate(env, *a, **kw):
        result = original(env, *a, **kw)
        if len(result['ee_path']) > 1:
            job = resident.CURRENT; folder = Path(job['folder']); name = Path(job['scene']).stem
            trial = resident.COUNTS[name] + 1
            path = np.asarray(result['ee_path'], dtype=np.float32).reshape(len(result['ee_path']), -1, 3)
            assert path.shape[1] == 1
            path = path[:, 0, :]; target = folder/'trajectories'/name/f'{trial:03d}.npz'
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, ee_position_robot_m=path, step_dt_s=float(env.env.step_dt))
            row = {'env_name': name, 'trial': trial, 'global_trial': trial + job.get('repeat_offset', 0),
                   'status': int(result['status']), 'ee_path_length_m': float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                   'trajectory_path': str(target), 'n_steps': len(path), 'step_dt_s': float(env.env.step_dt),
                   'task_time_s': len(path)*float(env.env.step_dt)}
            with (folder/'trajectory_metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n'); stream.flush()
        return result
    namespace['simulate'] = simulate
    return resident.instrument(main, app, args, namespace)


def main():
    scripts = Path(__file__).resolve().parents[1]/'evaluation/official_dom/scripts'
    sys.path.insert(0, str(scripts))
    low.PromptSchedule.instruction = prompt
    low.PromptSchedule.advance = lambda *_: None  # Resident observer owns per-block prompt recording.
    low.with_trajectory_metrics = instrument
    target = (Path(os.environ['DOM_OFFICIAL_SOURCE'])/'simulations/evaluate.py').resolve()
    original = Path.read_text
    def read(path, *a, **kw):
        text = original(path, *a, **kw)
        if path.resolve() == target:
            Path.read_text = original
            return resident.patch_source(text)
        return text
    Path.read_text = read
    try: low.main()
    finally: Path.read_text = original


if __name__ == '__main__': main()
