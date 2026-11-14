"""Frozen prompts, native closed-loop control, and per-step pickup evidence."""
import json
from pathlib import Path
import resident_low_clutter_server as server
import resident_table1_server as resident

_instrument = server.instrument

def instrument(main, app, args, namespace):
    import numpy as np
    original = namespace['simulate']
    def simulate(env, *a, **kw):
        trace = []
        step = env.step
        def observed_step(action):
            scene = env.unwrapped.scene
            trace.append(np.concatenate([
                scene['object'].data.root_pos_w.detach().cpu().numpy().reshape(-1),
                scene['ee_frame'].data.target_pos_w[..., 0, :].detach().cpu().numpy().reshape(-1),
                scene['object'].data.root_lin_vel_w.detach().cpu().numpy().reshape(-1),
            ]))
            return step(action)
        env.step = observed_step
        try: result = original(env, *a, **kw)
        finally: env.step = step
        if len(result['ee_path']) > 1:
            assert result['status'] in (0, 1), 'No-action episode is invalid'
            data = np.asarray(trace)
            folder = Path(resident.CURRENT['folder'])
            np.savez_compressed(folder/'object_hand_trace.npz', object_position_world_m=data[:, :3],
                hand_position_world_m=data[:, 3:6], object_velocity_world_mps=data[:, 6:9],
                step_dt_s=float(env.env.step_dt))
            lift = data[:, 2] - data[0, 2]
            near = np.linalg.norm(data[:, :3]-data[:, 3:6], axis=1) < .10
            lifted = (lift > .03) & near
            picked = len(lifted) >= 3 and bool(np.convolve(lifted.astype(int), np.ones(3), 'valid').max() == 3)
            (folder/'pickup_diagnostic.json').write_text(json.dumps({
                'heuristic_pickup': picked, 'definition': 'object lifted >3cm and within 10cm of hand for 3 consecutive frames',
                'max_lift_m': float(lift.max()), 'placement_success': result['status'] == 0,
                'not_a_contact_verified_grasp': True}, indent=2)+'\n')
        return result
    namespace['simulate'] = simulate
    return _instrument(main, app, args, namespace)

if __name__ == '__main__':
    server.instrument = instrument
    server.main()
