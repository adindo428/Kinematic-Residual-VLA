"""Apply bounded velocity correction immediately before the simulator step."""
import closed_loop_pilot_server as pilot
import resident_table1_server as resident

_patch = resident.patch_source


def patch_source(text):
    edits = {
        '    rcv_action = False': '''    from velocity_innovation import VelocityInnovation
    from resident_table1_server import CURRENT
    _velocity_ctrl = VelocityInnovation(CURRENT['correction_mode'], float(env.env.step_dt))
    rcv_action = False''',
        '        sim_results["cam_views"].append(cam_view)':
            '        _velocity_ctrl.observe(len(sim_results["cam_views"]), curr_state["object"]["velocity"].detach().cpu().numpy())\n        sim_results["cam_views"].append(cam_view)',
        '        if action is not None and "action" in action:':
            '        if action is not None and "action" in action:\n            _velocity_ctrl.set_origin(action["chunk_origin_index"])',
        '        env.step(last_action)':
            '        corrected = _velocity_ctrl.correct(last_action.detach().cpu().numpy())\n        env.step(torch.as_tensor(corrected, device=last_action.device, dtype=last_action.dtype))',
        '    return sim_results':
            '    _velocity_ctrl.finish(CURRENT["folder"])\n    return sim_results',
    }
    for old, new in edits.items():
        # return sim_results appears in both simulate and get_sim_results.
        if old == '    return sim_results':
            assert text.count(old) == 2
            text = text.replace(old, new, 1)
        else:
            assert text.count(old) == 1, old
            text = text.replace(old, new)
    return _patch(text)


if __name__ == '__main__':
    resident.patch_source = patch_source
    pilot.server.instrument = pilot.instrument
    pilot.server.main()
