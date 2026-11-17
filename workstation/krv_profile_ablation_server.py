"""Unchanged low-clutter simulator; VI integrates error against the active plan."""
import resident_low_clutter_server as server
import resident_table1_server as resident
_patch=resident.patch_source

def patch_source(text):
    edits={
      '    rcv_action = False':'''    from krv_profile_math import ProfileInnovation
    from resident_table1_server import CURRENT
    _profile_ctrl=ProfileInnovation(bool(CURRENT['vi']),float(env.env.step_dt))
    rcv_action = False''',
      '        sim_results["cam_views"].append(cam_view)':'        _profile_ctrl.observe(curr_state["object"]["velocity"].detach().cpu().numpy())\n        sim_results["cam_views"].append(cam_view)',
      '        if action is not None and "action" in action:':'        if action is not None and "action" in action:\n            _profile_ctrl.set_plan(action["profile_plan"])',
      '        env.step(last_action)':'        corrected=_profile_ctrl.correct(last_action.detach().cpu().numpy())\n        env.step(torch.as_tensor(corrected,device=last_action.device,dtype=last_action.dtype))',
      '    return sim_results':'''    from pathlib import Path
    from resident_table1_server import COUNTS
    _name=Path(CURRENT['scene']).stem
    _trace=Path(CURRENT['folder'])/'profile_traces'/f'{_name}-{COUNTS[_name]+1:03d}'
    _trace.mkdir(parents=True,exist_ok=True)
    _profile_ctrl.finish(_trace)
    return sim_results'''}
    for old,new in edits.items():
        assert text.count(old)==(2 if old=='    return sim_results' else 1),old
        text=text.replace(old,new,1)
    return _patch(text)

if __name__=='__main__':
    resident.patch_source=patch_source
    server.main()
