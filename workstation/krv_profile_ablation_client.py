"""Native streaming client with explicit frozen-profile metadata per action."""
from pathlib import Path
source=Path(__file__).with_name('resident_table1_client.py').read_text()
source=source.replace('from resident_table1_worker import ResidentPool,minimum_retention','from krv_profile_ablation_worker import ResidentPool,minimum_retention')
old="    return source.replace('time.sleep(10)','_resident_fence()')"
new='''    source=source.replace('time.sleep(10)','_resident_fence()')
    old='act_socket.send_pyobj({"action": action}, flags=zmq.NOBLOCK)'
    assert source.count(old)==1
    return source.replace(old,'act_socket.send_pyobj({"action": action, "profile_plan": vla_model.profile_plan}, flags=zmq.NOBLOCK)')'''
assert source.count(old)==1;source=source.replace(old,new)
old='            front.reset();module._resident_fence=front.idle'
source=source.replace(old,old+"\n            front._episode_seed=int(job['seed']);front.profile_plan=None")
source=source.replace("script=source/'scripts/inference.py'", "script=Path(__file__).resolve().parents[1]/'scratch/dom45/training/krv100-full-20261010/source/scripts/inference.py'")
exec(compile(source,str(Path(__file__).with_name('resident_table1_client.py')),'exec'))
