"""Continue only low-clutter CR trials, retaining finished scores from the prior scope."""
import copy,inspect,json
from pathlib import Path
import run_krv_profile_ablation as base

def launch(full):
    pending=copy.deepcopy(full)
    pending['jobs']=[j for j in full['jobs'] if not (Path(j['folder'])/'trial_results.jsonl').exists()]
    assert all(j['dimension']=='CR' for j in pending['jobs'])
    base.low.write(base.ROOT/'active-plan.json',pending)
    runner=base.runner;runner.ROOT=base.ROOT;runner.status=base.status
    runner.verify=lambda _:base.verify(base.low.read(base.ROOT/'plan.json'))
    text=inspect.getsource(runner.launch).replace("'closed-loop-pilot'","'krv-profile-ablation'").replace('closed_loop_pilot_server.py','krv_profile_ablation_server.py').replace('closed_loop_pilot_client.py','krv_profile_ablation_client.py').replace('total=75',"total=len(plan['jobs'])").replace('completed=75',"completed=len(plan['jobs'])")
    text=text.replace("'--plan', str(ROOT/'plan.json')", "'--plan', str(ROOT/'active-plan.json')")
    exec(compile(text,str(Path(__file__).resolve()),'exec'),runner.__dict__)
    runner.launch(pending)

if __name__=='__main__':
    base.launch=launch
    base.main()
