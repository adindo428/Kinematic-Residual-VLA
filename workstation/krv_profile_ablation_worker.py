"""Keep native trained policy intact; add a frozen causal profile to its chunk."""
from pathlib import Path
import inspect,textwrap
from types import MethodType
import velocity_innovation_worker as base

source=base._source
needle="                return {'actions':actions.transpose(0,1).cpu(),'index':index}"
replacement='''                import numpy as np
                from krv_profile_math import profile_extra
                env=batch['observation.environment_state'][0]
                velocities=env[:,6:9].detach().cpu().numpy()
                current=velocities[-1]
                acceleration=(current-velocities[0])/.08
                gated=bool(cfg.enable_residual_flow)
                if gated and cfg.residual_phase_gate_mode=='distance_threshold':
                    ee=batch['observation.state'][0,-1,:3]
                    gated=bool(torch.linalg.vector_norm(env[-1,:3]-ee) > cfg.residual_distance_threshold_m)
                plan={'origin':index,'velocity':current.tolist(),'acceleration':acceleration.tolist(),'gate':gated,'vpd':bool(spec.get('vpd',False))}
                if plan['vpd'] and gated:
                    extra=np.stack([profile_extra(acceleration,h*cfg.residual_action_dt) for h in range(cfg.chunk_size)])
                    actions[...,:3]+=torch.as_tensor(extra,device=actions.device,dtype=actions.dtype)[None]
                return {'actions':actions.transpose(0,1).cpu(),'index':index,'profile_plan':plan}'''
assert source.count(needle)==1
source=source.replace(needle,replacement)
# The usual probe compares to native actions. Constant-velocity synthetic probes
# have zero acceleration so every arm must retain exact native output there.
exec(compile(source,str(Path(base.__file__).with_name('resident_table1_worker.py')),'exec'))
_BasePool=ResidentPool

def stream(method,namespace):
    code=textwrap.dedent(inspect.getsource(method))
    edits={
      '        skip_n_actions = batch["index"] - actions["index"]':'        origin=int(actions["index"]);profile_plan=actions["profile_plan"]\n        skip_n_actions = batch["index"] - actions["index"]',
      '{"index": actions["index"] + i, "action": a}':'{"index": actions["index"] + i, "action": a, "plan":profile_plan}',
      '            self._action_index = action["index"]':'            self._action_index = action["index"]\n            self.profile_plan=action["plan"]',
      '        self._action_index = action["index"]':'        self._action_index = action["index"]\n        self.profile_plan=action["plan"]',
      '    self.q_in.update({"obs": (batch, noise)})':'    batch=dict(batch,_episode_seed=self._episode_seed)\n    self.q_in.update({"obs": (batch, noise)})'}
    for old,new in edits.items():
        if old not in code:continue
        assert code.count(old)==1,(old,code)
        code=code.replace(old,new)
    space=dict(namespace);exec(compile(code,'<profile_stream>','exec'),space);return space['_get_streaming_action']

class ResidentPool(_BasePool):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
        module=__import__(DynamicVLAPolicy.__module__,fromlist=['*'])
        method=stream(DynamicVLAPolicy._get_streaming_action,vars(module))
        for front in self.fronts.values():
            front._episode_seed=0;front.profile_plan=None
            front.select_action=MethodType(method,front)
