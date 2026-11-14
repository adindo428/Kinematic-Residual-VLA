"""Run the checkpoint's existing causal gate with diagnostic telemetry."""
import json
import os
from pathlib import Path
import sys


def record_gate(policy,observation,episode):
    import numpy as np
    trackers=policy._dom_phase_trackers
    gate=observation[policy.config.residual_phase_gate_key]
    row={'episode_sequence':episode+1,'index':int(observation['index']),'time_s':float(observation['timestamp']),
         'phase':trackers[0].phase,'gate_first':float(gate[0,0,0]),'gate_last':float(gate[0,-1,0]),
         'previous_applied_gripper':None if observation['previous_gripper_command'] is None else np.asarray(observation['previous_gripper_command']).tolist(),
         'ee_position':np.asarray(observation['observation.state']['end_effector']['pos']).tolist(),
         'object_state':{k:np.asarray(v).tolist() for k,v in observation['observation.environment_state']['object'].items()}}
    with Path(os.environ['DOM_GATE_TRACE']).open('a') as f:f.write(json.dumps(row)+'\n')


def main():
    i=sys.argv.index('--source');source=Path(sys.argv[i+1]).resolve();del sys.argv[i:i+2]
    os.environ['KRVLA_POLICY_SOURCE']=str(source);sys.path.insert(0,str(source))
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
    from causal_gate_diagnostic_runtime import inference_loop
    DynamicVLAPolicy._inference_loop=staticmethod(inference_loop)
    script=source/'scripts/inference.py';text=script.read_text()
    before='            _update_live_phase_gate(vla_model, observation)';assert text.count(before)==1
    text=text.replace(before,before+'\n            _record_gate(vla_model, observation, n_tests)')
    sys.argv[0]=str(script)
    exec(compile(text,str(script),'exec'),{'__name__':'__main__','__file__':str(script),'_record_gate':record_gate})


if __name__=='__main__':main()
