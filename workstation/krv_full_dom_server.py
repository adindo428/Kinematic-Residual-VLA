"""Original DOM prompt generation with frozen matched wording and passive loop metrics."""
import json
from pathlib import Path
import krv_profile_ablation_server as profile
import resident_table1_server as resident


def prompt(self,cfg):
    from utils.instruction_generator import InstructionGenerator
    scene=resident.CURRENT['scene'];trial=str(resident.COUNTS[Path(scene).stem]+1)
    saved=resident.PROMPTS.setdefault(scene,{})
    if trial not in saved:saved[trial]=InstructionGenerator.generate_instruction(cfg['instruction'])
    resident.LAST_PROMPT=saved[trial]
    return saved[trial]

_original=profile.patch_source

def patch_source(text):
    # Count drained messages separately from network loss: latest-only is intentional.
    text=text.replace('    action = None\n    while True:', '    action = None\n    _received = 0\n    while True:',1)
    text=text.replace('            action = act_socket.recv_pyobj(flags=zmq.NOBLOCK)', '            action = act_socket.recv_pyobj(flags=zmq.NOBLOCK)\n            _received += 1',1)
    text=text.replace('    return action\n\n\ndef _get_action_tensor', '    if action is not None: action["_drained_messages"] = _received\n    return action\n\n\ndef _get_action_tensor',1)
    text=_original(text)
    text=text.replace('    rcv_action = False','    _health = {"steps":0,"fresh_actions":0,"superseded_messages":0,"step_ms":[],"chunk_age_steps":[]}\n    rcv_action = False',1)
    text=text.replace('        action = get_latest_action(act_socket)','        _health["steps"] += 1\n        action = get_latest_action(act_socket)\n        if action is not None and "action" in action:\n            _health["fresh_actions"] += 1\n            _health["superseded_messages"] += max(0,action.get("_drained_messages",1)-1)\n            _health["chunk_age_steps"].append(len(sim_results["cam_views"])-1-action["profile_plan"]["origin"])',1)
    text=text.replace('        step_time = time.perf_counter() - tick','        step_time = time.perf_counter() - tick\n        _health["step_ms"].append(step_time*1000)',1)
    text=text.replace('    _profile_ctrl.finish(_trace)','    _profile_ctrl.finish(_trace)\n    with (Path(CURRENT["folder"])/"loop_health.jsonl").open("a") as _hf:\n        _hf.write(__import__("json").dumps(_health)+"\\n")',1)
    return text

if __name__=='__main__':
    profile.server.prompt=prompt
    resident.patch_source=patch_source
    profile.server.main()
