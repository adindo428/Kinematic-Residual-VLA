#!/usr/bin/env python3
"""Collection-compatible object observations with a shared frozen prompt schedule."""
import json
import os
from pathlib import Path
import sys

from collection_matched_server import CollectionObjectSelector


class PromptSchedule:
    def __init__(self):
        self.completed = {}

    def instruction(self, cfg):
        name = cfg['evaluation_case_id']
        prompts = cfg['evaluation_prompts']
        assert len(prompts) == 5 and all(isinstance(p,str) and p for p in prompts)
        index = self.completed.get(name, 0)
        assert index < len(prompts), 'Unexpected extra scored repeat'
        return prompts[index]

    def advance(self, cfg, result):
        # The upstream loop skips one-frame reset artifacts; do not shift the
        # prompts relative to the scored repeat when such an artifact occurs.
        if len(result['ee_path']) <= 1:
            return
        assert result['status'] in (0,1), 'No-action trial invalidates the phase'
        name = cfg['evaluation_case_id']
        index = self.completed.get(name,0)
        path = Path(os.environ['DOM_TRIAL_RESULTS']).with_name('prompt_trace.jsonl')
        row = {'env_name':name,'trial':index+1,'text':self.instruction(cfg),'seed':cfg['seed']}
        with path.open('a') as f:
            f.write(json.dumps(row)+'\n');f.flush()
        self.completed[name] = index+1


def main():
    scripts = Path(__file__).resolve().parents[1]/'evaluation/official_dom/scripts'
    sys.path.insert(0,str(scripts))
    from server_observer import instrument_and_run
    source = Path(os.environ['DOM_OFFICIAL_SOURCE'])/'simulations/evaluate.py'
    text = source.read_text(); schedule = PromptSchedule()
    replacements = {
        '    rcv_action = False':'    rcv_action = False\n    _collection_target = _CollectionObjectSelector(env, sim)',
        '            object_state=env.unwrapped.scene["object"].data,':'            object_state=_collection_target.state(),',
        '                **{"observation.images.%s" % k: v["rgb"] for k, v in cam_view.items()},':
            '                "observation.environment_state": {"object": {\n'
            '                    k: v.cpu().numpy() for k, v in curr_state["object"].items()}},\n'
            '                **{"observation.images.%s" % k: v["rgb"] for k, v in cam_view.items()},',
        '    instruction = InstructionGenerator.generate_instruction(env_cfg["instruction"])':
            '    instruction = _prompt_for(env_cfg)',
        '    sim_results = simulate(env, obs_socket, act_socket, sim_cfg["init_poses"])':
            '    sim_results = simulate(env, obs_socket, act_socket, sim_cfg["init_poses"])\n    _advance_prompt(env_cfg, sim_results)',
        '    main(app_launcher.app, args)':'    _instrument_and_run(main, app_launcher.app, args, globals())'}
    for before, after in replacements.items():
        assert text.count(before)==1, 'Pinned simulator hook changed: '+before
        text = text.replace(before,after)
    sys.path.insert(0,str(source.parent.parent))
    namespace = {'__name__':'__main__','__file__':str(source),'_instrument_and_run':instrument_and_run,
        '_CollectionObjectSelector':CollectionObjectSelector,'_prompt_for':schedule.instruction,'_advance_prompt':schedule.advance}
    exec(compile(text,str(source),'exec'),namespace)


if __name__=='__main__':
    main()
