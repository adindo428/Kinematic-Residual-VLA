#!/usr/bin/env python3
"""Write scene-match evidence and unqueued speed pairs after expert preparation."""
import argparse
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();root=args.root
    ready=root/'expert_scene_preparation_complete.json'
    while not ready.exists():
        if not args.wait:raise FileNotFoundError(ready)
        state=subprocess.run(['systemctl','--user','is-active','krvla-expert-prompt-matched20k-20261102.service'],capture_output=True,text=True).stdout.strip()
        if state not in ['active','activating','deactivating']:
            raise RuntimeError('Scene preparation is not active; inspect controller.log before retrying')
        time.sleep(15)
    selection=json.loads((root/'selection.json').read_text());rows=[]
    for case in selection['cases']:
        evidence=json.loads((root/'case_evidence'/(case['name']+'.json')).read_text())
        expert=evidence['expert_feasibility'];assert expert['collection_eligible']
        vec=expert['first_object_position'][:2]+expert['first_object_velocity'][:2]
        values={}
        for key,state in case['training_first_states'].items():
            reference=[state[0],state[1],state[6],state[7]]
            values[key]=math.sqrt(sum(((x-y)/.25)**2 for x,y in zip(vec,reference)))
        nearest=min(values,key=values.get);reference=case['training_first_states'][nearest]
        rows.append({'case':case['name'],'task':case['task'],'n_objects':case['n_objects'],'n_containers':case['n_containers'],
            'container_asset':case['container_asset'],'prompt_witnesses':evidence['compatible_prompt_witnesses'],
            'expert_success':True,'expert_steps':expert['steps'],'nearest_training_episode':int(nearest),
            'expert_first_state_distance':values[nearest],
            'position_xy_gap_m':math.sqrt(sum((x-y)**2 for x,y in zip(vec[:2],reference[:2]))),
            'velocity_xy_gap_m_s':math.sqrt(sum((x-y)**2 for x,y in zip(vec[2:],reference[6:8])))})
    summary={'cases':len(rows),'training_supported_prompts':5*len(rows),'collection_expert_completed':len(rows),
        'policy_results_used_for_selection':False,'house_membership_verified':False,
        'median_expert_first_state_distance':statistics.median(r['expert_first_state_distance'] for r in rows),
        'distance_definition':'Nearest XY object position/velocity among the exact training task/principal-asset/rigid-count/speed-bin witnesses; scales 0.25 m and 0.25 m/s.',
        'per_case':rows}
    (root/'similarity_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Evidence for the training-distribution match','','Approximately IID; house membership remains unverified. Final scenes are selected by fixed dataset/collection criteria, never VLA success. Prototype pilots and earlier scores remain separate.','','| Check | Evidence |','|---|---|',
        '| Task mix | 9 pick, 9 place, 2 long horizon; 45/45/10 scored trials |',
        '| Task/asset/count/speed stratum | Training witnesses for every scenario |',
        '| Joint prompts | All 100 frozen prompt variants are supported by training metadata and its instruction generator |',
        '| Long-horizon containers | Both are boxes, matching all 2,000 training long-horizon instructions |',
        '| Physical feasibility | All 20 complete with the original collection expert at 0.015 m tolerance and pass its early-stop exclusion |',
        '| Cameras | Runner verifies dimensions, focal lengths, offsets and rotations against training metadata |',
        '| Method comparison | Identical scenarios, reset seeds, prompts and success rules; prompt traces verified |',
        '| Velocity OOD | Prepared separately; not scheduled or counted in the 100 IID trials |','',
        'Thirty-two layouts per anchor are ranked using training initial-state similarity. The closest candidate passing the independent collection expert and semantic checks is frozen. The generator removes direction descriptions if the collection direction-change check requires it. This deliberately concentrates on training-like, physically feasible configurations; it does not prove IID or estimate full benchmark performance.','',
        '| Case | Objects / containers | Container | Prompt witnesses | Expert steps | Training XY position gap (m) | Velocity gap (m/s) |',
        '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['n_objects']} / {r['n_containers']} | {r['container_asset']} | {r['prompt_witnesses']} | {r['expert_steps']} | {r['position_xy_gap_m']:.3f} | {r['velocity_xy_gap_m_s']:.3f} |")
    lines+=['','State gaps compare the expert rollout\'s first state with recorded training first states. Translation/frame-timing differences can remain. Exact house, original distractor identities/poses and complete training/pretraining speed exclusion are not established. Expert feasibility without rendering does not guarantee VLA success. Five repeats share a scene.','']
    (root/'distribution_match.md').write_text('\n'.join(lines))
    subprocess.run([sys.executable,str(REPO/'workstation/prepare_velocity_shift_cases.py'),'--root',str(root)],check=True)
    (root/'evidence_finalization_complete.json').write_text(json.dumps({'complete':True,'expert_checked_cases':len(rows),'velocity_pairs_prepared':len(rows),'velocity_trials_queued':0},indent=2)+'\n')
    print('Finalized training-match evidence and unqueued velocity pairs',flush=True)


if __name__=='__main__':main()
