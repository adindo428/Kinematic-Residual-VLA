"""Three resident VLASH variants, corrected prompts, frozen low-clutter scenes."""
import argparse
import json
import os
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[1]
LOW = REPO/'scratch/dom45/evaluations/low-clutter-full-table1-20261102'
ROOT = LOW/'vlash-grasp-gating-20261111'
os.environ['DOM_RESIDENT_ADAPTER_ROOT'] = str(ROOT)

import run_resident_adapter_table1 as runner


def prepare():
    if (ROOT/'plan.json').exists():
        return runner.low.read(ROOT/'plan.json')
    low = runner.low
    prior = LOW/'resident_adapters-20261108'
    assert low.read(prior/'completion.json')['complete']
    ROOT.mkdir(parents=True, exist_ok=True)
    evidence = ROOT/'previous_results'; evidence.mkdir(exist_ok=True)
    for name in ['catalog.json', 'table1_comparison.md']:
        shutil.copy2(LOW/name, evidence/name)
    catalog = low.read(LOW/'catalog.json')
    protocol = low.read(LOW/'protocol.json')
    for test in protocol['tests']:
        assert low.sha(test['path']) == test['config_sha256']
    tests = {t['name']: t for t in protocol['tests']}
    gate = ROOT/'frozen_phase_gate.py'
    shutil.copy2(REPO/'scratch/dom45/training/krv100-full-deltaai-20261106/source/proactive_vla/phase_gate.py', gate)
    variants = [('vlash_baseline_fixed', 'vlash_baseline', False, 'VLASH (epoch 10; corrected prompts)'),
                ('vlash_kr_fixed', 'vlash_kr', False, 'VLASH + KR (epoch 10; corrected prompts)'),
                ('vlash_kr_gated', 'vlash_kr', True, 'VLASH + KR + grasp gating (epoch 10)')]
    specs = []
    for key, parent, gated, label in variants:
        entry = dict(catalog['models'][parent])
        assert entry['progress']['epoch'] == 10 and entry['progress']['cursor'] == 0
        stat = Path(entry['checkpoint']).stat()
        assert (stat.st_size, stat.st_mtime_ns) == (entry['checkpoint_bytes'], entry['checkpoint_mtime_ns'])
        entry.update(label=label, parent_model=parent, client_kind='resident_adapter',
            evaluation_intervention='corrected_training_prompt_and_causal_grasp_gate' if gated else 'corrected_training_prompt',
            gate_mode='causal_attachment' if gated else 'constant',
            gate_training_changed=False, inference_only_ablation=gated)
        catalog['models'][key] = entry
        spec = {'key': key, 'entry': entry,
                'catalog': {k: catalog[k] for k in ['official_source', 'vlash_source']},
                'corrected_prompt_contract': True, 'grasp_gate': gated, 'gate_source': str(gate),
                'share_checkpoint_worker': True}
        specs.append(spec)
        dest = LOW/'results'/key; dest.mkdir(exist_ok=True)
        low.write(dest/'protocol.json', {**protocol, 'checkpoint': entry})
        shutil.copytree(LOW/'test_lists', dest/'test_lists', dirs_exist_ok=True)
    keys = [s['key'] for s in specs]
    # Existing scored records are retained under their original identities.
    held = low.read(prior/'held_records.json')
    old = low.read(prior/'plan.json')
    runner.MODELS[:] = [m['key'] for m in old['models']]
    runner.ROOT = prior
    try:
        held.update(runner.collect(old, validate=True))
    finally:
        runner.ROOT = ROOT
    low.write(ROOT/'held_records.json', held)
    low.write(ROOT/'existing.json', {'records': {key: [] for key in keys}, 'sources': []})
    lists = {dim: [s for s in (LOW/'test_lists'/(dim+'.txt')).read_text().splitlines() if s] for dim in low.DIMS}
    jobs = []
    for index, model in enumerate([*keys, keys[0]]):
        scene = lists['CR'][0]; key = f'pilot-{index}-{model}'
        jobs.append(dict(id=key, model=model, scene=scene, scene_sha256=tests[Path(scene).stem]['config_sha256'],
            repeats=1, repeat_offset=0, folder=str(ROOT/'blocks'/key), dimension='CR', scored=False))
    for pass_index, offset in [(1, 0), (2, 10)]:
        for dimension in low.DIMS:
            for scene in lists[dimension]:
                for model in keys:
                    key = f'pass{pass_index}-{dimension}-{Path(scene).stem}-{model}'
                    jobs.append(dict(id=key, model=model, scene=scene,
                        scene_sha256=tests[Path(scene).stem]['config_sha256'], repeats=10, repeat_offset=offset,
                        folder=str(ROOT/'blocks'/key), dimension=dimension, scored=True, pass_index=pass_index))
    for job in jobs:
        folder = Path(job['folder']); folder.mkdir(parents=True, exist_ok=True)
        assert not (folder/'trial_results.jsonl').exists()
        (folder/'case.txt').write_text(job['scene']+'\n')
    files = [REPO/'workstation'/f for f in ['run_vlash_gated_comparison.py', 'run_resident_adapter_table1.py',
        'resident_adapter_table1.py', 'resident_adapter_table1_client.py', 'resident_low_clutter_server.py',
        'resident_table1_server.py', 'resident_table1_worker.py', 'adapter_grasp_gating.py',
        'smolvla_table1_client.py', 'causal_gate_diagnostic_server.py', 'collection_matched_server.py']]
    files.append(gate)
    for spec in specs:
        files += [p for p in Path(spec['entry']['code_root']).rglob('*') if p.is_file() and p.suffix in ('.py', '.json')]
    plan = dict(models=specs, catalog={k: catalog[k] for k in ['official_source', 'vlash_source']}, jobs=jobs,
        files_sha256={str(p): low.sha(p) for p in files}, protocol_sha256=low.sha(LOW/'protocol.json'),
        new_scored_trials=5400, success_history={key: [] for key in keys}, minimum_latency_retention=.9,
        first_pass_repeats=10, remaining_repeats=10, automatic_second_pass_authorized=True,
        corrected_prompt_contract=True, causal_gate_inference_only=True, target_lead_s=0, created_utc=runner.stamp())
    low.write(ROOT/'plan.json', plan)
    low.write(ROOT/'checkpoint_policy.json', {'minimum_completed_epochs': 10, 'enabled_models': keys, 'held_models': {}})
    low.write(ROOT/'request.json', {'enabled': True, 'models': keys, 'user_authorized': True,
        'minimum_latency_retention': .9, 'all_three_gpu_resident_if_qualified': True,
        'scene_first': True, 'concurrent_inference': False, 'block_repeats_maximum': 10,
        'automatic_second_pass_authorized': True, 'target_per_model': 1800, 'notify_on_completion': True})
    catalog['order'] = [k for k in catalog['order'] if k not in ['vlash_baseline', 'vlash_kr']]
    pos = catalog['order'].index('pi05_baseline')
    catalog['order'][pos:pos] = keys
    catalog.update(enabled_models=keys, queue_order=keys, held_models=[], active_resident_adapter_root=str(ROOT))
    low.write(LOW/'catalog.json', catalog)
    runner.MODELS[:] = keys
    low.write(ROOT/'execution/status.json', {'stage': 'preparing', 'updated_utc': runner.stamp()})
    runner.publish(plan)
    table = LOW/'table1_comparison.md'
    note = ('\nVLASH rerun: all three displayed VLASH variants use corrected training-consistent language/state tokens. '
            'The grasp-gated row is an inference-only causal attachment-gate ablation using the same KR weights. '
            'Earlier VLASH scores with mismatched prompts are archived under `vlash-grasp-gating-20261111/previous_results/`.\n')
    with table.open('a') as f: f.write(note)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    plan = prepare()
    print(json.dumps({'models': [m['key'] for m in plan['models']], 'scored_trials': 5400, 'root': str(ROOT)}), flush=True)
    if not args.prepare_only:
        runner.main()
