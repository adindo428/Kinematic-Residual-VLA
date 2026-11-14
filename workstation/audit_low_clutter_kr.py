"""Read-only CPU audit of saved low-clutter scores and KR coordinate contracts."""
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / 'scratch/dom45'
LOW = WORK / 'evaluations/low-clutter-full-table1-20261102'
OUT = WORK / 'audits/kr-low-clutter-20261111'


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def functions(text):
    return {n.name: ast.dump(n, include_attributes=False)
            for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef)}


def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    catalog = read(LOW/'catalog.json')
    protocol = read(LOW/'protocol.json')
    tests = {t['name']: t for t in protocol['tests']}
    remote = read(OUT/'palmetto_sources.private.json')['roots']
    evidence = read(OUT/'palmetto_training_evidence.private.json')
    result = {'scope': 'Read-only CPU audit; no checkpoint, score, or running job changed',
              'scores': {}, 'adapters': {}, 'native': {}, 'paired_outcomes': {}}
    all_records = {}
    resident = LOW/'resident_adapters-20261108'
    for name in catalog['order']:
        entry = catalog['models'][name]
        aggregate = resident/'aggregates'/name/'trial_results.jsonl'
        if aggregate.exists():
            records = rows(aggregate)
        else:
            records = []
            for path in sorted((LOW/'results'/name/'full').glob('*/trial_results.jsonl')):
                dim = path.parent.name.split('_', 1)[1]
                records += [{**r, 'dimension': dim, 'source': str(path.parent)} for r in rows(path)]
        assert len(records) == 1800, (name, len(records))
        by_key = {(r['env_name'], r['trial']): r for r in records}
        assert len(by_key) == 1800
        assert Counter(r['env_name'] for r in records) == Counter({n: 20 for n in tests})
        cache = {}
        for r in records:
            assert r['status'] in (0, 1) and r['success'] == (r['status'] == 0)
            assert abs(r['task_time_s']-r['n_steps']*r['step_dt_s']) < 1e-5
            assert abs(r['step_dt_s']-.04) < 1e-8
            folder = Path(r['source'])
            if folder not in cache:
                trace = rows(folder/'prompt_trace.jsonl')
                runtime = read(folder/'runtime.json') if (folder/'runtime.json').exists() else {}
                cache[folder] = (trace, runtime)
            trace, runtime = cache[folder]
            offset = runtime.get('repeat_offset', 0)
            match = [p for p in trace if p['env_name'] == r['env_name'] and p['trial']+offset == r['trial']]
            assert len(match) == 1, (name, r['env_name'], r['trial'], folder, offset)
            config = read(tests[r['env_name']]['path'])
            assert match[0]['text'] == config['evaluation_prompts'][r['trial']-1]
            if 'seed' in match[0]:
                assert match[0]['seed'] == config['seed']
            if 'checkpoint_sha256' in runtime and 'checkpoint_sha256' in entry:
                assert runtime['checkpoint_sha256'] == entry['checkpoint_sha256']
        success = sum(r['success'] for r in records)
        result['scores'][name] = {'trials': 1800, 'successes': success, 'success_percent': success/18,
                                 'unique_cases': 90, 'repeats_per_case': 20,
                                 'prompts_status_time_verified': True,
                                 'seed_check': 'Matches frozen config where prompt trace records a seed; resident traces omit this field'}
        all_records[name] = by_key
        if entry.get('method') not in ('smolvla', 'pi05', 'vlash'):
            cfg = read(Path(entry['checkpoint'])/'config.json')
            result['native'][name] = {k: cfg.get(k) for k in ['enable_residual_flow',
                'condition_on_object_velocity', 'residual_action_dt', 'residual_phase_gate_mode',
                'residual_flow_formulation', 'residual_normalize_drift']}
            if cfg['enable_residual_flow']:
                fetched = read(OUT/'native_training.private.json')['files']
                for relative in ['policies/dynamicvla/modeling_dynamicvla.py', 'proactive_vla/residual_flow.py']:
                    assert (Path(entry['source'])/relative).read_text() == fetched['source/'+relative]['text']
                spec = importlib.util.spec_from_file_location('audit_native_flow', Path(entry['source'])/'proactive_vla/residual_flow.py')
                flow = importlib.util.module_from_spec(spec); spec.loader.exec_module(flow)
                velocity = torch.tensor([[.1, -.2, .05]])
                std = torch.linspace(.1, 1., 32)
                center = torch.tensor(cfg['residual_drift_mean'])
                kwargs = dict(action_std=std, drift_mean=center, center_drift=True,
                              action_dt=cfg['residual_action_dt'])
                drift, _ = flow.centered_action_drift(velocity, (1, 20, 32), **kwargs)
                expected = velocity[:, None, :]*torch.arange(20)[None, :, None]*.04/std[:3]-center[None, :, :3]
                torch.testing.assert_close(drift[..., :3], expected)
                assert not drift[..., 3:].any()
                actions = torch.randn(1, 20, 32)
                restored, _, _ = flow.restore_residual_actions(actions-drift, velocity, **kwargs)
                torch.testing.assert_close(restored, actions)
                result['native'][name].update(remote_training_source_equal=True,
                                              xyz_dt_and_normalized_roundtrip_passed=True)
            continue
        root = Path(entry['code_root'])
        recipe = read(entry['recipe'])
        # Exact train-time helper and functional method comparison against live remote source.
        training_root = next(rr for rr in remote if
            (recipe['method'] == 'smolvla' and rr.endswith('main8-h200-only-20261022')) or
            (recipe['method'] != 'smolvla' and rr.endswith('main8-replicas-h200-20261021')))
        source = remote[training_root]
        assert (root/'common.py').read_text() == source['common.py']['text']
        train_funcs = functions(source['policy.py']['text'])
        infer_funcs = functions((root/'policy_inference.py').read_text())
        for method in ('affine', 'prepare', 'forward', 'predict'):
            assert train_funcs[method] == infer_funcs[method], (name, method)
        assert read(root/'statistics.json') == json.loads(evidence[training_root]['statistics.json'])
        assert read(root/'data-provenance.json') == json.loads(evidence[training_root]['data-provenance.json'])
        checkpoint = torch.load(entry['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
        assert checkpoint['identity'] == entry['identity']
        assert checkpoint['progress'] == entry['progress']
        actual_lrs = [group['lr'] for group in checkpoint['optimizer']['param_groups']]
        mean = checkpoint['model']['velocity_mean']
        torch.testing.assert_close(mean, torch.tensor(read(root/'data-provenance.json')['velocity_mean']))
        del checkpoint
        spec = importlib.util.spec_from_file_location('audit_common', root/'common.py')
        common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
        v = torch.tensor([[.1, -.2, .05], [0., 0., 0.]])
        offset = torch.tensor([0, recipe['max_delay']])
        correction = common.drift(v, mean, 20, offset)
        expected = (v-mean)[:, None, :]*(torch.arange(20)[None, :]+offset[:, None])[:, :, None]*.04
        torch.testing.assert_close(correction[..., :3], expected)
        assert not correction[..., 3:].any()
        raw = torch.randn(2, 20, 7)
        stats = read(root/'statistics.json')['action']
        if recipe['method'] == 'smolvla':
            center, scale = torch.tensor(stats['mean']), torch.tensor(stats['std'])
        else:
            lo, hi = torch.tensor(stats['q01']), torch.tensor(stats['q99'])
            center, scale = (lo+hi)/2, ((hi-lo)/2).clamp_min(1e-6)
        normalized = (common.coordinates(raw, correction)-center)/scale
        reconstructed = common.coordinates(normalized*scale+center, correction, inverse=True)
        torch.testing.assert_close(reconstructed, raw, atol=1e-6, rtol=1e-5)
        curve = []
        for rr, fs in evidence.items():
            if rr != training_root and not (name == 'vlash_baseline' and rr.endswith('/vlash_baseline')):
                continue
            for filename, text in fs.items():
                if filename.startswith('runs/'+name+'/validation-'):
                    x = json.loads(text)
                    curve.append({'epoch': x['progress']['epoch'], 'updates': x['progress']['updates'],
                                  'action_mse': x['action_mse']})
                elif filename == 'runs/'+name+'/completed.json':
                    x = json.loads(text)
                    curve.append({'epoch': x['progress']['epoch'], 'updates': x['progress']['updates'],
                                  'action_mse': x['validation_h20_action_mse']})
        curve = sorted({x['updates']: x for x in curve}.values(), key=lambda x: x['updates'])
        # Parent scheduler is already past its cosine decay at the start of continuation.
        result['adapters'][name] = {'checkpoint_identity_progress_mean_verified': True,
            'train_infer_function_AST_equal': True, 'remote_statistics_provenance_equal': True,
            'xyz_dt_and_affine_roundtrip_passed': True,
            'phase_gate': recipe['phase_gate'], 'neural_velocity_input': recipe['velocity_conditioning'],
            'training_progress': entry['progress'], 'parent_updates': recipe['parent_updates'],
            'schedule_steps': recipe['schedule_steps'],
            'continuation_starts_at_lr_floor': recipe['parent_updates'] >= recipe['schedule_steps'],
            'continuation_lr': recipe['decay_lr'], 'configured_peak_lr': recipe['lr'],
            'checkpoint_optimizer_lrs': actual_lrs,
            'static_target_final_centered_correction_m': common.drift(torch.zeros(1, 3), mean, 20)[0, -1, :3].tolist(),
            'validation_history': curve,
            'best_recorded_validation': min(curve, key=lambda x: x['action_mse']) if curve else None}
    for b, k in [('dom100_best', 'krv100_best'), ('dom100_latest', 'krv100_latest'),
                 ('smolvla_baseline', 'smolvla_kr'), ('vlash_baseline', 'vlash_kr'), ('pi05_baseline', 'pi05_kr')]:
        counts = Counter()
        for key, a in all_records[b].items():
            z = all_records[k][key]
            counts['both' if a['success'] and z['success'] else 'baseline_only' if a['success']
                   else 'kr_only' if z['success'] else 'neither'] += 1
        result['paired_outcomes'][b+'__'+k] = dict(counts)
    (OUT/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
