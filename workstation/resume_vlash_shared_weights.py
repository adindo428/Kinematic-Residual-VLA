"""Preserve durable trials and resume the three variants using two networks."""
from collections import Counter
import fcntl
import json
from pathlib import Path
import shutil

import run_vlash_gated_comparison as driver


def main():
    runner = driver.runner; low = runner.low; root = driver.ROOT
    with (low.ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = low.read(root/'plan.json')
        assert not old.get('shared_checkpoint_workers'), 'Sharing already configured'
        runner.MODELS[:] = [m['key'] for m in old['models']]
        records = runner.collect(old, validate=True)
        tests = {t['name']: t for t in low.read(low.ROOT/'protocol.json')['tests']}
        sources = list(low.read(root/'existing.json')['sources'])
        for job in old['jobs']:
            folder = Path(job['folder']); raw_path = folder/'trial_results.jsonl'
            if not job['scored'] or not raw_path.exists(): continue
            raw = runner.rows(raw_path)
            if not (folder/'resident_client_completion.json').exists():
                prompts = runner.rows(folder/'prompt_trace.jsonl')
                metrics = runner.rows(folder/'trajectory_metrics.jsonl')
                count = min(len(raw), len(prompts), len(metrics))
                prefix = raw[:count]
                runner.validate_rows(prefix, prompts[:count], metrics[:count], tests[Path(job['scene']).stem], job['repeat_offset'])
                records[job['model']] += [{**r, 'trial': r['trial']+job['repeat_offset'],
                    'dimension': job['dimension'], 'source': str(folder)} for r in prefix]
            sources.append({'model': job['model'], 'folder': str(folder),
                            'records_sha256': low.sha(raw_path), 'trials': len(raw), 'durable_prefix': True})
        for key, rr in records.items():
            ids = [(r['env_name'], r['trial']) for r in rr]
            assert len(ids) == len(set(ids))
        archive = root/'archives'/('before-shared-weights-'+runner.stamp()[:19].replace(':', '').replace('-', ''))
        archive.mkdir(parents=True)
        for name in ['plan.json', 'existing.json', 'live_results.json', 'request.json', 'start_receipt.json']:
            shutil.copy2(root/name, archive/name)
        for name in ['execution', 'failure.json']:
            path = root/name
            if path.exists(): shutil.move(str(path), str(archive/name))
        specs = old['models']
        for spec in specs: spec['share_checkpoint_worker'] = True
        jobs = []
        for job in old['jobs']:
            if job['scored']: continue
            updated = {**job, 'id': 'shared-'+job['id'], 'folder': str(root/'blocks'/('shared-'+job['id']))}
            jobs.append(updated)
        for job in old['jobs']:
            if not job['scored']: continue
            present = {r['trial'] for r in records[job['model']] if r['env_name'] == Path(job['scene']).stem}
            begin = job['repeat_offset']+1; end = begin+job['repeats']
            missing = [i for i in range(begin, end) if i not in present]
            if not missing: continue
            assert missing == list(range(missing[0], end))
            updated = {**job, 'id': 'shared-'+job['id'], 'repeat_offset': missing[0]-1,
                       'repeats': len(missing), 'folder': str(root/'blocks'/('shared-'+job['id']))}
            jobs.append(updated)
        for job in jobs:
            folder = Path(job['folder']); folder.mkdir(parents=True, exist_ok=True)
            assert not (folder/'trial_results.jsonl').exists()
            (folder/'case.txt').write_text(job['scene']+'\n')
        pins = {path: low.sha(path) for path in old['files_sha256']}
        pins[str(Path(__file__).resolve())] = low.sha(__file__)
        plan = {**old, 'models': specs, 'jobs': jobs, 'files_sha256': pins,
                'shared_checkpoint_workers': True, 'physical_networks': 2, 'comparison_variants': 3,
                'new_scored_trials': sum(j['repeats'] for j in jobs if j['scored']),
                'success_history': {k: sorted({r['env_name'] for r in rr if r['success']}) for k, rr in records.items()},
                'resume_archive': str(archive), 'updated_utc': runner.stamp()}
        low.write(root/'existing.json', {'records': records, 'sources': sources})
        low.write(root/'plan.json', plan)
        request = low.read(root/'request.json')
        request.update(shared_checkpoint_workers=True, physical_networks=2, comparison_variants=3)
        low.write(root/'request.json', request)
        low.write(root/'shared_weights_resume_receipt.json', {'retained_trials': {k: len(rr) for k, rr in records.items()},
            'remaining_scored_trials': plan['new_scored_trials'], 'archive': str(archive),
            'checkpoint_weights_unchanged': True, 'gate_changes_reconstruction_only': True})
        low.write(root/'execution/status.json', {'stage': 'preparing', 'updated_utc': runner.stamp()})
        runner.publish(plan)
        print(json.dumps(low.read(root/'shared_weights_resume_receipt.json'), indent=2))


if __name__ == '__main__': main()
