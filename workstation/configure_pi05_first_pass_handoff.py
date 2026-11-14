"""Apply the user-requested VLASH first-pass handoff without restarting rollouts."""
import hashlib
import json
from pathlib import Path
import shutil

import queue_pi05_kr_epoch3 as queue


def main():
    low = queue.runner.low
    parent, target = queue.PARENT, queue.ROOT
    plan = low.read(parent/'plan.json')
    state = low.read(parent/'execution/status.json')
    assert state.get('stage') == 'evaluating' and state['job'].get('pass_index') == 1
    assert not (parent/'completion.json').exists()
    assert not (parent/'first_pass_handoff.json').exists()
    assert not any((Path(j['folder'])/'trial_results.jsonl').exists()
                   for j in plan['jobs'] if j.get('pass_index') == 2)
    queued = low.read(target/'plan.json')
    script = str(Path(queue.__file__).resolve())
    for path, digest in queued['files_sha256'].items():
        if path != script:
            assert low.sha(path) == digest, f'Unrelated pinned source changed: {path}'
    if 'deferred_jobs' not in plan:
        deferred = [j for j in plan['jobs'] if j.get('pass_index') == 2]
        assert len(deferred) == 270 and sum(j['repeats'] for j in deferred) == 2700
        shutil.copy2(parent/'plan.json', parent/'plan.before-first-pass-handoff.json')
        plan['jobs'] = [j for j in plan['jobs'] if j.get('pass_index') != 2]
        plan['deferred_jobs'] = deferred
        plan['handoff_after_trials_per_model'] = 900
        # The simulator reads this file before each block. Its old client retains
        # the original schedule; a pass-two ID is intentionally unavailable here
        # until that client is stopped. No pass-two scored rollout can begin.
        plan['server_schedule_barrier'] = 'Pass-two jobs deferred for pi05 epoch 3.208; queue stops the old client at verified 900/model.'
        low.write(parent/'plan.json', plan)
    request = low.read(parent/'request.json')
    request.update(automatic_second_pass_authorized=False, second_pass_deferred_for=queue.KEY,
                   handoff_after_trials_per_model=900)
    low.write(parent/'request.json', request)
    request = low.read(target/'request.json')
    request.update(handoff_after_trials_per_model=900,
                   wait_for_completion=str(parent/'first_pass_handoff.json'))
    low.write(target/'request.json', request)
    queued['files_sha256'][script] = low.sha(script)
    low.write(target/'plan.json', queued)
    note = 'Queued π0.5+KR follow-up: epoch 3 + partial epoch 4 (3.208 epochs), original ungated reconstruction; 1,800 low-clutter trials after all three VLASH variants complete their first 900 trials each. VLASH repeats 11–20 are deferred; saved first-pass trials are retained.'
    table = queue.LOW/'table1_comparison.md'
    text = '\n'.join(note if line.startswith('Queued π0.5+KR follow-up:') else line
                     for line in table.read_text().splitlines()) + '\n'
    from table1_display_state import write_if_changed
    write_if_changed(table, text)
    receipt = {'state': 'armed', 'updated_utc': queue.runner.stamp(),
               'handoff_trials_per_vlash_variant': 900, 'pi05_target_trials': 1800,
               'deferred_vlash_jobs': len(plan['deferred_jobs']),
               'active_vlash_rollout_restarted': False,
               'queued_script_sha256': hashlib.sha256(Path(script).read_bytes()).hexdigest()}
    low.write(target/'first_pass_handoff_configuration.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
