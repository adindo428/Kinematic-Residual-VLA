"""Select figure videos after scoring, without changing trial execution."""
import hashlib
import json
import os
from pathlib import Path

_FIRST_SUCCESS = set()
_HISTORY_LOADED = False
POLICY = {
    'name': 'first_trial_and_first_success_per_configuration',
    'maximum_videos_per_configuration': 2,
    'maximum_videos_per_1800_trial_suite': 180,
    'selection': 'Always save repeat 1; also save the first successful repeat if different.',
    'scope': 'Only post-episode frame conversion and video encoding. All trial scores and trajectories are retained.',
    'qualitative_sampling': 'Includes an outcome-selected success; selected videos are not an unbiased performance sample.',
}


def should_save(env_name, trial, result):
    global _HISTORY_LOADED
    if not _HISTORY_LOADED:
        history=os.environ.get('DOM_VIDEO_SELECTION_HISTORY')
        if history:
            for line in Path(history).read_text().splitlines():
                row=json.loads(line)
                if row['status']==0:_FIRST_SUCCESS.add(row['env_name'])
        _HISTORY_LOADED=True
    assert isinstance(trial, int) and trial >= 1
    trial+=int(os.environ.get('DOM_REPEAT_OFFSET','0'))
    status = int(result['status'])
    assert status in (0, 1), 'No-action episodes must not enter scored video selection'
    first_success = status == 0 and env_name not in _FIRST_SUCCESS
    if status == 0:
        _FIRST_SUCCESS.add(env_name)
    save = trial == 1 or first_success
    reason = 'first_trial' if trial == 1 else 'first_success' if first_success else 'not_selected'
    destination = Path(os.environ['DOM_TRIAL_RESULTS']).with_name('video_selection.jsonl')
    with destination.open('a') as stream:
        stream.write(json.dumps({'env_name': env_name, 'trial': trial, 'status': status,
            'save_video': save, 'reason': reason, 'policy': POLICY['name']}) + '\n')
    return save


def patch_source(text):
    before = '                if args.save:'
    assert text.count(before) == 1, 'Pinned post-episode video branch changed'
    after = "                if args.save and __import__('table1_video_sampling').should_save(env_name, n_tests, sim_results):"
    return text.replace(before, after)


def wrap_command(command, folder):
    command = list(command)
    position = command.index('-p') + 1
    wrapped = Path(command[position])
    base = Path(__file__).resolve().parent
    assert wrapped.name in ('collection_matched_server.py', 'low_clutter_table1_server.py')
    assert '--save' in command
    command[position] = str(base/'sampled_video_table1_server.py')
    command += ['--wrapped-server', str(wrapped)]
    paths = [Path(__file__), base/'sampled_video_table1_server.py', wrapped]
    receipt = {**POLICY, 'wrapped_server': str(wrapped),
        'files_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (Path(folder)/'video_policy.json').write_text(json.dumps(receipt, indent=2)+'\n')
    return command
