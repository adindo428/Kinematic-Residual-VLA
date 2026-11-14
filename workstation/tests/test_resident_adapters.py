"""Resident adapter switches preserve prompt indices and reject invalid reuse."""
import ast
import json
from pathlib import Path
import queue
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resident_adapter_table1 import PendingChunk, heaviest_set
from resident_low_clutter_server import prompt, resident
from run_resident_adapter_table1 import validate_rows, training_checkpoint_allowed


def test_partial_training_exception_is_limited_to_explicit_checkpoint():
    policy={'minimum_completed_epochs':10,'authorized_exceptions':{'pi05_kr':{'user_authorized':True,'checkpoint_sha256':'approved'}}}
    model={'key':'pi05_kr','entry':{'progress':{'epoch':6},'checkpoint_sha256':'approved'}}
    assert training_checkpoint_allowed(model,policy)
    model['entry']['checkpoint_sha256']='replacement'
    assert not training_checkpoint_allowed(model,policy)
    model['entry']['progress']['epoch']=10
    assert training_checkpoint_allowed(model,policy)


def test_partial_training_checkpoint_remains_held_without_authorization():
    model={'key':'pi05_kr','entry':{'progress':{'epoch':6},'checkpoint_sha256':'approved'}}
    assert not training_checkpoint_allowed(model,{'minimum_completed_epochs':10})


def test_pending_chunk_is_not_lost_by_repeated_ready_checks():
    output = queue.Queue(); pending = PendingChunk(output)
    assert not pending.done()
    value = (2, 17, 'physical actions', .3); output.put(value)
    assert pending.done() and pending.done() and pending.result() == value
    assert output.empty()


def test_capacity_qualification_tests_the_heaviest_idle_models():
    keys = ['smol', 'vlash', 'pi_base', 'pi_kr']
    receipts = {k: {'gpu_allocated': n} for k, n in zip(keys, [1, 7, 14, 14])}
    assert set(heaviest_set('vlash', keys, receipts, 3)) == {'vlash', 'pi_base', 'pi_kr'}
    assert set(heaviest_set('pi_kr', keys, receipts, 2)) == {'pi_base', 'pi_kr'}


def test_frozen_prompt_uses_global_repeat_and_restarts_for_next_checkpoint(monkeypatch):
    from collections import Counter
    cfg = {'evaluation_case_id': 'avocado', 'evaluation_prompts': [f'prompt {i}' for i in range(20)]}
    monkeypatch.setattr(resident, 'CURRENT', {'repeat_offset': 10})
    monkeypatch.setattr(resident, 'COUNTS', Counter(avocado=3))
    assert prompt(None, cfg) == 'prompt 13'
    resident.COUNTS.clear()
    assert prompt(None, cfg) == 'prompt 10'


def test_reused_rows_require_matching_frozen_prompt_and_physical_time(tmp_path):
    scene = tmp_path/'scene.json'; scene.write_text(json.dumps({'evaluation_prompts': [f'prompt {i}' for i in range(20)]}))
    trajectory = tmp_path/'ee.npz'; trajectory.touch()
    test = {'name': 'avocado', 'path': str(scene), 'horizon_seconds': 12}
    raw = [{'env_name': 'avocado', 'trial': 1, 'status': 0, 'success': True, 'n_steps': 10, 'step_dt_s': .04, 'task_time_s': .4}]
    metrics = [{'trial': 1, 'status': 0, 'trajectory_path': str(trajectory)}]
    trace = [{'trial': 1, 'text': 'prompt 10'}]
    validate_rows(raw, trace, metrics, test, 10)
    with pytest.raises(AssertionError): validate_rows(raw, trace, metrics, test, 0)
    raw[0]['task_time_s'] = .8
    with pytest.raises(AssertionError): validate_rows(raw, trace, metrics, test, 10)


def test_low_clutter_server_hook_retains_native_scene_cache_and_physics():
    root = Path(__file__).resolve().parents[2]
    text = (root/'scratch/dom45/official/source/simulations/evaluate.py').read_text()
    patched = resident.patch_source(text)
    functions = lambda x: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(x).body if isinstance(n, ast.FunctionDef)}
    a, b = functions(text), functions(patched)
    for key in ['get_sim_results', 'get_test_env', 'get_latest_action', '_get_action_tensor']: assert a[key] == b[key]
    assert patched.count("__import__('table1_video_sampling').should_save") == 1
    for needle in ['    rcv_action = False', '            object_state=env.unwrapped.scene["object"].data,',
                   '                **{"observation.images.%s" % k: v["rgb"] for k, v in cam_view.items()},',
                   '    instruction = InstructionGenerator.generate_instruction(env_cfg["instruction"])']:
        assert patched.count(needle) == 1
