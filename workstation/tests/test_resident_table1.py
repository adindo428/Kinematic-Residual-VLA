"""Resident switching preserves scored code and fences in-flight predictions."""
import ast
import multiprocessing as mp
from pathlib import Path
import queue
import sys
import threading

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from resident_table1_worker import session_loop
from resident_table1_server import patch_source
from causal_gate_diagnostic_server import patch_source as original_observation_patch
from resident_table1_client import patch_run_tests

REPO=Path(__file__).resolve().parents[2]

def functions(text):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}

def test_simulation_reset_action_and_scoring_code_preserved():
    original=(REPO/'scratch/dom45/official/source/simulations/evaluate.py').read_text()
    old=functions(original);new=functions(patch_source(original))
    for name in ('get_sim_results','get_test_env','get_latest_action','_get_action_tensor'):
        assert new[name]==old[name]
    assert new['simulate']==functions(original_observation_patch(original))['simulate']

def test_only_post_episode_cooldown_changes_in_client():
    text=(REPO/'scratch/dom45/training/krv100-full-deltaai-20261106/source/scripts/inference.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='run_tests')
    expected=ast.get_source_segment(text,node).replace('time.sleep(10)','_resident_fence()')
    assert functions(patch_run_tests(text))==functions(expected)
    assert functions(patch_run_tests(text))['run_tests']!=ast.dump(node,include_attributes=False)

def test_fence_waits_for_inflight_chunk_and_clears_stale_output():
    manager=mp.get_context('spawn').Manager()
    parent,child=mp.Pipe();inbox=manager.dict();output=manager.Queue(maxsize=1)
    started=threading.Event();release=threading.Event()
    def compute(batch,noise):
        started.set();assert release.wait(2)
        return {'index':batch['index'],'model':'A'}
    thread=threading.Thread(target=session_loop,args=(inbox,output,child,compute),daemon=True);thread.start()
    parent.send({'kind':'activate','log':None});assert parent.recv()['active']
    inbox['obs']=({'index':99},None);assert started.wait(2)
    parent.send({'kind':'idle'});assert not parent.poll(.05)
    release.set();assert parent.poll(2) and not parent.recv()['active']
    assert output.empty() and not inbox
    parent.send({'kind':'activate','log':None});assert parent.recv()['active']
    parent.send({'kind':'idle'});assert not parent.recv()['active']
    parent.send({'kind':'close'});thread.join(2);assert not thread.is_alive()
    manager.shutdown()
