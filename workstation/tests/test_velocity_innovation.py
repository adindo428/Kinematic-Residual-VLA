import ast
from collections import deque
from copy import deepcopy
from pathlib import Path
from queue import Queue
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from velocity_innovation import VelocityInnovation


def pose(gripper=1.):
    return np.array([[.3, .1, .2, 1., 0., 0., 0., gripper]], dtype=np.float32)


def test_constant_motion_zero_and_known_innovation():
    c = VelocityInnovation('innovation', .04, max_step=1.)
    for i in range(11):
        c.observe(i, [.1, 0, 0])
        c.set_origin(0)
        np.testing.assert_array_equal(c.correct(pose()), pose())
    for i in range(11, 16):
        c.observe(i, [.25, 0, 0])
        c.correct(pose())
    # Trapezoidal convention: one transition half-step + four full steps.
    assert c.offset[0] == pytest.approx(.15*.04*4.5)


def test_origin_reanchors_without_changing_prior_outputs():
    c = VelocityInnovation('innovation', .04, max_step=1.)
    for i, v in enumerate([.1, .2, .3, .4]):
        c.observe(i, [v, 0, 0])
    c.set_origin(0)
    earlier = c.correct(pose()).copy()
    c.set_origin(3)
    np.testing.assert_array_equal(c.correct(pose()), pose())
    assert earlier[0, 0] > pose()[0, 0]
    with pytest.raises(AssertionError):
        c.set_origin(4)  # Future observation forbidden.


def test_clipping_fade_and_gate_latch():
    c = VelocityInnovation('innovation', .04)
    for i in range(30):
        c.observe(i, [0 if i == 0 else 100, 100 if i else 0, 0])
        c.set_origin(0)
        original = pose(1 if i < 15 or i > 17 else -1)
        corrected = c.correct(original)
        np.testing.assert_array_equal(corrected[:, 3:], original[:, 3:])
    offsets = np.array([r['offset_m'] for r in c.rows])
    assert np.linalg.norm(offsets, axis=1).max() <= .05000001
    assert np.linalg.norm(np.diff(offsets, axis=0), axis=1).max() <= .00500001
    assert c.closed
    np.testing.assert_allclose(c.offset, 0, atol=1e-12)


def test_none_is_exact_and_naive_is_distinct():
    a = VelocityInnovation('none', .04)
    b = VelocityInnovation('naive', .04, max_step=1.)
    for i in range(5):
        for c in (a, b):
            c.observe(i, [.1, 0, 0]); c.set_origin(0)
        np.testing.assert_array_equal(a.correct(pose()), pose())
        b.correct(pose())
    assert b.offset[0] == pytest.approx(.016)


def test_streaming_metadata_preserves_native_actions():
    import torch
    from velocity_innovation_worker import streaming_with_origin
    path = Path(__file__).resolve().parents[2]/'scratch/dom45/official/source/policies/dynamicvla/modeling_dynamicvla.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DynamicVLAPolicy')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_get_streaming_action')
    namespace = dict(torch=torch, ACTION='action', logging=__import__('logging'))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    native = namespace['_get_streaming_action']
    patched = streaming_with_origin(native, namespace)
    def front():
        return SimpleNamespace(q_in={}, q_out=Queue(), config=SimpleNamespace(n_action_steps=5),
            _queues={'action':deque()}, _action_index=-1, _episode_seed=123)
    a, b = front(), front()
    for index in range(8):
        if index in (1, 3, 5):
            chunk = dict(index=index-1, actions=torch.arange(35).reshape(5, 1, 7).float()+index)
            a.q_out.put(deepcopy(chunk)); b.q_out.put(deepcopy(chunk))
        aa = native(a, {'index':index}); bb = patched(b, {'index':index})
        assert (aa is None) == (bb is None)
        if aa is not None:
            torch.testing.assert_close(aa, bb, rtol=0, atol=0)
            assert b.chunk_origin_index <= index
        assert b.q_in['obs'][0]['_episode_seed'] == 123


def test_server_patch_compiles_with_existing_hooks():
    from velocity_innovation_server import patch_source
    path = Path(__file__).resolve().parents[2]/'scratch/dom45/official/source/simulations/evaluate.py'
    code = patch_source(path.read_text())
    compile(code, str(path), 'exec')
    assert code.count('_velocity_ctrl.correct(') == 1
