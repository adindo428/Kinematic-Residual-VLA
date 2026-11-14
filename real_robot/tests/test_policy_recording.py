import json
import numpy as np
import pytest
from real_robot.policy_recording import InferenceArchive


def test_lossless_recording_flushes_all_inputs_and_full_chunk(tmp_path):
    writer=InferenceArchive(tmp_path)
    arrays=dict(front=np.arange(240,dtype=np.uint8).reshape(2,4,10,3),state=np.ones((2,6)),
                actions=np.arange(140,dtype=np.float32).reshape(20,7),valid=np.array([True,False]),
                metadata=np.array(json.dumps({'trial':1,'index':12})))
    writer.submit('sample.npz',arrays);writer.close()
    assert writer.submitted==writer.completed==1
    assert not list(tmp_path.glob('*.tmp'))
    with np.load(tmp_path/'sample.npz',allow_pickle=False) as saved:
        for key,value in arrays.items():np.testing.assert_array_equal(saved[key],value)
    with pytest.raises(RuntimeError,match='closed'):writer.submit('late.npz',arrays)


def test_recording_reports_disk_failure_instead_of_silent_loss(tmp_path):
    writer=InferenceArchive(tmp_path/'missing')
    (tmp_path/'missing').rmdir()
    writer.submit('sample.npz',{'state':np.ones(6)})
    with pytest.raises(RuntimeError,match='recording failed'):writer.close()
