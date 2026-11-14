import argparse,json
from pathlib import Path
import numpy as np
import pytest
from real_robot.object_estimation import add_object_estimation_arguments,resolve_model
from real_robot.policy_stream import LiveObjectEstimator
from real_robot.tests.test_policy_stream import occlusion_record


def dataset(root,static):
    (root/'meta').mkdir(parents=True)
    (root/'meta/episodes.jsonl').write_text(json.dumps(dict(episode_index=0,tasks=[json.dumps(dict(objects='rolling orange'))]))+'\n')
    (root/'meta/sources.jsonl').write_text(json.dumps(dict(episode_index=0,static_object_collection={'declared':True} if static else None))+'\n')
    return root


def test_shared_defaults_resolve_static_and_moving_without_changing_measured_inputs(tmp_path):
    parser=argparse.ArgumentParser();add_object_estimation_arguments(parser);args=parser.parse_args([])
    assert args.object_prediction_max_gap_s==2 and args.object_prediction_max_displacement_m==.05
    assert resolve_model(dataset=dataset(tmp_path/'static',True),task='orange')=='contact_stop_assumption'
    assert resolve_model(dataset=dataset(tmp_path/'moving',False),task='orange')=='constant_velocity'
    assert resolve_model()=='constant_velocity'
    assert resolve_model(motion='moving',dataset=tmp_path/'static',task='orange')=='constant_velocity'
    assert resolve_model(model='constant_velocity',motion='static')=='constant_velocity'
    for motion in ['static','moving']:
        estimator=LiveObjectEstimator(max_gap_s=2.,motion_model=resolve_model(motion=motion))
        measured=estimator.read(occlusion_record(1.,velocity=(.1,0,0)),1.02,'r')
        np.testing.assert_allclose(measured[6:],[.1,0,0])


def test_static_profile_bridges_arm_occlusion_and_moving_profile_keeps_velocity():
    static=LiveObjectEstimator(max_gap_s=2.,motion_model=resolve_model(motion='static'))
    static.read(occlusion_record(1.,velocity=(0,0,0)),1.02,'r')
    prediction=static.read(occlusion_record(2.8,missing=True),2.82,'r')
    np.testing.assert_allclose(prediction[:3],[.3,0,.1]);np.testing.assert_allclose(prediction[6:],0)
    moving=LiveObjectEstimator(max_gap_s=2.,motion_model=resolve_model(motion='moving'))
    moving.read(occlusion_record(1.,velocity=(.1,0,0)),1.02,'r')
    prediction=moving.read(occlusion_record(1.3,missing=True),1.32,'r')
    np.testing.assert_allclose(prediction[:3],[.33,0,.1]);np.testing.assert_allclose(prediction[6:],[.1,0,0])
    with pytest.raises(ValueError):moving.read(occlusion_record(1.8,missing=True),1.82,'r')
    # A fresh genuine observation immediately replaces either hypothesis.
    recovered=moving.read(occlusion_record(1.9,position=(.36,0,.1),velocity=(.2,0,0)),1.92,'r')
    np.testing.assert_allclose(recovered[:3],[.36,0,.1]);assert not moving.diagnostics['predicted']
