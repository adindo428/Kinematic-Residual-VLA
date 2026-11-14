"""Tracking repairs and pose targets must never invent trustworthy stationary data."""
import importlib.util
import ast
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from real_robot.dataset_quality import repair_tracking, phases, session_splits

path=Path(__file__).resolve().parents[2]/'third_party/DynamicVLA/utils/real_robot_contract.py'
spec=importlib.util.spec_from_file_location('real_robot_contract',path)
contract=importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def fixture():
    rows=[];quality=[]
    for i in range(6):
        rows.append({'action':[0,0,0,0,0,0,1], 'observation.state':[0]*6,
                     'observation.environment_state':[i*.04,0,0,0,0,0,1,0,0]})
        quality.append(dict(frame_index=i,anchor_monotonic_s=i*.04,
                            position_age_s=0,position_current=True,velocity_estimated=True,velocity_zero_filled=False))
    return rows,quality


class DatasetQualityTests(unittest.TestCase):
    def test_short_bracket_repair_deduplicates_held_measurement(self):
        rows,q=fixture()
        for i in [1,2]:
            rows[i]['observation.environment_state'][:3]=[0,0,0]
            rows[i]['observation.environment_state'][6:9]=[0,0,0]
            q[i].update(position_age_s=i*.04,position_current=False,velocity_estimated=False,velocity_zero_filled=True)
        result,quality=repair_tracking(rows,q)
        np.testing.assert_allclose(result[1]['observation.environment_state'][:3],[.04,0,0])
        np.testing.assert_allclose(result[2]['observation.environment_state'][6:9],[1,0,0])
        self.assertTrue(quality[1]['tracking_repair_uses_future'])
        self.assertTrue(quality[1]['velocity_zero_filled'])  # original provenance survives
        self.assertFalse(q[1].get('position_repaired',False))

    def test_no_long_gap_extrapolation_or_gripper_boundary(self):
        rows,q=fixture()
        for i in range(1,6):
            q[i].update(position_age_s=i*.04,position_current=False,velocity_estimated=False,velocity_zero_filled=True)
        _,quality=repair_tracking(rows,q)
        self.assertFalse(any(v['tracking_usable'] for v in quality[1:]))
        q[-1].update(position_age_s=0,position_current=True,velocity_estimated=True,velocity_zero_filled=False)
        _,quality=repair_tracking(rows,q,max_gap_s=.1)
        self.assertFalse(quality[2]['tracking_usable'])
        rows[3]['action'][-1]=-1
        _,quality=repair_tracking(rows,q,max_gap_s=.5)
        self.assertFalse(quality[2]['tracking_usable'])

    def test_lift_marker_is_not_visual_confirmation(self):
        rows,q=fixture()
        labels,required=phases(rows,q,{'lift':.08})
        self.assertTrue(required.all())
        self.assertEqual(labels[2],'marked_lift_candidate')
        labels,required=phases(rows,q,held_intervals=[[2,5]])
        self.assertEqual(required.tolist(),[True,True,False,False,False,True])

    def test_confirmed_hold_uses_neutral_input_without_claiming_measurement(self):
        rows,q=fixture()
        q[2].update(position_current=False,velocity_estimated=False,velocity_zero_filled=True)
        required=np.ones(6,bool);required[2]=False
        rows,quality=repair_tracking(rows,q,required=required)
        self.assertTrue(quality[2]['tracking_usable'])
        self.assertEqual(rows[2]['observation.environment_state'][6:9],[0,0,0])
        self.assertEqual(quality[2]['velocity_input_source'],'neutral_after_confirmed_lift')

    def test_wrapped_training_and_deployment_are_rotation_equivalent(self):
        state=torch.tensor([0.,0,0,2*np.pi-.01,.2,.01])
        action=torch.tensor([[.1,0,0,.01,.3,2*np.pi-.01,-1.]])
        delta=contract.delta_pose(action,state,'wrapped_xyz_v1')
        self.assertLess(float(delta[...,3:6].abs().max()),.11)
        restored=contract.absolute_pose(delta,state,'wrapped_xyz_v1')
        torch.testing.assert_close(restored,action,atol=1e-6,rtol=1e-6)
        self.assertGreater(float(contract.delta_pose(action,state)[0,3].abs()),6)

    def test_filter_preserves_frame_timeline_and_checks_history(self):
        self.assertEqual(contract.usable_anchor_indices([7],[6],{'7':[0,1,4,5]},[-1,0]),[0,1,5])

    def test_empty_explicit_holdout_never_uses_fallback(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'meta').mkdir()
            (root/'meta/splits.json').write_text(json.dumps(dict(train=[0,1],test=[],validation=[])))
            with self.assertRaisesRegex(ValueError,'no test episodes'):
                contract.explicit_split(root,'test',2)

    def test_session_split_needs_three_sessions_per_task(self):
        with self.assertRaisesRegex(ValueError,'at least three sessions'):
            session_splits([dict(raw_session='one')],[dict(episode_index=0,tasks=['orange'])])

    def test_live_quality_policy_rejects_missing_pregrasp_and_neutralizes_hold(self):
        v=torch.ones(2,3)
        with self.assertRaisesRegex(ValueError,'quality flags'):
            contract.checked_velocity(v,{},require_quality=True)
        batch={'dynamic_object.velocity_required':torch.tensor([[True,True],[True,False]]),
               'dynamic_object.velocity_valid':torch.tensor([[True,True],[True,False]])}
        torch.testing.assert_close(contract.checked_velocity(v,batch,True),torch.tensor([[1.,1,1],[0.,0,0]]))
        batch['dynamic_object.velocity_valid'][0,-1]=False
        with self.assertRaisesRegex(ValueError,'unavailable'):
            contract.checked_velocity(v,batch,True)

    def test_confirmed_hold_disables_even_centered_geometric_drift(self):
        source=path.parents[1]/'policies/dynamicvla/modeling_dynamicvla.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DynamicVLAPolicy')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_phase_gate')
        ns={'torch':torch,'OBS_STATE':'observation.state'}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),'exec'),ns)
        policy=SimpleNamespace(config=SimpleNamespace(residual_phase_gate_mode='constant',chunk_size=20))
        batch={'observation.state':torch.zeros(2,2,6),
               'dynamic_object.velocity_required':torch.tensor([[True,True],[True,False]])}
        gate=ns['prepare_phase_gate'](policy,batch,torch.device('cpu'))
        self.assertEqual(tuple(gate.shape),(2,20,1))
        self.assertTrue((gate[0]==1).all())
        self.assertTrue((gate[1]==0).all())


if __name__=='__main__':
    unittest.main()
