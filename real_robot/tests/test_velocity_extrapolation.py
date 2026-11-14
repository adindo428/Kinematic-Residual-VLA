import copy
import unittest

from real_robot.velocity_extrapolation import extrapolate_episode


def episode():
    rows=[];quality=[]
    for i in range(10):
        valid=i<3
        rows.append({'action':[0,0,0,0,0,0,1], 'observation.environment_state':[.1,.2,.3,0,0,0,.3 if valid else 0,0,0]})
        quality.append(dict(frame_index=i,anchor_monotonic_s=100+i*.04,
            velocity_age_s=0.,velocity_measurement_monotonic_s=100+i*.04,
            velocity_estimated=valid,velocity_zero_filled=not valid,velocity_repaired=False,
            velocity_required=True,position_current=valid,position_repaired=False,
            depth_position_corrected=True,phase_confirmed=False,
            manipulation_phase='open_candidate',tracking_usable=valid))
    return rows,quality


class ExtrapolationTests(unittest.TestCase):
    def test_bounded_and_no_recursive_fill(self):
        rows,q=episode();before=copy.deepcopy(rows)
        result,flags=extrapolate_episode(rows,q,.08)
        self.assertEqual(rows,before)
        self.assertEqual([i for i,f in enumerate(flags) if f.get('velocity_extrapolated')],[3,4])
        self.assertTrue(flags[3]['velocity_zero_filled']) # original flag retained
        self.assertTrue(flags[3]['velocity_repaired'])
        self.assertFalse(flags[3]['velocity_extrapolation_uses_future'])
        self.assertAlmostEqual(result[3]['observation.environment_state'][6],.3)

    def test_transition_and_confirmed_holding(self):
        rows,q=episode();rows[3]['action'][-1]=-1
        self.assertFalse(extrapolate_episode(rows,q,.08)[1][3].get('velocity_extrapolated',False))
        rows,q=episode();q[3]['phase_confirmed']=True;q[3]['velocity_required']=False
        result,flags=extrapolate_episode(rows,q,.08)
        self.assertEqual(result[3],rows[3])
        self.assertFalse(flags[3].get('velocity_extrapolated',False))

    def test_stale_position_not_made_usable(self):
        rows,q=episode();q[3]['position_current']=False
        result,flags=extrapolate_episode(rows,q,.08)
        self.assertTrue(flags[3]['velocity_extrapolated'])
        self.assertFalse(flags[3]['tracking_usable'])
        self.assertEqual(rows[3]['observation.environment_state'][:6],result[3]['observation.environment_state'][:6])

    def test_existing_repairs_and_measured_values_preserved(self):
        rows,q=episode();q[3]['velocity_repaired']=True;rows[3]['observation.environment_state'][6]=.4
        result,_=extrapolate_episode(rows,q,.08)
        self.assertEqual(result[:4],rows[:4])

    def test_unstable_motion_rejected(self):
        rows,q=episode();rows[1]['observation.environment_state'][6]=.7
        self.assertFalse(extrapolate_episode(rows,q,.08)[1][3].get('velocity_extrapolated',False))

    def test_position_fit_is_causal_and_uses_measured_positions(self):
        rows,q=episode()
        for i in range(10):
            rows[i]['observation.environment_state'][:3]=[i*.04*.3,0,0]
            rows[i]['observation.environment_state'][6:9]=[0,0,0]
            q[i].update(position_current=True,velocity_estimated=False,velocity_zero_filled=True)
        result,flags=extrapolate_episode(rows,q,.08)
        self.assertAlmostEqual(result[5]['observation.environment_state'][6],.3,places=6)
        self.assertEqual(flags[5]['velocity_extrapolation']['source_measurement_kind'],'positions')
        self.assertTrue(flags[5]['tracking_usable'])
        rows[6]['observation.environment_state'][:3]=[100,100,100]
        self.assertEqual(extrapolate_episode(rows,q,.08)[0][5],result[5])


if __name__=='__main__':unittest.main()
