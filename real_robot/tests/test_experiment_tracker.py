import json
from pathlib import Path
import tempfile
import unittest

from real_robot.experiment_tracker import (digest, refresh, scan_collections, table,
                                           tex, wilson, evaluation_summary)


class ExperimentTrackerTests(unittest.TestCase):
    def setUp(self):
        self.plan=json.loads(Path('real_robot/collection_plan.json').read_text())

    def test_pending_is_not_zero_percent_and_plan_totals_match(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            snapshot=refresh(root/'collections',root/'tracker',root/'tables')
            self.assertEqual(sum(row['target'] for row in snapshot['progress']),1600)
            self.assertEqual(sum(row['included'] for row in snapshot['progress']),0)
            self.assertIn('Pending', (root/'tables/real_robot_results.tex').read_text())

    def test_duplicate_exports_and_review_gate_and_source_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            collections=Path(d);session=collections/'test-session';raw=session/'recording'
            (raw/'episodes').mkdir(parents=True)
            manifest=raw/'manifest.json'
            manifest.write_text(json.dumps(dict(schema='dom_dynamic_capture_v1',mode='execute',task_id='blue_tape',split='train')))
            episode=raw/'episodes/episode_000000.json'
            episode.write_text(json.dumps(dict(task_id='blue_tape',split='train',outcome='success',start_monotonic_s=1,end_monotonic_s=4)))
            for name in ('dom','dom-copy'):
                dest=session/name;(dest/'meta').mkdir(parents=True)
                (dest/'export_report.json').write_text('{"complete":true}')
                (dest/'meta/sources.jsonl').write_text(json.dumps(dict(source_episode=str(episode.resolve()),split='train'))+'\n')
                (dest/'meta/raw_hashes.json').write_text(json.dumps({str(manifest.resolve()):digest(manifest),str(episode.resolve()):digest(episode)}))
            attempts,_,_=scan_collections(collections,self.plan,{})
            self.assertEqual(len(attempts),1)
            self.assertEqual(attempts[0]['exported'],1)
            self.assertEqual(attempts[0]['included'],0)
            identity=attempts[0]['attempt_id']
            with self.assertRaisesRegex(ValueError,'reviewed speed'):
                scan_collections(collections,self.plan,{identity:dict(decision='include')})
            attempts,_,_=scan_collections(collections,self.plan,{identity:dict(decision='include',speed_bin='slow')})
            self.assertEqual(attempts[0]['included'],1)
            episode.write_text(episode.read_text()+'\n')
            attempts,_,warnings=scan_collections(collections,self.plan,{})
            self.assertEqual(attempts[0]['exported'],0)
            self.assertTrue(warnings)

    def test_takeover_counts_failure_pending_excluded_and_sets_separate(self):
        def row(i,outcome,evaluation_set='test'):
            return dict(trial_id=str(i),run_id='run1',task_id='blue_tape',condition='iid_slow',
                        evaluation_set=evaluation_set,outcome=outcome)
        rows=[row(1,'success'),row(2,'takeover'),row(3,'pending'),row(4,'failure','development')]
        results=evaluation_summary(rows,{'run1':{}},self.plan)
        test=next(r for r in results if r['evaluation_set']=='test')
        self.assertEqual((test['successes'],test['trials'],test['success_rate']),(1,2,.5))
        self.assertLess(test['ci_low'],.5)
        self.assertGreater(test['ci_high'],.5)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            evaluation_summary([rows[0],rows[0]],{'run1':{}},self.plan)

    def test_tex_escaping_and_wilson_boundaries(self):
        self.assertEqual(tex('KR_1 & 50%'),r'KR\_1 \& 50\%')
        self.assertEqual(wilson(0,0),(None,None))
        self.assertAlmostEqual(wilson(0,20)[0],0)
        self.assertAlmostEqual(wilson(20,20)[1],1)
        self.assertIn(r'\begin{tabular}',table('A','tab:a',['A','B'],[['x',1]]))


if __name__=='__main__':unittest.main()
