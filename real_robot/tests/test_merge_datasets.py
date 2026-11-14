import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq

from real_robot.dom_dataset import export_sessions,sha256,validate_dataset,write_parquet
from real_robot.merge_datasets import merge_datasets
from real_robot.tests.test_dom_collection import make_session


class MergeDatasetsTests(unittest.TestCase):
    def test_bottle_exports_pick_only_metadata_and_preserves_existing_place_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);make_session(root/'a')
            manifest,episode=make_session(root/'bottle',offset=200)
            plan=json.loads(Path('real_robot/collection_plan.json').read_text())
            manifest.update(task_id='blue_bottle',task=next(t for t in plan['tasks'] if t['id']=='blue_bottle'))
            episode['task_id']='blue_bottle'
            (root/'bottle/manifest.json').write_text(json.dumps(manifest))
            (root/'bottle/episodes/episode_000000.json').write_text(json.dumps(episode))
            export_sessions([root/'a'],root/'base');export_sessions([root/'bottle'],root/'new')
            merge_datasets([root/'base',root/'new'],root/'merged',append=True)
            tasks=[json.loads(json.loads(l)['task']) for l in (root/'merged/meta/tasks.jsonl').read_text().splitlines()]
            self.assertEqual(tasks[0]['task'],'place')
            self.assertEqual(tasks[0]['containers'],'box')
            self.assertEqual(tasks[1],dict(task='pick',objects='rolling blue bottle',instruction='Pick up the rolling blue bottle.'))
            self.assertEqual(validate_dataset(root/'merged')['episodes'],2)

    def test_marker_export_appends_new_instruction_without_relabeling_existing_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            make_session(root/'a')
            manifest,episode=make_session(root/'marker',offset=200)
            plan=json.loads(Path('real_robot/collection_plan.json').read_text())
            manifest.update(task_id='marker',object_instance='marker',task=next(t for t in plan['tasks'] if t['id']=='marker'))
            episode['task_id']='marker'
            (root/'marker/manifest.json').write_text(json.dumps(manifest))
            (root/'marker/episodes/episode_000000.json').write_text(json.dumps(episode))
            export_sessions([root/'a'],root/'base')
            export_sessions([root/'marker'],root/'new')
            merge_datasets([root/'base',root/'new'],root/'merged',append=True)
            self.assertEqual(validate_dataset(root/'merged')['episodes'],2)
            tasks=[json.loads(l) for l in (root/'merged/meta/tasks.jsonl').read_text().splitlines()]
            self.assertEqual(json.loads(tasks[0]['task'])['objects'],'rolling blue tape')
            self.assertEqual(json.loads(tasks[1]['task'])['objects'],'moving green marker')
            for ep,task in [(0,0),(1,1)]:
                rows=pq.read_table(root/f'merged/data/chunk-000/episode_{ep:06d}.parquet').to_pylist()
                self.assertTrue(all(r['task_index']==task for r in rows))

    def test_reconcile_stale_base_preserves_published_prefix_and_deduplicates_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for i,name in enumerate(('a','b','c')):
                make_session(root/name,split='test' if name=='b' else 'train',offset=100*(i+1))
                export_sessions([root/name],root/('d'+name))
            merge_datasets([root/'da',root/'db'],root/'published',append=True)
            merge_datasets([root/'da',root/'dc'],root/'stale',append=True)
            result=merge_datasets([root/'published',root/'stale'],root/'reconciled',append=True)
            self.assertEqual(result['episodes'],3)
            self.assertEqual(sum(x['exported'] for x in result['attempts']),3)
            self.assertEqual(validate_dataset(root/'reconciled')['splits'],{'train':2,'validation':0,'test':1})
            for path in (root/'published').rglob('*'):
                if path.is_file() and path.relative_to(root/'published').parts[0] in ('data','videos','quality'):
                    self.assertEqual(sha256(path),sha256(root/'reconciled'/path.relative_to(root/'published')))
            for name in ('sources','episodes','episodes_stats','camera','tasks'):
                previous=(root/f'published/meta/{name}.jsonl').read_text().splitlines()
                current=(root/f'reconciled/meta/{name}.jsonl').read_text().splitlines()
                self.assertEqual(current[:len(previous)],previous)
            merge_datasets([root/'reconciled',root/'stale'],root/'retried',append=True)
            self.assertEqual(validate_dataset(root/'retried')['episodes'],3)
            quality=root/'stale/quality/episode_000000.jsonl'
            original=quality.read_bytes();quality.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError,'changed video or quality data'):
                merge_datasets([root/'published',root/'stale'],root/'changed_quality',append=True)
            quality.write_bytes(original)
            # A matching source path alone must not hide edited trajectory data.
            path=root/'stale/data/chunk-000/episode_000000.parquet'
            rows=pq.read_table(path).to_pylist();rows[0]['action'][0]+=.1;write_parquet(rows,path)
            with self.assertRaisesRegex(ValueError,'changed numeric data'):
                merge_datasets([root/'published',root/'stale'],root/'tampered',append=True)

    def test_merge_preserves_video_and_actions_updates_indices_and_split_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            make_session(root/'a',split='train')
            make_session(root/'b',split='test',offset=200)
            export_sessions([root/'a'],root/'da')
            export_sessions([root/'b'],root/'db')
            report=merge_datasets([root/'db',root/'da'],root/'merged')
            result=validate_dataset(root/'merged')
            self.assertEqual(result['splits'],{'train':1,'validation':0,'test':1})
            self.assertEqual(report['episodes'],2)
            old=pq.read_table(root/'db/data/chunk-000/episode_000000.parquet').to_pylist()
            new=pq.read_table(root/'merged/data/chunk-000/episode_000001.parquet').to_pylist()
            np.testing.assert_array_equal([r['action'] for r in old],[r['action'] for r in new])
            self.assertTrue(all(r['episode_index']==1 for r in new))
            self.assertGreater(new[0]['index'],0)
            before=root/'db/videos/chunk-000/observation.images.opst_cam/episode_000000.mp4'
            after=root/'merged/videos/chunk-000/observation.images.opst_cam/episode_000001.mp4'
            self.assertEqual(sha256(before),sha256(after))
            stats=[json.loads(s) for s in (root/'merged/meta/episodes_stats.jsonl').read_text().splitlines()]
            self.assertEqual(stats[1]['stats']['episode_index']['mean'],[1.])
            self.assertEqual(stats[1]['stats']['index']['min'],[new[0]['index']])
            with self.assertRaisesRegex(ValueError,'Repeated source'):
                merge_datasets([root/'da',root/'da'],root/'duplicates')


if __name__=='__main__':unittest.main()
