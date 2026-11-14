"""Selection leakage, duplicate integrity, quotas, and sampler budget contracts."""
import copy
import unittest

from budget import advance, lr_factor, validate_budget, validate_progress
from subset import allocate, build_subset


class SubsetContracts(unittest.TestCase):
    def setUp(self):
        self.rows = []
        self.split = {'splits': {'train': [], 'validation': [], 'test': []}}
        for task in ('pick', 'place', 'long_horizon'):
            for speed in (1, 2):
                for group in range(12):
                    for duplicate in range(2 if group % 3 == 0 else 1):
                        ep = len(self.rows)
                        self.rows.append({'episode_index': ep, 'task': task, 'speed_bin': speed,
                            'duplicate_id': f'{task}:{speed}:{group}', 'length': 30 + ep})
                        self.split['splits']['train'].append(ep)
        for name in ('validation', 'test'):
            ep = len(self.rows)
            self.rows.append({'episode_index': ep, 'task': 'pick', 'speed_bin': 2,
                              'duplicate_id': name, 'length': 50})
            self.split['splits'][name].append(ep)

    def test_deterministic_exact_quotas_and_whole_duplicates(self):
        quotas = {'pick': 11, 'place': 9, 'long_horizon': 5}
        split, report, rows = build_subset(self.rows, self.split, quotas)
        reversed_result = build_subset(list(reversed(self.rows)), self.split, quotas)
        self.assertEqual(split, reversed_result[0])
        self.assertEqual(report['train_episodes'], 25)
        self.assertEqual(report['train_anchors'], sum(r['length'] for r in rows))
        self.assertEqual(split['splits']['validation'], self.split['splits']['validation'])
        self.assertEqual(split['splits']['test'], self.split['splits']['test'])
        ids = set(split['splits']['train'])
        selected_groups = {r['duplicate_id'] for r in rows}
        self.assertTrue(all(r['episode_index'] in ids for r in self.rows if r['duplicate_id'] in selected_groups))
        for task, count in quotas.items():
            self.assertEqual(sum(r['task'] == task for r in rows), count)
        self.assertNotEqual(ids, set(build_subset(self.rows, self.split, quotas, seed=43)[0]['splits']['train']))

    def test_reject_cross_partition_duplicates_and_overlap(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]['duplicate_id'] = rows[0]['duplicate_id']
        with self.assertRaisesRegex(ValueError, 'crosses partitions'):
            build_subset(rows, self.split, {'pick': 10})
        overlap = copy.deepcopy(self.split)
        overlap['splits']['test'].append(0)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            build_subset(self.rows, overlap, {'pick': 10})

    def test_explicit_task_exclusion_and_capacity_failure(self):
        _, report, rows = build_subset(self.rows, self.split, {'pick': 10, 'place': 10, 'long_horizon': 0})
        self.assertEqual(report['excluded_task_families'], ['long_horizon'])
        self.assertFalse(any(r['task'] == 'long_horizon' for r in rows))
        with self.assertRaisesRegex(ValueError, 'capacity'):
            build_subset(self.rows, self.split, {'pick': 1000})

    def test_capped_speed_allocation(self):
        self.assertEqual(allocate({0: 2, 1: 100, 2: 100}, 42), {0: 2, 1: 20, 2: 20})
        allocated = allocate({0: 80, 1: 1400, 2: 43000, 3: 71000}, 9000)
        self.assertEqual(sum(allocated.values()), 9000)
        self.assertEqual(allocated[0], 80)
        self.assertGreater(allocated[1], 9000 * 1400 / 115480)


class BudgetContracts(unittest.TestCase):
    def recipe(self, batch):
        return {'training_anchors': 137, 'global_batch': batch, 'min_epochs': 5,
                'steps': 5 * (137 // batch), 'schedule_steps': 5 * (137 // batch),
                'schedule_policy': 'fresh_subset_cosine', 'warmup_updates': 2,
                'initialization': 'released_pretrained', 'lr': 1e-4, 'decay_lr': 2.5e-6}

    def test_complete_epochs_and_exact_continuation(self):
        for batch in (32, 64):
            recipe = self.recipe(batch)
            validate_budget(recipe, 137)
            p = {'updates': 0, 'samples': 0, 'epoch': 0, 'cursor': 0}
            validate_progress(p, recipe)
            for step in range(recipe['steps']):
                advance(p, recipe)
                if step == 2:
                    p = copy.deepcopy(p)  # serialized checkpoint continuation
            self.assertEqual((p['epoch'], p['cursor']), (5, 0))
            self.assertEqual(p['samples'], 5 * 128)
            self.assertAlmostEqual(lr_factor(recipe['steps'], recipe), .025)
            self.assertEqual(lr_factor(0, recipe), .5)
            with self.assertRaises(ValueError):
                advance(p, recipe)

    def test_reject_full_data_counts_schedule_or_initialization(self):
        r = self.recipe(32)
        for patch in ({'training_anchors': 22370201}, {'schedule_steps': 30000},
                      {'initialization': 'full_dom_checkpoint'}):
            with self.assertRaises(ValueError):
                validate_budget({**r, **patch}, 137)


if __name__ == '__main__':
    unittest.main()
