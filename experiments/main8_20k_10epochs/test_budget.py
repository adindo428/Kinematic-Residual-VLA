import unittest
from budget import initial_progress, validate_budget, validate_progress, advance, lr_factor


class ContinuationBudget(unittest.TestCase):
    def recipe(self):
        return {'method':'pi05','training_anchors':137,'training_episodes':20000,
                'parent_updates':60004,'parent_samples':1920128,'global_batch':32,
                'min_epochs':10,'steps':60044,'initialization':'own_verified_checkpoint',
                'schedule_policy':'preserve_parent_schedule','schedule_steps':30000,
                'lr':2.5e-5,'decay_lr':2.5e-6}
    def test_ten_new_epochs_exclude_prior_exposure(self):
        r=self.recipe();validate_budget(r,137);p=initial_progress(r)
        for _ in range(40):advance(p,r)
        self.assertEqual((p['epoch'],p['cursor'],p['updates']),(10,0,60044))
        self.assertEqual(p['samples']-r['parent_samples'],1280)
        with self.assertRaises(ValueError):advance(p,r)
    def test_original_scheduler_is_at_floor_without_new_warmup(self):
        r=self.recipe()
        self.assertAlmostEqual(lr_factor(r['parent_updates'],r),.1)
        self.assertAlmostEqual(lr_factor(r['steps'],r),.1)
    def test_old_sample_cursor_cannot_be_used_with_new_subset(self):
        r=self.recipe();p=initial_progress(r);p['cursor']=1920128
        with self.assertRaises(ValueError):validate_progress(p,r)
    def test_short_budget_is_rejected(self):
        r=self.recipe();r['min_epochs']=5
        with self.assertRaises(ValueError):validate_budget(r,137)


if __name__=='__main__':unittest.main()
