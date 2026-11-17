"""Two resident policies, retaining each cohort's original sampling convention."""
from pathlib import Path

source = Path(__file__).with_name('krv_profile_ablation_worker.py').read_text()
old = 'source=base._source'
new = '''source=base._source.replace(
    'if noise is None and episode_seed is not None:',
    "if spec.get('deterministic_observation_noise', False) and noise is None and episode_seed is not None:")
needle = "            actual=compute(batch,noise)['actions'].transpose(0,1)"
check = """            rng_before = torch.cuda.get_rng_state()
            marked_batch = dict(batch, _episode_seed=937)
            sampled = compute(marked_batch)['actions']
            torch.cuda.set_rng_state(rng_before)
            if spec.get('deterministic_observation_noise', False):
                generator = torch.Generator(device='cuda').manual_seed(937*1000003)
                expected_noise = torch.randn(1,cfg.chunk_size,cfg.max_action_dim,device='cuda',generator=generator)
                reference_sample = compute(batch,expected_noise)['actions']
            else:
                reference_sample = compute(batch)['actions']
            torch.testing.assert_close(sampled,reference_sample,rtol=0,atol=0)
            torch.cuda.set_rng_state(rng_before)
"""
assert source.count(needle) == 1
source = source.replace(needle, check+needle)
source = source.replace("'same_noise_native_actions_exact':True,", "'same_noise_native_actions_exact':True,'model_sampling_contract_exact':True,")
needle = '            for _ in range(5):compute(batch,noise)'
check = """            velocity_changes_actions = None
            if spec.get('require_velocity_sensitivity', False):
                assert cfg.condition_on_object_velocity
                varied = dict(batch)
                varied['observation.environment_state'] = batch['observation.environment_state'].clone()
                varied['observation.environment_state'][...,6:9] = torch.tensor([.23,-.17,.09])
                changed = compute(varied,noise)['actions'].transpose(0,1)
                velocity_changes_actions = not torch.equal(actual,changed)
                assert velocity_changes_actions, 'Velocity-conditioned checkpoint ignored changed velocity'
"""
assert source.count(needle) == 1
source = source.replace(needle, check+needle)
source = source.replace("'reference_grad_mode':'native_no_grad',", "'reference_grad_mode':'native_no_grad','velocity_conditioning_changes_actions':velocity_changes_actions,")'''
assert source.count(old) == 1
source = source.replace(old, new)
exec(compile(source, str(Path(__file__).with_name('krv_profile_ablation_worker.py')), 'exec'))
