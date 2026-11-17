"""Existing metadata client with model-specific sampling in a shared resident pool."""
from pathlib import Path

source = Path(__file__).with_name('krv_profile_ablation_client.py').read_text()
old = 'from krv_profile_ablation_worker import ResidentPool,minimum_retention'
assert source.count(old) == 1
source = source.replace(old, 'from dom_krv_scene_worker import ResidentPool,minimum_retention')
exec(compile(source, str(Path(__file__).with_name('krv_profile_ablation_client.py')), 'exec'))
