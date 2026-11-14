"""Freeze the original 20k recipe with a current-speed-only KR gate."""
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT/'scratch/dom45/training/krv-speed-gate-old20k-20261112'
PRIOR = ROOT/'scratch/dom45/training/krv100-fresh20k-h200-20261029'

def replace(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (path, old)
    path.write_text(text.replace(old, new))

def main():
    if (STAGE/'plan.json').exists():
        raise RuntimeError('Already prepared; preserve frozen run')
    STAGE.mkdir(parents=True, exist_ok=True)
    source = STAGE/'source'
    shutil.copytree(PRIOR/'source', source, ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    shutil.copyfile(ROOT/'scratch/dom45/training/krv100-full-20261010/source/core/dom_distributed_profiling.py', source/'core/dom_distributed_profiling.py')
    sys.path.insert(0, str(source))
    from utils.dom_experiment import atomic_json, digest, file_digest, source_digest
    family = STAGE/'family'
    family.mkdir(exist_ok=True)
    manifest = json.loads((PRIOR/'family/manifest.json').read_text())
    for name, sha in manifest['artifacts'].items():
        src = PRIOR/'family'/name
        if name == 'initialization/config.json':
            src = ROOT/'scratch/dom45/training/fresh20k-20261025/family'/name
        assert file_digest(src) == sha, name
        dst = family/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    weights = ROOT/'scratch/dom45/training/fresh20k-20261025/family/initialization/model.safetensors'
    assert file_digest(weights) == manifest['weights_sha256']
    (family/'initialization/model.safetensors').symlink_to(weights)
    data = ROOT/'scratch/dom45/cache/huggingface/lerobot/hzxie/DOM'
    info = json.loads((data/'meta/info.json').read_text())
    split = json.loads((family/'split.json').read_text())['splits']['train']
    hashes = json.loads((family/'parquet_hashes.json').read_text())
    def scan(ep):
        path = data/info['data_path'].format(episode_chunk=ep//info['chunks_size'], episode_index=ep)
        assert file_digest(path) == hashes[str(ep)], ep
        v = np.asarray(pq.read_table(path, columns=['observation.environment_state'])[0].to_pylist(), dtype=np.float64)[:,6:9]
        assert np.isfinite(v).all()
        return np.linalg.norm(v, axis=1)
    with ThreadPoolExecutor(max_workers=8) as pool:
        speeds = []
        for i, v in enumerate(pool.map(scan, split)):
            speeds.append(v)
            if i % 2500 == 0: print('Verified training episodes', i+1, flush=True)
    speeds = np.concatenate(speeds)
    assert len(split) == 20000 and len(speeds) == manifest['train_frames']
    maximum = float(speeds.max()); threshold = .05*maximum
    assert maximum > 0
    gate = dict(reference_speed_m_s=maximum, threshold_m_s=threshold, fraction=.05,
                enabled='current_speed >= threshold', reference='maximum L2 velocity across original train frames only',
                train_enabled_fraction=float((speeds >= threshold).mean()),
                speed_quantiles=dict(zip(['p50','p90','p95','p99','p999'],np.quantile(speeds,[.5,.9,.95,.99,.999]).tolist())),
                split_sha256=manifest['split_sha256'], frames=len(speeds))
    atomic_json(family/'speed_gate.json', gate)
    config = source/'policies/dynamicvla/configuration_dynamicvla.py'
    replace(config, '    residual_phase_gate_mode: str = "constant"', '    residual_speed_threshold: float = 0.0\n    residual_phase_gate_mode: str = "constant"')
    replace(config, 'not in {"constant", "causal_rules"}', 'not in {"constant", "causal_rules", "speed_threshold"}')
    replace(config, '        from proactive_vla.phase_gate import PhaseGateConfig', '        if self.residual_phase_gate_mode == "speed_threshold" and not self.residual_speed_threshold > 0:\n            raise ValueError("Speed gate needs a positive fixed training threshold")\n        from proactive_vla.phase_gate import PhaseGateConfig')
    modeling = source/'policies/dynamicvla/modeling_dynamicvla.py'
    replace(modeling, '    def prepare_phase_gate(self, batch, device):\n', '''    def prepare_phase_gate(self, batch, device):
        if self.config.residual_phase_gate_mode == "speed_threshold":
            # Current unnormalized observed velocity only; no pose, future or gripper input.
            velocity = self.prepare_residual_velocity(batch, device, torch.float32)
            active = (torch.linalg.vector_norm(velocity, dim=-1) >= self.config.residual_speed_threshold)
            return active.to(torch.float32)[:, None, None].expand(-1, self.config.chunk_size, 1)
''')
    training = source/'core/dom_training.py'
    replace(training, 'phase_cache=root/"phase_cache", frozen_files=root', 'phase_cache=None, frozen_files=root')
    replace(training, '"residual_phase_gate_mode": phase,', '"residual_phase_gate_mode": "speed_threshold" if residual else phase,\n        "residual_speed_threshold": manifest["speed_gate"]["threshold_m_s"],')
    replace(training, 'for phase in torch.unique(batch["dynamic_object.phase_id"]).tolist():', 'for phase in (torch.unique(batch["dynamic_object.phase_id"]).tolist() if "dynamic_object.phase_id" in batch else []):')
    # This isolated family never uses position-dependent phase replay.
    domdata = source/'utils/dom_data.py'
    text = domdata.read_text(); start = text.index('        ep = int(batch["episode_index"])', text.index('    def __getitem__'))
    end = text.index('        return batch', start)
    domdata.write_text(text[:start]+text[end:])
    distributed = source/'core/dom_distributed_training.py'
    replace(distributed, "name='fresh20k-palmetto-'+variant+'-s42'", "name='krv-speed-gate-old20k-s42'")
    # Keep sample cursor/RNG semantics, but bound worker lifetimes on the full dataset.
    replace(distributed, "                    if halt:return {**progress,'interrupted':True}", "                    if halt:return {**progress,'interrupted':True}\n                    if progress['updates'] % 256 == 0 and progress['cursor'] < len(data):break")
    replace(distributed, "                if progress['cursor'] != len(data):raise RuntimeError('Distributed epoch omitted samples')", "                if progress['cursor'] < len(data):continue\n                if progress['cursor'] != len(data):raise RuntimeError('Distributed epoch omitted samples')")
    # Checkpoints every 15 minutes, plus existing early checkpoints.
    replace(distributed, 'time.monotonic()-last_save>=1800', 'time.monotonic()-last_save>=900')
    manifest.pop('fingerprint')
    manifest.update(dataset_root=str(data), initialization=str(family/'initialization'), speed_gate=gate,
                    source_sha256=source_digest(source))
    manifest['artifacts']['speed_gate.json'] = file_digest(family/'speed_gate.json')
    manifest['fingerprint'] = digest(manifest)
    atomic_json(family/'manifest.json', manifest)
    recipe = json.loads((PRIOR/'family/recipe.json').read_text()); recipe.pop('fingerprint')
    recipe['manifest'] = manifest['fingerprint']; recipe['fingerprint'] = digest(recipe)
    atomic_json(family/'recipe.json', recipe)
    atomic_json(STAGE/'plan.json', dict(family=str(family), source=str(source), manifest=manifest['fingerprint'],
        recipe=recipe, speed_gate=gate, baseline=str(PRIOR), variant='krv100', method='KRV current-speed gate',
        innovation=False, object_position_for_gate=False, reference_profile='fresh20k-20261025/profiling/selected_runtime.json',
        candidates=[dict(microbatch=m,workers=8,prefetch=2,pin_memory=True,fused=True,precision='bf16') for m in [32,40,44]],
        scope='Local 2x A6000; unchanged original split, normalization, initialization, batch, schedule and horizons'))
    print(json.dumps(gate, indent=2), flush=True)

if __name__ == '__main__': main()
