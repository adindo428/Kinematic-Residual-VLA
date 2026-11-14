"""Freeze the approved 5 cm proximity-gate experiment from the matched 20k family."""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'scratch/dom45/training/krv-speed-gate-old20k-20261112'
STAGE = ROOT/'scratch/dom45/training/krv-distance-gate-old20k-20261112'

def replace(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (path, old)
    path.write_text(text.replace(old, new))

def main():
    assert not STAGE.exists(), 'Do not overwrite a frozen experiment'
    STAGE.mkdir()
    shutil.copytree(OLD/'source', STAGE/'source', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(OLD/'family', STAGE/'family', symlinks=True)
    source, family = STAGE/'source', STAGE/'family'
    sys.path.insert(0, str(source))
    from utils.dom_experiment import atomic_json, digest, source_digest, file_digest
    config = source/'policies/dynamicvla/configuration_dynamicvla.py'
    text = config.read_text().replace('residual_speed_threshold', 'residual_distance_threshold_m').replace('speed_threshold', 'distance_threshold').replace('Speed gate needs a positive fixed training threshold', 'Distance gate needs a positive radius in metres')
    config.write_text(text)
    modeling = source/'policies/dynamicvla/modeling_dynamicvla.py'
    replace(modeling, '''        return fields

    def prepare_residual_xyz''', '''        if self.config.residual_phase_gate_mode == "distance_threshold":
            # Preserve raw physical EE coordinates before normalize_inputs.
            fields["dynamic_object.gate_ee_position_xyz"] = batch[OBS_STATE][..., :3].clone()
        return fields

    def prepare_residual_xyz''')
    text = modeling.read_text(); start = text.index('        if self.config.residual_phase_gate_mode == "speed_threshold":',text.index('    def prepare_phase_gate'))
    end = text.index('        if self.config.residual_phase_gate_mode == "constant":',start)
    text = text[:start]+'''        if self.config.residual_phase_gate_mode == "distance_threshold":
            from proactive_vla.distance_gate import current_distance_gate
            return current_distance_gate(
                batch["dynamic_object.gate_ee_position_xyz"], batch[self.config.residual_position_key],
                self.config.residual_distance_threshold_m, self.config.chunk_size, self.config.n_obs_steps, device)
'''+text[end:]; modeling.write_text(text)
    (source/'proactive_vla/distance_gate.py').write_text('''"""Current physical-distance gate; no future action, velocity, or phase inputs."""
import torch

def current_distance_gate(ee, obj, radius, horizon, observations, device):
    def current(value):
        value = torch.as_tensor(value, dtype=torch.float32, device=device)
        if value.ndim == 3:
            if value.shape[1] != observations:
                raise ValueError("Gate accepts observation history, never a future trajectory")
            value = value[:, -1]
        if value.ndim != 2 or value.shape[-1] != 3 or not torch.isfinite(value).all():
            raise ValueError("Gate requires finite physical positions [batch,3]")
        return value
    ee, obj = current(ee), current(obj)
    if ee.shape != obj.shape:
        raise ValueError("EE/object position shapes differ")
    active = torch.linalg.vector_norm(obj-ee, dim=-1) > radius
    return active.to(torch.float32)[:,None,None].expand(-1,horizon,1)
''')
    training = source/'core/dom_training.py'
    text = training.read_text().replace('"speed_threshold"', '"distance_threshold"').replace('"residual_speed_threshold": manifest["speed_gate"]["threshold_m_s"]', '"residual_distance_threshold_m": manifest["distance_gate"]["radius_m"]')
    training.write_text(text)
    distributed = source/'core/dom_distributed_training.py'
    text = distributed.read_text().replace('krv-speed-gate-old20k-s42','krv-distance-gate-old20k-s42')
    text = text.replace("                    if progress['updates'] % 256 == 0 and progress['cursor'] < len(data):break\n",'')
    text = text.replace("loader = distributed_loader(data,runtime,progress['epoch'],progress['cursor'],recipe['sampling'])", "loader = distributed_loader(data,runtime,progress['epoch'],progress['cursor'],recipe['sampling'],limit_updates=256)")
    text = text.replace("def distributed_loader(data, runtime, epoch=0, cursor=0, sampling='frames', lengths=None):", '''class LimitedUpdates:
    """Drain a finite sampler prefix without prefetching work past its boundary."""
    def __init__(self, sampler, limit):
        self.sampler, self.limit = sampler, limit
    def __iter__(self):
        count = 0
        for batch in self.sampler:
            yield batch
            count += int(batch[0][4])
            if count >= self.limit: return

def distributed_loader(data, runtime, epoch=0, cursor=0, sampling='frames', lengths=None, limit_updates=None):''')
    text = text.replace("    options = dict(num_workers=runtime['workers']", "    if limit_updates is not None: sampler = LimitedUpdates(sampler, limit_updates)\n    options = dict(num_workers=runtime['workers']")
    text = text.replace("persistent_workers=True,\n                       multiprocessing_context", "persistent_workers=limit_updates is None,\n                       multiprocessing_context")
    text = text.replace("    key = digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':variant})[:16]", "    key = leader_call(lambda: digest({'hardware':hardware,'world_size':world,'runtime':runtime,'variant':variant,'manifest':manifest['fingerprint']})[:16])")
    distributed.write_text(text)
    profile = source/'core/dom_distributed_profiling.py'
    replace(profile,"loader = distributed_loader(data, runtime, epoch=epoch, sampling='frames')", "loader = distributed_loader(data, runtime, epoch=epoch, sampling='frames', limit_updates=warmup+measured_updates)")
    replace(profile, '        measurements = telemetry.summary(start_measurement)', '''        if mode == 'end_to_end':
            try:
                next(iterator)
                raise RuntimeError('Bounded profiling loader exceeded its update limit')
            except StopIteration:
                pass
        measurements = telemetry.summary(start_measurement)''')
    gate = dict(radius_m=.05, enabled='current EE-to-object-centre distance > 0.05 m',
                frame='shared robot-relative metric coordinates', timing='current observation anchor; fixed within predicted chunk',
                position_used_only_by_gate=True, innovation=False, hysteresis=False)
    atomic_json(family/'distance_gate.json',gate)
    (family/'speed_gate.json').unlink()
    manifest = json.loads((family/'manifest.json').read_text()); manifest.pop('fingerprint'); manifest.pop('speed_gate')
    manifest['artifacts'].pop('speed_gate.json'); manifest['artifacts']['distance_gate.json'] = file_digest(family/'distance_gate.json')
    manifest.update(initialization=str(family/'initialization'),distance_gate=gate,source_sha256=source_digest(source))
    manifest['fingerprint'] = digest(manifest); atomic_json(family/'manifest.json',manifest)
    recipe = json.loads((family/'recipe.json').read_text()); recipe.pop('fingerprint');recipe['manifest']=manifest['fingerprint'];recipe['fingerprint']=digest(recipe)
    atomic_json(family/'recipe.json',recipe)
    plan = json.loads((OLD/'plan.json').read_text()); plan.pop('speed_gate');plan.pop('controller_sha256',None)
    plan.update(source=str(source),family=str(family),manifest=manifest['fingerprint'],recipe=recipe,distance_gate=gate,
                method='KRV 5cm distance gate',object_position_for_gate=True,
                aborted_speed_experiment=str(OLD),loader_recycle_updates=256)
    atomic_json(STAGE/'plan.json',plan)
    print('Prepared', STAGE)

if __name__ == '__main__': main()
