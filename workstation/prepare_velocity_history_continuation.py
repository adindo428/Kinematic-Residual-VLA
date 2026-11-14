"""Fork local distance-gated KRV at validated epoch 1 with velocity history."""
import json
import os
from pathlib import Path
import shutil
import sys

from velocity_history_patch import patch_source

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'scratch/dom45/training/krv-distance-gate-old20k-20261112'
STAGE = ROOT/'scratch/dom45/training/krv-distance-gate-history-20261113'


def main():
    if STAGE.exists():
        raise FileExistsError('Preserve existing history experiment')
    source, family = STAGE/'source', STAGE/'family'
    shutil.copytree(PARENT/'source', source, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(PARENT/'family', family, symlinks=True, copy_function=os.link,
                    ignore=shutil.ignore_patterns('runs', '__pycache__'))
    patch_source(source)
    (source/'tests').mkdir(exist_ok=True)
    for name in ('test_dom_velocity_conditioning.py', 'test_residual_coordinate_flow.py'):
        shutil.copyfile(ROOT/'third_party/DynamicVLA/tests'/name, source/'tests'/name)
    sys.path.insert(0, str(source))
    from utils.dom_experiment import atomic_json, digest, file_digest, source_digest
    parent_run = PARENT/'family/runs/krv100'
    pointer = json.loads((parent_run/'epoch-001.json').read_text())
    assert pointer['progress']['epoch'] == 1 and pointer['progress']['cursor'] == 0
    assert not pointer['progress'].get('pending_validation')
    generation = parent_run/'generations'/pointer['generation']
    proxy = STAGE/'parent-epoch001'
    proxy.mkdir()
    (proxy/'generations').symlink_to(parent_run/'generations', target_is_directory=True)
    atomic_json(proxy/'latest.json', pointer)
    transition = dict(kind='Changed conditioning, not an exact same-regime resume',
        parent_run=str(parent_run), parent_fingerprint=pointer['fingerprint'],
        parent_generation=pointer['generation'], parent_weights_sha256=file_digest(generation/'model.safetensors'),
        parent_checkpoint_proxy=str(proxy), progress=pointer['progress'],
        observation_offsets_frames=[-2, 0], velocity_layout='current_xyz, previous_xyz in state slots 6:12',
        velocity_conditioning_mode='history', optimizer_preserved=True, scheduler_preserved=True,
        rank_rng_preserved=True, new_wandb_run=True, analytic_drift='unchanged current-velocity extrapolation',
        previous_partial_epoch='Preserved in parent latest checkpoint; continuation branches from epoch 1')
    atomic_json(family/'velocity_history_transition.json', transition)
    distributed = source/'core/dom_distributed_training.py'
    text = distributed.read_text()
    anchor = "        elif (destination/'provenance.json').exists():\n"
    assert text.count(anchor) == 1
    text = text.replace(anchor, '''        elif "velocity_history_transition" in manifest:
            transition = manifest["velocity_history_transition"]
            state = load_generation(Path(transition["parent_checkpoint_proxy"]), policy, optimizer, scheduler,
                                    transition["parent_fingerprint"], restore_random=False)
            progress = dict(state["progress"])
            if progress["epoch"] != 1 or progress["cursor"] != 0 or progress.get("pending_validation"):
                raise ValueError("History transition must start at validated epoch 1")
            progress["conditioning_transition"] = "current_to_observation_history_after_epoch_1"
            best, wandb_id = None, None
''' + anchor)
    text = text.replace('krv-distance-gate-old20k-s42', 'krv-distance-gate-history-s42')
    distributed.write_text(text)
    manifest = json.loads((family/'manifest.json').read_text())
    manifest.pop('fingerprint')
    manifest.update(source_sha256=source_digest(source), initialization=str(family/'initialization'),
                    velocity_conditioning_mode='history', velocity_history_transition=transition)
    manifest['artifacts']['velocity_history_transition.json'] = file_digest(family/'velocity_history_transition.json')
    manifest['fingerprint'] = digest(manifest)
    atomic_json(family/'manifest.json', manifest)
    recipe = json.loads((family/'recipe.json').read_text())
    recipe.pop('fingerprint')
    recipe.update(manifest=manifest['fingerprint'],
        selection='Continue epoch-1 distance-gated KRV with aligned causal velocity history; preserve data, optimizer, schedule and global batch.')
    recipe['fingerprint'] = digest(recipe)
    atomic_json(family/'recipe.json', recipe)
    selected = json.loads((PARENT/'selected_runtime.json').read_text())
    selected.update(manifest=manifest['fingerprint'], scope='Parent runtime candidate; fresh calibration and readiness required')
    atomic_json(STAGE/'selected_runtime.json', selected)
    atomic_json(STAGE/'plan.json', dict(manifest=manifest['fingerprint'], candidates=[selected['runtime']], transition=transition))
    controller = (ROOT/'workstation/run_distance_gate20k.py').read_text()
    controller = controller.replace("krv-distance-gate-old20k-20261112", STAGE.name)
    anchor = "        status('readiness', runtime=json.loads(selected.read_text())['runtime'])\n"
    assert controller.count(anchor) == 1
    controller = controller.replace(anchor, '''        runtime = json.loads(selected.read_text())['runtime']
        status('calibrating_history', runtime=runtime)
        output = profiles/'history-calibration.json'
        if execute('profile', runtime, output, 'history-calibration'):
            raise RuntimeError('History calibration failed')
        if not json.loads(output.read_text())['safe']:
            raise RuntimeError('History calibration exceeded memory limits')
''' + anchor)
    controller = controller.replace("    import psutil\n", '    import psutil\n    # Local shared-profiler memory limit in MiB.\n    os.environ.setdefault("SLURM_MEM_PER_NODE", str(psutil.virtual_memory().total // 1024**2))\n')
    controller_path = ROOT/'workstation/run_velocity_history_training.py'
    controller_path.write_text(controller)
    atomic_json(STAGE/'status.json', dict(phase='prepared', transition=transition))
    (STAGE/'README.md').write_text(
        '# Velocity-history continuation\n\n'
        'Branches from validated epoch 1 of distance-gated KRV. Both velocity samples match '
        'the image/observation offsets [-2, 0] at 25 Hz. Current velocity remains in slots '
        '6:9, previous velocity occupies 9:12. Old checkpoints default to current-only. '
        'Analytic drift, gate, dataset, global batch, optimizer and LR schedule are preserved. '
        'History supplies acceleration evidence; it does not provide future velocity.\n')
    print(json.dumps(dict(stage=str(STAGE), manifest=manifest['fingerprint'], parent_generation=pointer['generation'])))


if __name__ == '__main__':
    main()
