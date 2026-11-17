"""CPU-only evidence report, refreshed after the queued evaluation completes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import threading

REPO = Path(__file__).resolve().parents[1]
WORK = REPO/'scratch/dom45'
EVAL = WORK/'evaluations/velocity-uncertainty-cr200-20261116'
OUT = EVAL/'evidence-audit'
STOP = threading.Event()


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def records(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def historical_pairs():
    from report_cdf_ablation import records as completed
    base = WORK/'evaluations'
    output = []
    groups = [('KRV', 'krv_best_constant', base/'closed-loop-krv-innovation-only-20261112'),
              ('DOMV', 'domv_best', base/'closed-loop-domv-innovation-cdf-20261112')]
    for label, key, vi_root in groups:
        vi_plan = read(vi_root/'plan.json')
        vi_model = vi_plan['models'][0]
        on = {r['scene']: r for r in completed(vi_root)}
        for speed, directory in [(.1, 'closed-loop-pilot-20261112'), (.5, 'closed-loop-speed-mid-20261112'), (1., 'closed-loop-speed-ood-20261112')]:
            root = base/directory
            model = next(m for m in read(root/'plan.json')['models'] if m['key'] == key)
            for off in completed(root, key):
                if off['scene'] not in on:
                    continue
                row = on[off['scene']]
                assert row['scene_sha256'] == off['scene_sha256']
                assert vi_model['weight_sha256'] == model['weight_sha256']
                gain = int(row['success'])-int(off['success'])
                provenance = read(Path(model['checkpoint'])/'provenance.json', {})
                output.append(dict(model=label, speed_mps=speed, scene_id=Path(row['scene']).stem,
                    scene_path=row['scene'], scene_sha256=row['scene_sha256'],
                    rollout_seed=row.get('seed'), training_seed=provenance.get('recipe', {}).get('seed'),
                    checkpoint=model['checkpoint'], checkpoint_sha256=model['weight_sha256'],
                    VI_off_success=bool(off['success']), VI_on_success=bool(row['success']),
                    observed_effect='helped' if gain > 0 else 'hurt' if gain < 0 else 'same outcome',
                    VI_off_folder=off['folder'], VI_on_folder=row['folder'],
                    limitation='scene-matched exploratory pair; historical policy RNG is not paired; no independent training replicate'))
    write(OUT/'historical_VI_paired_rollouts.json', output)
    lines = ['# Historical scene-matched VI outcomes', '',
             'Same checkpoint weights verified within each pair. Independent policy sampling is not controlled; these outcome flips do not by themselves establish a causal VI benefit.', '',
             '| Model | Speed | Scene ID | VI off | VI on | Observed change |', '|---|---:|---|---:|---:|---|']
    for r in output:
        if r['observed_effect'] != 'same outcome':
            lines.append(f"| {r['model']} | {r['speed_mps']} | {r['scene_id']} | {int(r['VI_off_success'])} | {int(r['VI_on_success'])} | {r['observed_effect']} |")
    (OUT/'historical_VI_changes.md').write_text('\n'.join(lines)+'\n')


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = read(EVAL/'plan.json')
    inventory = []
    for model in plan['models']:
        cp = Path(model['checkpoint'])
        config = read(cp/'config.json')
        provenance = read(cp/'provenance.json', {})
        meta = read(cp/'checkpoint_metadata.json', {})
        fields = ['chunk_size', 'n_action_steps', 'n_obs_steps', 'num_steps', 'use_delta_action',
                  'enable_residual_flow', 'condition_on_object_velocity', 'normalization_mapping',
                  'use_checkpoint_normalization', 'velocity_input_mean', 'velocity_input_std',
                  'residual_phase_gate_mode', 'residual_flow_formulation', 'residual_drift_mode',
                  'residual_action_dt', 'residual_center_drift', 'residual_normalize_drift']
        inventory.append(dict(model=model['key'], checkpoint=str(cp), weight_sha256=model['weight_sha256'],
            config_sha256=model['config_sha256'], source=model['source'],
            training_seed=provenance.get('recipe', {}).get('seed'),
            progress=meta.get('progress'), settings={k: config.get(k) for k in fields},
            drift_mean_sha256=hashlib.sha256(json.dumps(config.get('residual_drift_mean')).encode()).hexdigest(),
            independent_training_replicates='not demonstrated by this evaluation'))
    write(OUT/'checkpoint_inventory.json', inventory)
    # One row per scored trial, with identities and paths to unmodified source records.
    rollouts = []
    for job in plan['jobs'] + plan.get('retained_jobs', []):
        model = next(m for m in plan['models'] if m['key'] == job['model'])
        for row in records(Path(job['folder'])/'trial_results.jsonl'):
            repeat = int(row['trial'])+job.get('repeat_offset', 0)
            rollouts.append(dict(**row, global_repeat=repeat, condition=job['condition'],
                checkpoint=model['checkpoint'], checkpoint_sha256=model['weight_sha256'], model=job['model'],
                training_seed=next(m['training_seed'] for m in inventory if m['model'] == job['model']),
                scene_id=Path(job['scene']).stem, scene_sha256=job['scene_sha256'],
                VI=False, VPD=False, raw_log=str(Path(job['folder'])/'trial_results.jsonl'),
                tracker_summary=str(Path(job['folder'])/'velocity_diagnostics'/f'repeat{repeat:02d}.json')))
    write(OUT/'per_condition_rollouts.json', rollouts)
    historical = []
    for folder in ['closed-loop-krv-innovation-only-20261112', 'closed-loop-krv-velocity-innovation-20261112',
                   'closed-loop-domv-innovation-cdf-20261112']:
        root = WORK/'evaluations'/folder
        historical.append(dict(root=str(root), comparison=str(root/'comparison.md'),
            protocol=read(root/'protocol.json'), status=read(root/'status.json')))
    write(OUT/'historical_VI_inventory.json', historical)
    historical_pairs()
    source = Path(plan['models'][0]['source'])
    refs = [source/'policies/dynamicvla/modeling_dynamicvla.py', source/'proactive_vla/residual_flow.py',
            REPO/'workstation/resident_table1_worker.py', REPO/'workstation/krv_profile_math.py',
            REPO/'workstation/velocity_uncertainty_sensor.py', REPO/'workstation/efficienttam_stream_worker.py']
    write(OUT/'execution_source_hashes.json', {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in refs})
    complete = bool(read(EVAL/'completion.json', {}).get('verified'))
    text = f'''# Velocity evidence audit

Generated {datetime.now(timezone.utc).isoformat()}. Noise evaluation complete: **{complete}**.
This report distinguishes implementation facts, observed results and missing experiments.

## Exact execution details

- VI-off disables the execution-time innovation offset. It does **not** disable KR residual prediction or analytic displacement restoration. With VPD/VI off the external correction is the identity. KRV still restores its trained centered constant-velocity displacement once after the flow solve, before output unnormalization and delta-to-absolute conversion.
- Delayed chunks retain the native streaming rule: skip_n_actions = current observation index minus source chunk observation index; discard that prefix and keep the native queue splice. This is observation-index alignment, not an extra measured wall-clock latency extrapolation. Source-observation age and action-arrival timing must not be conflated.
- The displacement time grid is h*0.04 seconds with h starting at zero. For this historical constant-velocity checkpoint, d_h = v(t0)*h*0.04. The displacement origin is the chunk's source observation, not receipt time. Delta actions are converted using the source observation's saved end-effector state; object position is not added as an absolute position restoration in this trajectory mode.
- Current observed object velocity in robot-base coordinates enters KRV's three additional normalized state channels. The same measurement supplies its analytic residual reconstruction. The last observation's velocity is broadcast over the horizon; this is **not VPD**. DOMV is the raw-action velocity-conditioned baseline; it receives the same corrupted velocity measurement through its trained neural conditioning pathway, with no KR restoration.
- Checkpoint input/action normalizers stay fixed. Neural velocity uses saved (v-mean)/std. Analytic drift is scaled by saved action standard deviations and the trained mean drift is centred consistently; restoration adds that centred drift back in residual coordinates. Config values and hashes are in checkpoint_inventory.json; the full drift means and normalizer tensors remain in the named checkpoint. No new statistics are estimated on the evaluation set.
- Tracker: EfficientTAM-Ti 512, official revision {plan['protocol']['tracker_revision']}, opposite-camera RGB-D, calibrated optical-to-base transforms, first-frame privileged instance mask only. No subsequent target pose/velocity is provided to the segmentation worker or position-slope estimator. Velocity is a causal trailing-0.16s least-squares slope with at least three samples; missing masks/depth clear history and yield zero. Initial conditioning plus 64 recent non-conditioning outputs are retained. GPU0 runs simulation/tracker; GPU1 runs one active resident policy at a time. Model-specific sampling RNG conventions remain different between historical KRV and DOMV.

## Per-condition rollout evidence and VI

per_condition_rollouts.json provides scene ID/hash, checkpoint ID/hash, condition, global repeat, outcome and diagnostic-log paths for every completed noise-sweep trial. Simulator/reset/prompt/action logs remain in the original block folders. A rollout seed or repeated evaluation of the same checkpoint is **not** an independently trained seed.

The current noise sweep has VI disabled in every arm, so it cannot measure VI's causal effect. historical_VI_inventory.json points to earlier VI comparisons and their actual checkpoints. historical_VI_paired_rollouts.json joins each historical VI-on/off trial by scene, verifies matching checkpoint and scene hashes, and records wins/losses, IDs and log paths; historical_VI_changes.md lists observed outcome flips. One archived epoch-3 KRV comparison reports baseline versus VI successes of 12/25 versus 12/25 at 0.10 m/s, 6/25 versus 3/25 at 0.50 m/s, and 2/25 versus 2/25 at 1.00 m/s. These are exploratory speed tests, not results from the current epoch-4+partial-5 checkpoint or the new noise sweep. The archived comparison records policy-RNG and training-history limitations. Outcome flips alone do not identify failure causes. No independently trained replicate series is established here.

## Matched controls

| Requested control | Evidence currently established |
|---|---|
| Raw-action policy plus online displacement | Historical naive/innovation code and pilots exist, but the current DOM/KRV noise sweep is not a newly trained, matched online-displacement control. Historical innovation is not equivalent to naive displacement. |
| Object-relative actions | No matched training-and-closed-loop result is established by these runs. |
| Direct object-position restoration | A lookahead/position code path exists in the source; the active trajectory checkpoint does not establish a matched trained position-restoration baseline. |
| Constant-velocity subtraction during training and restoration during execution | This is the historical constant-gate KRV representation, with saved centring/normalization. A raw-action versus residual causal attribution still needs matched dataset, update budget, normalization handling and independent training replicates. |

Do not label unrun controls as negative results. These evaluations do not create the missing trained controls.

## Baseline convergence and published protocol

Checkpoint progress is recorded in checkpoint_inventory.json where available. The historical KRV checkpoint is update304483, after epoch4 plus part of epoch5; its saved early-stopping state selects epoch3 with best h20/action_mse approximately0.17704 and one bad validation at epoch4. This is checkpoint-selection evidence, **not** a complete convergence curve for DOMV on the custom split. The saved recipe's seed is not a replicate count. Different checkpoints from one training trajectory are not independent training seeds.

The local released README documents Euler rotations, delta actions and continuous inference, and the official paper describes discarding latency-invalid chunk prefixes. Those mechanisms are retained. The current reduced-clutter 10-CR-scene uncertainty sweep, custom checkpoint histories, simulator wrappers, first-frame tracker initialization and historical RNG choices are experiment-specific changes. Success tolerance is0.03m. Equivalence to the full published benchmark, exact original training schedule and full custom-split convergence are **not established** by matching these runtime options alone. A full published-protocol reproduction and baseline learning curves remain separate evidence requirements.

Primary references: [official implementation](https://github.com/hzxie/DynamicVLA), [paper](https://arxiv.org/abs/2601.22153). Local README and pinned source provide reproducible commands; do not silently substitute the evolving upstream source for the frozen evaluation source.

## Post-grasp, occlusion and non-privileged identity

The original success/status logs support success/failure counts. They do not certify contact, stable grasp, dropped-object events, transport failure or release failure. Action/gripper logs and sampled videos can support a manual audit; pickup heuristics are not contact-ground-truth labels. A validated per-stage failure breakdown is not yet established.

The EfficientTAM arm records per-observation privileged position/velocity references, measured velocity, estimated visible-surface position, mask size, readiness, fitting delay and sensor wall time in compressed traces. error_metrics.json reports per-axis bias/std/RMSE and unavailable fraction separately for each model. Depth/segmentation loss is observable; a small mask is not itself a certified occlusion label. This experiment does not introduce controlled occlusion sweeps. Position error compares a visible surface centroid with object COM, so geometric offset is included and must be disclosed.

The experiment uses a privileged first-frame target mask. It is **not** an evaluation without privileged target identity. That claim requires a distinct language-grounded or otherwise non-privileged target-initialization experiment. After initialization, object truth is used only for diagnostics, with no privileged fallback on tracking loss.

## Dynamic results

- Noise success table: ../live_results.md
- Per-model estimator errors: ../error_metrics.json
- Per-condition trial records: per_condition_rollouts.json
- Checkpoint and training-seed inventory: checkpoint_inventory.json
- Historical VI provenance: historical_VI_inventory.json

Pending or missing evidence is preserved explicitly; no efficacy or robustness claim follows from passing CPU readiness checks.
'''
    (OUT/'answers.md').write_text(text)
    write(OUT/'status.json', {'updated_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_complete': complete, 'indexed_noise_rollouts': len(rollouts), 'report': str(OUT/'answers.md')})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-eval', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    build()
    if args.after_eval:
        while not STOP.is_set():
            if read(EVAL/'completion.json', {}).get('verified'):
                build();return
            STOP.wait(30)


if __name__ == '__main__':
    main()
