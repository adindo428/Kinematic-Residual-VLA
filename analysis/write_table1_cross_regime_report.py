"""Render the saved Table 1 cross-regime audit and its standalone figure."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'scratch/dom45/audits/table1-cross-regime-20261112'
def read(name):return json.loads((OUT/name).read_text())
def link(label,path):return f'[{label}]({ROOT/path})'
pairs=read('paired-results.json');targets=read('target-statistics.json');data=read('dataset-comparison.json')
training=read('training-inventory.json');tracking=read('tracking-full/summary.json')
assert tracking['complete'] and tracking['pairing_valid'] and tracking['recorded_rollouts']==7200

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(15,10),layout='constrained')
ax=axes[0,0]
labels=['DOM','DOM + velocity','KR coordinate','KR + velocity']
values=[r['Success'] for r in tracking['table']]
ax.barh(labels,values,color=['#9cabb7','#5386b2','#41a89a','#16766e'])
for i,v in enumerate(values):ax.text(v+.5,i,f'{v:.2f}% ',va='center')
ax.set_xlim(0,60);ax.invert_yaxis();ax.set_xlabel('Mean time within 3 cm (%)')
ax.set_title('A. Complete tracking benchmark: 1,800 episodes/model',loc='left',fontweight='bold')
ax=axes[0,1]
entries=[('DOM/KRV best (20k)',pairs['low_clutter']['dom100_best__krv100_best']),
         ('DOM/KRV epoch 10 (20k)',pairs['low_clutter']['dom100_latest__krv100_latest']),
         ('SmolVLA (20k continuation)',pairs['low_clutter']['smolvla_baseline__smolvla_kr']),
         ('VLASH corrected (900 trials)',pairs['low_clutter']['vlash_baseline_fixed__vlash_kr_fixed']),
         ('π0.5 (unequal training budgets)',pairs['low_clutter']['pi05_baseline__pi05_kr']),
         ('DOMV/KRV best full (first 900)',pairs['full_scale_balanced_first900']['domv_best__krv_best_constant']),
         ('DOMV best / KRV latest full (900)',pairs['full_scale_balanced_first900']['domv_best__krv_latest_constant'])]
for i,(label,r) in enumerate(entries):
    lo,hi=r['scenario_bootstrap_95_pp'];v=r['delta_pp']
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#ad5353' if i<5 else '#16766e',capsize=3)
ax.set_yticks(range(len(entries)),[e[0] for e in entries]);ax.invert_yaxis();ax.axvline(0,color='gray',lw=1)
ax.set_xlabel('KR minus baseline success (percentage points)')
ax.set_title('B. Manipulation: scenario-bootstrap 95% intervals',loc='left',fontweight='bold')
ax=axes[1,0]
for i,(key,label) in enumerate([('tracking','Tracking (80 train episodes)'),('20k_dom','DOM 20k (512 sampled episodes)'),('full_domv','DOMV full (512 sampled episodes)'),('full_krv','KRV full (512 sampled episodes)')]):
    r=targets[key];value=r['all']['residual_to_raw_variance_ratio'];lo,hi=r['episode_bootstrap_variance_ratio_95']
    ax.barh(i,value,color='#16766e' if key=='tracking' else '#ad5353')
    ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='none',color='black',capsize=3)
    ax.text(max(value,hi)+.03,i,f'{value:.3f}',va='center')
ax.set_yticks(range(4),['Tracking (all 80 train episodes)','DOM 20k (512 episodes)','DOMV full (512 episodes)','KRV full (512 episodes)'])
ax.invert_yaxis();ax.axvline(1,color='gray',lw=1,ls='--');ax.set_xlim(0,1.6)
ax.set_xlabel('Residual / raw XYZ target variance (physical units)')
ax.set_title('C. Does the correction simplify demonstration targets?',loc='left',fontweight='bold')
ax=axes[1,1]
for key,label,color in [('tracking','Tracking','#16766e'),('20k_dom','DOM 20k','#5386b2'),('full_krv','KRV full','#ad5353')]:
    r=targets[key]['by_horizon_index_variance_ratio'];x=[0,4,9,19]
    ax.plot(np.array(x)*.04,[r[str(i)] for i in x],'-o',label=label,color=color)
ax.axhline(1,color='gray',ls='--',lw=1);ax.legend();ax.set_xlabel('Analytic drift horizon j × 0.04 s')
ax.set_ylabel('Residual / raw XYZ target variance')
ax.set_title('D. DOM benefit reverses at longer horizons',loc='left',fontweight='bold')
fig.suptitle('KR helps controlled tracking; manipulation exposes target and protocol differences',fontsize=15,fontweight='bold')
fig.savefig(OUT/'analysis-overview.png',dpi=180)
fig.savefig(OUT/'analysis-overview.pdf')

lines=['# Why KR transfers weakly from tracking to Table 1 manipulation','',
'Audit snapshot: 12 November 2026. This analysis reads saved results and pinned training artifacts, and runs CPU-only statistical diagnostics on the local workstation. It does not change scores, checkpoints, training, the paper, or running evaluations.','',
'The strongest explanation is a combination of **a less suitable constant-velocity target decomposition for manipulation, a misleading comparison against the tracking baseline without velocity, and training/selection mismatches**. The evidence does not support a universal KR implementation failure. It also does not establish that any proposed training change will improve closed-loop success.','',
'![Audit overview](analysis-overview.png)','',
'## 1. What the results actually establish','',
'The complete tracking table was recovered from the existing private Hugging Face archive at a pinned repository revision. Its summary reports **7,200/7,200 rollouts, pairing valid, and 100% episode completion**. The older local paper evidence covers only CR/DA/LS. “Full tracking” here means all nine evaluation conditions; it still uses the 100-episode tracking dataset, not the full DOM training split.','',
'| Tracking model | Mean coverage within 3 cm |','|---|---:|']
lines += [f"| {r['Method']} | {r['Success']:.2f}% |" for r in tracking['table']]
lines += ['',
'Adding neural velocity to DOM gives **+34.82 percentage points**. Adding KR when neural velocity is already available gives **+5.41 points**. The latter is the more relevant comparison for isolating residualization. Coordinate-only KR also receives privileged current velocity through its analytic branch, so its comparison with DOM is not an equal-information comparison. KR+velocity improves seven of nine conditions over DOM+velocity; it loses 0.79 points on CR and 2.63 on MG. Thus tracking does not show uniform superiority, even before manipulation.','',
'Manipulation results recomputed from saved trial records:','',
'| Comparison | Trials/model | Baseline success | KR success | Δ pp | Scenario-bootstrap 95% interval |','|---|---:|---:|---:|---:|---|']
for label,r in entries:
    lo,hi=r['scenario_bootstrap_95_pp']
    lines.append(f"| {label} | {r['matched_trials']} | {r['baseline_percent']:.2f}% | {r['kr_percent']:.2f}% | {r['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
lines += ['',
'Intervals resample entire scenario clusters within the nine conditions, 10,000 times. Repeats from the same scene are not treated as 20 independently sampled environments. These intervals describe variation across benchmark scenarios and do not cover training-seed uncertainty. They are exploratory, with no multiplicity correction. Model inference noise is not paired.','',
'Most small gains/losses span zero under this analysis. The π0.5 gap is larger and its interval stays negative, but its checkpoints have unequal training budgets and histories. This is evidence about those saved checkpoints, not an identified causal effect of KR.','',
'For the complete 20k validation-best pair, KR succeeds on 235 trials where DOM fails, while DOM succeeds on 253 where KR fails; both succeed on 160. KR changes which cases succeed considerably, but those gains and losses largely cancel.','',
'For full-scale manipulation, the balanced first pass is the fairest available cross-variant summary: **DOMV best 26.11%, KRV best 27.00%, KRV latest constant 28.44%, KRV latest causal gate 27.00%**, each over 900 trials. The displayed full table mixes 200-trial completed columns and 100-trial provisional columns. Do not average those displayed cells and call that a completed 1,800-trial result. The raw ledgers also retain ten additional second-pass VU trials per main model; the audit records these separately in the all-common analysis, but the conclusions above use only the balanced first 900.','',
'## 2. A direct diagnostic of the KR training target','',
'I reconstructed physical XYZ action chunks from the parquet files using the trainer’s current-state reference: `a[j] = action[t+j, :3] − state[t, :3]`. The analytic correction is `d[j] = (v[t] − mean_train_velocity) × j × 0.04`, and the residual target is `r = a − d`. Rotation and gripper are excluded from this diagnostic because KR does not alter them.','',
'The sample contains all 80 tracking training episodes and a fixed random sample of 512 training episodes from each DOM split. It includes all interior, nonpadded 20-action anchors starting at frame 2 or later. It does not decode RGB or run policy inference. Variance is computed across anchors separately at each horizon/channel, then summed across XYZ and all 20 horizons.','',
'| Training data | Eligible anchors | Residual/raw XYZ variance | Episode-bootstrap 95% interval |','|---|---:|---:|---|']
for key,label in [('tracking','Tracking'),('20k_dom','DOM 20k'),('full_domv','DOMV full'),('full_krv','KRV full')]:
    r=targets[key];lo,hi=r['episode_bootstrap_variance_ratio_95']
    lines.append(f"| {label} | {r['all']['anchors']:,} | {r['all']['residual_to_raw_variance_ratio']:.3f} | [{lo:.3f}, {hi:.3f}] |")
lines += ['',
'The decomposition removes about **74% of tracking target variance**, but adds approximately **18% on the 20k DOM sample and 27–29% on the full DOM samples**. Bootstrap intervals use 2,000 episode-cluster resamples. The tracking point estimate covers all training episodes; its interval describes resampling those trajectories. These are marginal physical-target statistics, not normalized flow loss, conditional model error, or proof of a closed-loop cause. An invertible coordinate change need not reduce marginal variance to be learnable.','',
'The horizon breakdown is more revealing. At action index 4 (0.16 s analytic horizon), the DOM variance ratios are about **0.87–0.88**: the correction helps locally. At index 19 (0.76 s), they become **1.44–1.64**. Tracking remains at **0.31** at index 19. This makes a shorter or confidence-weighted extrapolation a stronger experimental candidate than blindly applying the full displacement throughout a 20-action chunk.','',
'The identity `Var(a−d) = Var(a) + Var(d) − 2 Cov(a,d)` explains the test. A unit-strength correction reduces variance only when `2 Cov(a,d) > Var(d)`. The sample’s minimum-variance scalar coefficient `Cov(a,d)/Var(d)` is about **1.19 for tracking**, **0.31 for 20k DOM**, and **0.24–0.25 for full DOM**. These are descriptive fits to training data, not validated deployment coefficients. They suggest testing a learned causal scale/horizon with held-out validation.','',
'The physical reason is plausible and supported by the data: tracking asks the end effector to keep following one moving object; manipulation asks it to approach, grasp, lift, transport to a different destination, release, and sometimes manipulate another object. Current object velocity ceases to be a universal desired hand-motion direction. Sampled 99th-percentile speed is about **0.14 m/s in tracking versus 0.99–1.15 m/s in DOM**, and the corresponding 99th-percentile final analytic correction is **0.11 m versus 0.76–0.87 m**. Contact, rapid motion, and releases make long constant-velocity extrapolation fragile. These are training-data tails, not measured velocities in every Table 1 failure.','',
'The task breakdown agrees: full-KRV sampled residual/raw variance is about **1.00 for pick, 1.40 for place, and 1.22 for long-horizon demonstrations**. Phase-cache breakdowns show particularly large ratios for the heuristic “released” phase, but those phase labels are causal rule outputs rather than ground-truth contact annotations.','',
'There is also an object-selection limitation. The evaluator reads velocity from `scene["object"]`, while long-horizon instructions can require placing the “entire set of objects,” including `object01`. The analytic branch does not change its tracked entity when the intended object changes. This is a confirmed interface limitation; its contribution to individual failures has not been measured.','',
'## 3. The training datasets are not interchangeable','',
'| Family | Train episodes | Train frames | Validation / test episodes | Pick / place / long-horizon train episodes |','|---|---:|---:|---:|---|']
for key,label in [('20k_dom','Fresh DOM/KRV 20k'),('full_domv','DOMV full'),('full_krv','KRV full')]:
    r=data[key];t=r['task_episodes'];c=r['counts']
    lines.append(f"| {label} | {c['train']:,} | {r['train_frames']:,} | {c['validation']:,} / {c['test']:,} | {t['pick']:,} / {t['place']:,} / {t['long_horizon']:,} |")
lines += ['',
'Tracking has **80/10/10 train/validation/test episodes**, 250 recorded frames each, from a 100-episode pilot in one room with one training apple and one hover task. Its recipes train for 75 epochs. This is strong repeated specialization to a simple behavior, not evidence from training on the full manipulation dataset. More epochs here does not mean more optimizer updates or total examples than DOM training.','',
'Fresh DOM and KRV 20k use exactly the same 20,000 train IDs, 2,015 validation IDs, 2,011 test IDs, and 2,406,773 training frames. Their training task quotas are 9,000 pick, 9,000 place, and 2,000 long-horizon episodes. This oversamples pick and long-horizon relative to the full DOM splits; place accounts for approximately 46% of 20k training frames versus 67% of full-KRV training frames. Most benchmark configurations are place tasks.','',
'DOMV full and KRV full share **194,740 training IDs**, but DOMV has **2,458 train IDs absent from KRV train**, and KRV has **8,411 absent from DOMV train**. Their validation sets overlap by only **770 episodes**. The 20k split is contained exactly in the DOMV full split; only 19,828/20,000 of its train IDs are in KRV’s train split. Therefore the full pair is not merely “the same split with 5,953 extra examples,” and its validation losses are not directly comparable as if evaluated on identical data.','',
'Within each frozen split I found **zero duplicate-trajectory-group overlap between train and validation or train and test**. That does not certify unseen scenes or remove exposure inherited from the released DOM-pretrained initialization. The prior scene audit could not identify the original training-room configuration for every Table 1 case. Low clutter is therefore not an IID guarantee.','',
'The H200 fresh-KRV directory contains a legacy `dataset_coverage.json` reporting 201,224 episodes, despite its actual frozen split and current manifest specifying the matched 20k family. This audit uses split membership, episode records, and current manifest fields, not that inherited coverage headline. The stale metadata should be cleaned up in a future provenance update; it is not evidence that the fresh KRV checkpoint trained on 201,224 episodes.','',
'## 4. Training and checkpoint selection','',
'**Fresh native DOM/KRV:** common released initialization and state/action normalizers, fresh optimizer, effective batch 352, BF16, seed 42, nominal LR 3e-5, 20-action horizon, frame sampling. The frozen scheduler uses 1,000-update warmup and a 100-epoch cosine horizon even when the operational budget is ten epochs. DOM validation-best is the actual epoch-8 checkpoint; “best epoch 7” in early-stopping state is the significant-improvement reference, not the minimum-MSE checkpoint. KRV validation-best is epoch 5. Both latest checkpoints have ten completed epochs. DOM has no neural velocity conditioning; KRV has it, so this pair changes both information and residual coordinates. Different hardware/runtime and one training seed further limit attribution.','',
'**Full native pair:** DOMV and KRV both have neural velocity, but differ in split, validation membership, training history, and runtime. Both “best” checkpoints are epoch 3; the latest KRV has four epochs plus part of five. KRV also has a documented topology transition from two to eight GH200 workers while preserving global batch, without bitwise equivalence. The released DynamicVLA result at 43.28% is an external pretrained checkpoint, not a baseline trained under the same new full-data recipe. Its advantage does not isolate KR.','',
'**Cross-VLA adapters:** SmolVLA, VLASH, and π0.5 rows are continuations of their own earlier DOM-trained checkpoints on the 20k subset. They are not the same “fresh 20k fine-tuning” experiment as native DOM/KRV. Their analytic branch receives current velocity, but the neural policy does not receive an explicit velocity input. SmolVLA keeps its vision encoder frozen and trains the expert/state projection; that trainable set differs from the fully unfrozen tracking family.','',
'The continuation recipes retain the parent scheduler. SmolVLA begins at update 20,000 on a 20,000-update schedule; VLASH at 30,000 on a 30,000 schedule; π0.5 KR at 60,004 on a 30,000 schedule. They therefore continue at the **2.5e-6 LR floor**. π0.5 baseline starts at 19,397/30,000 and also reaches the floor early. The prior checkpoint audit verified those actual optimizer learning rates. This could limit adaptation, but it is not sufficient evidence that a higher LR will improve the KR pair; both sides inherit substantial training.','',
'π0.5 baseline completed ten subset epochs; the evaluated KR has six plus a partial seventh. Earlier validation was better for KR at epoch 3 (action MSE 0.35689) than epoch 6 (0.36618). The retained 3.208-epoch checkpoint is now being evaluated, but it is not the exact epoch-3 validation checkpoint and its incomplete CR results cannot resolve the comparison. SmolVLA KR’s best saved validation measurement is epoch 9 rather than epoch 10. Historical VLASH validation used the input mismatch described below and cannot be trusted for corrected-input checkpoint selection.','',
'**The native selection metric is strongly misaligned with KR’s intended benefit.** The evaluator computes unnormalized squared error across XYZ in meters, Euler angles in radians, and gripper values, then averages seven channels. For selected DOM epoch 8, rotation contributes **99.16%**, XYZ **0.053%**, and gripper **0.788%** of that scalar. For KRV epoch 5 the shares are **99.10%, 0.053%, and 0.851%**. This is a selection-metric finding, not a claim that the normalized training loss has those weights. Raw Euler squared error also is not an SO(3) geodesic error.','',
'At their selected checkpoints, DOM’s translation MSE is 0.00022425 m² per coordinate versus KRV’s 0.00023469 m². Thus the current fresh-KRV checkpoint does not show an offline translation improvement hidden by the scalar score. Nevertheless, selecting almost entirely by orientation error is a weak test of a positional interception method. Gripper-frame accuracy above 99% also does not certify correct grasp/release transition timing.','',
'## 5. Implementation checks, timing, and gating','',
'The previous source/CPU audit checked train/inference transforms, XYZ-only correction, timestep and robot frame, normalization round trips, checkpoint identities, and one-time reconstruction. It found no common sign, axis, unit, zero-velocity-input, or double-addition defect explaining all KR regressions. These are useful checks, but they do not prove every forward pass or every trajectory correct.','',
'It did find a real **VLASH prompt mismatch**: training used 32 padded state tokens with right-inclusive discretization; native prediction rebuilt six tokens with the other boundary convention. The current first-pass table uses corrected prompts. Corrected VLASH is 5.67% versus corrected VLASH+KR 5.33% over 900 trials each. Earlier 1,800-trial mismatched-input scores remain historical and are excluded from the main comparison here. Absolute success is still low on both sides, so the input fix alone has not produced a broad KR advantage.','',
'**Inference-only gating is not a trained phase-aware KR policy.** For the VLASH ablation, the learned residual was trained as `r = a − d`, but evaluation reconstructs `r_hat + g*d`. Holding the residual prediction fixed changes the output by `(g−1)*d`. The network was never trained to compensate for that new reconstruction. Current VLASH constant-gate success is 5.33%, versus 3.67% with the causal grasp gate. The full-scale latest KRV comparison is 28.44% constant versus 27.00% causal over the matched first 900. Those observations reject the simple claim that enabling the existing gate alone fixes the issue. They do not rule out training with a consistent causal gate.','',
'**Timing differs materially between experiments.** Tracking pauses simulation during inference and replans every 0.8 s after three seconds of expert warmup. Manipulation uses the native asynchronous/latency-aware streaming controller. Current velocity at observation time, action age when executed, action index within the chunk, and target-label lead are distinct quantities. A continuous-following benefit measured with paused inference need not survive an asynchronous grasp deadline. Existing native chunk logs are roughly 0.28–0.31 s per call, with different measurement boundaries and workload modes; they do not establish KR-specific overhead.','',
'Both primary formulations use zero-based `j*0.04` drift. Tracking labels are explicitly measured tool poses five frames ahead; the downloaded tracking parquets match `action[t,:3] == state[t+5,:3]` exactly on tested interior rows. The official DOM translation utility constructs actions from a previous recorded trajectory’s future state and replays them through the simulator. DOM action labels differ from the newly measured `state[t+5]` by a median about 1.8–1.9 cm in the sampled data. This difference is not by itself a bug: command targets and realized replay states need not coincide. It prevents treating the DOM labels as exactly the same measured-future-state supervision used in tracking.','',
'Adding 0.2 s to reconstruction only is also a coordinate change without matched training. In the existing small three-scene diagnostic, gated success was 8/30 and gated-plus-lead was 6/30; this is too small and narrow for a general result, but gives no justification for silently applying that inference change.','',
'## 6. Low clutter does not remove the main difficulties','',
'Only 18 of 90 physical configurations were capped; 72 were physically unchanged. The benchmark retains language, viewpoint, speed, shape, disturbance, grasp, placement, and sequencing demands. For the fresh best pair, the KR difference is −2.50 points on the capped cases and −0.625 on unchanged cases. For the latest pair the signs differ: +2.78 on capped and −3.96 on unchanged. π0.5 KR loses on both groups. Clutter alone therefore does not account for the observed pattern, and these groups are not randomized controls for clutter difficulty.','',
'A lower all-trial mean time can mean an earlier failed attempt. Interpret completion speed on jointly successful trials separately from overall task success; do not turn a faster-failing gated policy into a claimed efficiency gain.','',
'## 7. What I would test next, in order','',
'1. **Use the current data to fit and validate the correction’s useful horizon.** On a frozen train/validation split, compare unit drift with a bounded learned scale or horizon-dependent decay. Break diagnostics out by approach, contact/transport, release, task, and velocity tail. Fit any coefficient on train only. Reconstruct with exactly the same rule used to form training residuals. The short-horizon variance benefit and long-horizon reversal make this a specific, testable hypothesis.','',
'2. **Run a truly matched native comparison.** DOM+velocity and KR+velocity should use identical episode membership, normalizers, initialization, optimizer/scheduler, example budget, validation anchors, and inference controller. Add DOM and KR without neural velocity if a four-way ablation is affordable. Use multiple training seeds. Do not use the current full pair’s small success delta as an isolated KR estimate.','',
'3. **Replace the single checkpoint-selection scalar with a prespecified validation protocol.** Report XYZ error in cm, rotation geodesic error, and grasp/release timing separately, by phase and horizon. Use a held-out closed-loop validation suite if feasible. Select before evaluating Table 1; do not search Table 1 for the best epoch. The 3.208-epoch π0.5 evaluation can diagnose checkpoint sensitivity but cannot supply an exact epoch-3 comparison.','',
'4. **Retrain gating consistently and supply the required context.** If drift should diminish during contact, train residual targets with that causal gate and use the identical gate at inference. Consider explicit velocity input to the residual expert so it can cancel unreliable motion priors. For multi-object tasks, define which object the velocity belongs to from causal task state. Avoid treating an inference-only gate as equivalent to this experiment.','',
'5. **Separate training and control timing effects.** Evaluate fixed checkpoints under paused-inference and deployed asynchronous control on a small prespecified diagnostic suite; measure observation age and selected action timestamps. Then train any delay-aware correction with the matching timing convention. This distinguishes a useful motion prior from a control-alignment failure.','',
'6. **For adapters, compare a deliberate continuation schedule against the inherited floor.** Keep each pair’s initialization and sample budget matched and select on corrected-input validation. Audit SmolVLA’s limited trainable set before interpreting its approximately 2–3% manipulation success as a KR-specific limitation.','',
'## 8. Evidence and reproducibility','',
'Run `scratch/dom45/envs/train/bin/python analysis/table1_cross_regime_audit.py`, then `scratch/dom45/envs/train/bin/python analysis/write_table1_cross_regime_report.py` from the repository root. The first script expects the pinned tracking parquets and full tracking metadata already downloaded into this audit directory. All computations are CPU-only. Sampled episode IDs and statistical definitions are recorded in JSON.','',
'- [Complete tracking table](tracking-full/tracking_table1.md), [summary](tracking-full/summary.json), and [pinned archive revision](tracking-archive-provenance.json).',
'- [Paired outcomes and confidence intervals](paired-results.json).',
'- [Dataset counts, task mix, split overlaps, and duplicate-group checks](dataset-comparison.json).',
'- [Training-target variance, horizons, phases, sampled IDs, and bootstrap intervals](target-statistics.json).',
'- [Training checkpoint inventory and native validation curves](training-inventory.json).',
'- [Low-clutter table snapshot](low-clutter-table-snapshot.md) and [full-scale table snapshot](full-scale-table-snapshot.md).',
'- '+link('Prior source/normalization/prompt audit','scratch/dom45/audits/kr-low-clutter-20261111/report.md')+'.',
'- '+link('Training-target reference subtraction','scratch/dom45/training/krv100-full-20261010/source/utils/datasets.py')+'; '+link('native physical validation metric','scratch/dom45/training/krv100-full-20261010/source/core/dom_training.py')+'.',
'- '+link('Adapter coordinate and drift definitions','scratch/dom45/evaluations/multivla20k-20261029/model_code/pi05_kr/common.py')+'; '+link('adapter neural input/training contract','scratch/dom45/evaluations/multivla20k-20261029/model_code/pi05_kr/policy_inference.py')+'.',
'- '+link('Tracking protocol and scope','experiments/tracking75/README.md')+'; '+link('DOM action translation/replay','scratch/dom45/official/source/scripts/translate_dataset_seq.py')+'.',
'- '+link('Full DOMV-matched training provenance','scratch/dom45/training/dom-full-matched-deltaai-20261107/README.md')+'; '+link('scene/IID evidence limit','scratch/dom45/evaluations/iid-scene-audit-20261102/audit.json')+'.','',
'The analysis establishes result differences, data/training confounds, target-statistic differences, and specific interface/selection limitations. It does not assign a percentage of closed-loop failure to each mechanism. That requires controlled retraining or evaluator ablations. The running π0.5 follow-up and paused second passes remain incomplete at this snapshot.','']
(OUT/'report.md').write_text('\n'.join(lines))
print(OUT/'report.md')
