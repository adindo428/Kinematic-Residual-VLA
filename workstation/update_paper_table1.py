#!/usr/bin/env python3
"""Update only generated blocks inside the paper's Experiments section."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from table1_display_state import low_rows

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'scratch/dom45/evaluations/low-clutter-full-table1-20261102'
PAPER = REPO / 'CVPR27_Kinematic Residual Flow Matching for Vision-Language-Action Policies/main.tex'
PAPER_COPIES = (PAPER, REPO / 'paper/main.tex')
DIMS = ['CR', 'DA', 'LS', 'VU', 'SR', 'MP', 'VG', 'MG', 'DR']
LABELS = {'dom100_best': 'DynamicVLA (validation-best)', 'krv100_best': 'DynamicVLA + KR + velocity (validation-best)',
 'dom100_latest': 'DynamicVLA (epoch 10)', 'krv100_latest': 'DynamicVLA + KR + velocity (epoch 10)',
 'smolvla_baseline': 'SmolVLA', 'smolvla_kr': 'SmolVLA + KR',
 'vlash_baseline': 'VLASH', 'vlash_kr': 'VLASH + KR',
 'pi05_baseline': r'$\pi_{0.5}$', 'pi05_kr': r'$\pi_{0.5}$ + KR',
 'krv100_full':r'KRV100 Full (validation-best epoch 3)'}
ORDER = list(LABELS)
ORDER.remove('krv100_full')
ORDER.insert(ORDER.index('smolvla_kr')+1,'krv100_full')


def read(path):
 return json.loads(path.read_text())


def verified_values(name):
 root = ROOT / 'results' / name
 values = {}
 for i, dimension in enumerate(DIMS):
  path = root / 'full' / f'{i:02d}_{dimension}' / 'verification.json'
  if path.exists():
   record = read(path)
   assert record['verified'] and record['matched_prompts_verified'] and record['trajectory_metrics_verified'] and record['trials'] == 200
   values[dimension] = record['success_percent']
 path = root / 'summary.json'
 if path.exists():
  summary = read(path)
  assert summary['complete'] and summary['verified_trials'] == 1800 and len(values) == 9
  values.update(summary['observed'])
 return values


def table1():
 text = r'''\begin{table*}[t]
\centering
\caption{Low-clutter DOM manipulation: success (\%) and all-trial mean
termination time (s). Each row is one checkpoint; each condition has 200 trials.
Overall scores require all 1,800 trials. ``--'' denotes unavailable results;
``running'' marks the active evaluation condition.}
\label{tab:dom_main}
\begingroup
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.08}
\scriptsize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l*{11}{r}@{}}
\toprule
& \multicolumn{3}{c}{Interaction} & \multicolumn{3}{c}{Perception}
& \multicolumn{3}{c}{Generalization} & \multicolumn{2}{c}{Average} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-12}
Method & CR & DA & LS & VU & SR & MP & VG & MG & DR & Success $\uparrow$ & Time $\downarrow$ \\
\midrule
'''
 for index, name in enumerate(ORDER):
  if name in ['smolvla_baseline','krv100_full','vlash_baseline','pi05_baseline']: text += '\\midrule\n'
  values = verified_values(name)
  cells = [f'{values[key]:.2f}' if key in values else '--' for key in [*DIMS, 'Success', 'Time']]
  for row in low_rows():
   dim=row.get('dimension')
   if row.get('model')==name and row.get('phase')=='full' and dim in DIMS and dim not in values:
    cells[DIMS.index(dim)]=r'\textit{running}'
  text += LABELS[name] + ' & ' + ' & '.join(cells) + r' \\' + '\n'
 text += r'''\bottomrule
\end{tabular*}
\endgroup
\par\smallskip
\parbox{\textwidth}{\footnotesize CR: closed-loop reactivity; DA: dynamic adaptation;
LS: long-horizon sequencing; VU: visual understanding; SR: spatial reasoning;
MP: motion perception; VG: visual generalization; MG: motion generalization;
DR: disturbance robustness. KRV100 Full is the full-data validation-best epoch-3
checkpoint, outside the primary 20k cohort. Its evaluation is pending;
scores from the retired mixed-data oldKRV100 checkpoint are excluded.
DOM/KRV also differs in neural velocity conditioning;
SmolVLA $\pm$ KR retains the same neural observations. KR uses privileged
simulator-state velocity. The historical released-checkpoint evaluation used
the original physical configurations and is not ranked against this variant.}
\end{table*}'''
 return text


def results():
 b, k = verified_values('dom100_best'), verified_values('krv100_best')
 text = ('Table~\\ref{tab:dom_main} reports the low-clutter 20k-subset study. '
         'The validation-selected DOM and KRV checkpoints were selected from training validation, '
         'and the latest checkpoints are reported separately. ')
 if 'Success' in b and 'Success' in k:
  delta = k['Success'] - b['Success']
  text += (f"The completed validation-best DOM checkpoint achieves {b['Success']:.2f}\\% "
           f"success with mean termination time {b['Time']:.2f}\\,s; KRV achieves {k['Success']:.2f}\\% "
           f"and {k['Time']:.2f}\\,s. KRV changes overall success by {delta:+.2f} percentage points, "
           'so this completed pair does not establish an overall improvement. ')
  gains = [f'{d} ({k[d]-b[d]:+.2f})' for d in DIMS if k[d] > b[d]]
  text += 'Its positive condition differences are ' + ', '.join(gains) + ' percentage points. '
 pair = ROOT / 'paired_analysis/dom100_best__vs__krv100_best/summary.json'
 if pair.exists():
  p = read(pair);counts = p['counts']
  text += (f"Of 1,800 paired trials, {counts.get('both_success',0)} succeed for both, "
           f"{counts.get('kr_success_baseline_failure',0)} only for KRV, "
           f"{counts.get('baseline_success_kr_failure',0)} only for DOM, and "
           f"{counts.get('both_failure',0)} fail for both. These counts retain both directions of change. ")
 smol_b,smol_k=verified_values('smolvla_baseline'),verified_values('smolvla_kr')
 text += 'The current DOM/KRV comparison combines residualization and neural velocity conditioning. '
 if 'Success' in smol_b and 'Success' in smol_k:
  text += (f"SmolVLA and SmolVLA+KR achieve {smol_b['Success']:.2f}\\% and {smol_k['Success']:.2f}\\%, "
           f"respectively ({smol_k['Success']-smol_b['Success']:+.2f} percentage points), with unchanged neural observations. ")
 else:text += 'The remaining SmolVLA comparison will assess residualization with unchanged neural inputs. '
 bootstrap=ROOT/'post_table1/paired_bootstrap.json'
 if bootstrap.exists():
  for pair in read(bootstrap)['pairs']:
   low,high=pair['ci95_pp']
   text += (f"The configuration-bootstrap 95\\% interval for {LABELS[pair['kr']]} minus "
            f"{LABELS[pair['baseline']]} is [{low:+.2f}, {high:+.2f}] percentage points. ")
 text += ('Partial rows do not support full-suite rankings.\n\n'
          'For historical context only, the released DynamicVLA checkpoint achieved 43.28\\% '
          'success and 8.62\\,s over 1,800 trials under the original physical configurations. '
          'That evaluation differs in checkpoint provenance and object clutter, so it is not '
          'a controlled comparison with Table~\\ref{tab:dom_main}.')
 return text


def runtime():
 root = ROOT / 'runtime_metrics'
 observed = read(root / 'runtime_summary.json')
 resources = read(root / 'resource_summary.json')
 text = r'''The live timing evidence in Table~\ref{tab:runtime} is observational. Native
timers surround the host policy call without explicit CUDA synchronization or
the final CPU action transfer. Adapter timers include input preparation,
physical-action reconstruction and the CPU copy. Both exclude the intentional
simulation-rate sleep. Only parallel-schedule, scored-phase samples are used;
post-episode in-flight chunks and unscored pilots are excluded. Task mixtures
and the different timing boundaries prevent attributing these differences to KR.

\begin{table}[t]
\centering
\caption{Observed parallel-evaluation latency (ms/chunk), not controlled KR
overhead. Mem.: sampled peak policy-process driver memory (GiB), excluding
simulation; sampling began Nov.~4. ``--'' denotes unmeasured values.}
\label{tab:runtime}
\setlength{\tabcolsep}{2pt}
\scriptsize
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
Checkpoint & Chunks & p50 & p95 & Mem. \\
\midrule
'''
 short = {'dom100_best':'DOM best', 'krv100_best':'KRV best', 'dom100_latest':'DOM epoch 10',
  'krv100_latest':'KRV epoch 10','smolvla_baseline':'SmolVLA','smolvla_kr':'SmolVLA + KR',
  'vlash_kr':'VLASH + KR','pi05_baseline':r'$\pi_{0.5}$','krv100_full':'KRV100 Full'}
 for name, label in short.items():
  row = next((x for x in observed['models'] if x['model']==name and x.get('execution_mode')=='parallel_shared' and x.get('phase')=='full'), None)
  memories = [value/1024 for key,value in resources['sampled_max_driver_memory_mib'].items() if key.startswith(name+'/policy/')]
  cells = [str(row['samples']),f"{row['p50_ms']:.2f}",f"{row['p95_ms']:.2f}"] if row else ['--']*3
  cells += [f'{max(memories):.2f}' if memories else '--']
  text += label + ' & ' + ' & '.join(cells) + r' \\' + '\n'
 text += r'''\bottomrule
\end{tabular*}
\end{table}

Controlled sequential profiling will measure synchronized observation-to-action
latency, parameter counts, allocator peaks, and same-checkpoint analytic-path
cost.
These quantities remain unmeasured; sampled driver memory and shared-GPU
timings are not substituted for them. Simulator-state velocity excludes visual
estimation cost. Configuration-cluster bootstrap intervals and matched
velocity-only/KR-only comparisons remain pending.
'''
 profile = ROOT / 'post_table1/controlled_runtime/summary.json'
 if profile.exists():
  p = read(profile)
  text += '\n% Controlled profile completed; full machine-readable record: post_table1/controlled_runtime/summary.json\n'
  text += r'''\begin{table*}[t]
\centering
\caption{Controlled sequential profiling on fixed synthetic observations (compute only).
E2E spans prepared CPU observations to reconstructed physical actions on CPU;
there is no visual velocity estimator. Parameter counts are unique loaded-module
parameters. Mem. is peak PyTorch allocated memory in GiB.}
\label{tab:runtime_controlled}
\scriptsize
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
Checkpoint & Parameters (M) & E2E p50 (ms) & E2E p95 (ms) & Mem. (GiB) \\
\midrule
'''
  for name, row in p['models'].items():
   text += short[name] + f" & {row['total_parameters']/1e6:.3f} & {row['enabled']['p50_ms']:.2f} & {row['enabled']['p95_ms']:.2f} & {row['enabled']['peak_allocated_bytes']/2**30:.2f}" + r' \\' + '\n'
  text += '\\bottomrule\n\\end{tabular*}\n\\end{table*}\n'
  text = text.replace('These quantities remain unmeasured; sampled driver memory and shared-GPU\ntimings are not substituted for them.',
    'Completed controlled measurements are reported in Table~\\ref{tab:runtime_controlled}; sampled driver memory and shared-GPU timings remain separate.')
  for pair in p['pairs']:
   text += (f"\nThe controlled {short[pair['kr']]} minus {short[pair['baseline']]} p50 difference is "
     f"{pair['delta_p50_ms']:+.2f}\\,ms ({pair['delta_p50_percent']:+.2f}\\%). ")
   if not pair['isolates_KR']:text += 'This pair also changes neural velocity conditioning. '
  for name,row in p['models'].items():
   if 'same_checkpoint_analytic_delta_p50_ms' in row:
    text += (f"The same-checkpoint analytic-path toggle for {short[name]} changes p50 by "
      f"{row['same_checkpoint_analytic_delta_p50_ms']:+.2f}\\,ms; this is a compute-only inference intervention. ")
 return text


def update_paper(path, generated):
 before = path.read_text()
 begin = before.index(r'\section{Experiments}')
 end = before.index(r'\section{Limitations}', begin)
 section = before[begin:end]
 for marker, content in generated:
  left = '% BEGIN GENERATED ' + marker
  right = '% END GENERATED ' + marker
  assert section.count(left)==section.count(right)==1
  start = section.index(left)+len(left)
  stop = section.index(right,start)
  section = section[:start] + '\n' + content + '\n' + section[stop:]
 after = before[:begin]+section+before[end:]
 assert after[:begin]==before[:begin]
 assert after[after.index(r'\section{Limitations}',begin):]==before[end:]
 if after != before:
  # Avoid overwriting an editor change made while results were being assembled.
  assert path.read_text()==before, 'Paper changed concurrently; retry next update'
  temporary = path.with_name('main.tex.table1.tmp')
  temporary.write_text(after);temporary.replace(path)
 return hashlib.sha256(after.encode()).hexdigest()


def update():
 generated = [('TABLE1',table1()),('RESULTS',results()),('RUNTIME',runtime())]
 hashes = {str(path.relative_to(REPO)):update_paper(path, generated) for path in PAPER_COPIES}
 record = {'updated_utc':datetime.now(timezone.utc).isoformat(), 'scope':'Experiments generated blocks only',
  'canonical_paper':str(PAPER.relative_to(REPO)), 'paper_sha256':hashes[str(PAPER.relative_to(REPO))],
  'paper_copies_sha256':hashes, 'outside_experiments_unchanged':True}
 (ROOT/'paper_updates').mkdir(exist_ok=True)
 (ROOT/'paper_updates/status.json').write_text(json.dumps(record,indent=2)+'\n')


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
 while True:
  update()
  if not args.watch:return
  post=ROOT/'post_table1/status.json'
  if post.exists() and read(post).get('state')=='complete':return
  time.sleep(60)


if __name__=='__main__':main()
