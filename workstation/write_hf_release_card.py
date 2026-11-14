#!/usr/bin/env python3
"""Write the public checkpoint index, usage examples, and upstream notices."""
import json
from pathlib import Path
from prepare_hf_checkpoint_release import ROOT, STAGE, REPO_ID


def main():
    catalog=json.loads((STAGE/'model-index.json').read_text())
    lines=['---','language: en','pipeline_tag: robotics','library_name: pytorch','license: other',
           'license_name: mixed-upstream-licenses',f'license_link: https://huggingface.co/{REPO_ID}/blob/main/licenses/NOTICES.md',
           'tags:','- vision-language-action','- kinematic-residual','- flow-matching','- robotics',
           'base_model:','- hzxie/dynamic-vla-DOM','- lerobot/pi05_base','- lerobot/smolvla_base','---','',
           '# Kinematic Residual VLA Checkpoints','',
           'Research checkpoints for evaluating and continuing fine-tuning of DynamicVLA/DOM, DOM with velocity conditioning, '
           'KRV100, SmolVLA, VLASH, and π0.5, including available KR variants. Baselines are included for matched comparisons.','',
           '**Upload status:** see `upload-status.json`. The index lists the intended complete release; a checkpoint is available '
           'when all files listed for it in `files.json` appear in the repository.','',
           '## Training histories and naming','',
           '- `full`: KRV100 epoch 3 and after epoch 4 plus part of epoch 5; audited complete retained split with 203,151 training episodes.',
           '- `large-audited-197198`: DOMV epochs 1, 3 (validation-best), 5, and 6 (final/latest). Its original available-data audit retained 197,198 training episodes and 97.066% of published episodes overall. It is not the same full-data split as KRV100.',
           '- `full-historical`: retained earlier full-data KRV100 generations, preceding the DeltaAI continuation.',
           '- `20k-fresh`: DOM/KRV100 fine-tuning on the common 20,000-episode training split, starting from released pretrained weights.',
           '- `20k-continuation`: other VLA snapshots continued on the 20k split; their prior full-data exposure is retained in provenance.',
           '- `mixed-full-then-20k`: historical KRV100 runs that switched from full-data training to the smaller split. These are not full-only checkpoints.',
           '- `epoch-NNN-partial-next` means N epochs completed plus part of the next epoch. Update counters are authoritative.',
           '- Gate variants are evaluation configurations, not separate trained weights. Original checkpoint configs and KR math are preserved.','',
           'The resumed π0.5 + KR checkpoint has six completed epochs plus part of epoch 7. The latest retained VLASH baseline '
           'is also incomplete; neither is represented as a finished ten-epoch model. Validation-best selection is only claimed '
           'where recorded provenance supports it.','',
           '## Checkpoints','',
           '| Model | Training scope | Completed epoch | Updates | Partial next epoch | Directory |',
           '|---|---|---:|---:|---|---|']
    for m in catalog['checkpoints']:
        p=m['progress'];lines.append(f'| {m["model"]} | {m["scope"]} | {p["epoch"]} | {p["updates"]} | {m["partial_next_epoch"]} | `{m["path"]}` |')
    lines += ['', '## Download only the checkpoint you need', '',
              'Install `huggingface_hub`, then bootstrap the small download helper:', '', '```bash',
              f'hf download {REPO_ID} tools/checkpoint_tools.py model-index.json --local-dir kr_models',
              'cd kr_models',
              'python tools/checkpoint_tools.py krv100-full-best --local-dir . --training-state', '```', '',
              'For native checkpoints, omit `--training-state` to download evaluation weights without optimizer state. '
              'Adapter `.pt` checkpoints contain model and optimizer state together, so their full file is downloaded. '
              'Use any alias below, or an exact checkpoint ID from `model-index.json`:', '',
              '```json', json.dumps(catalog['aliases'],indent=2), '```','',
              'A complete snapshot is approximately 313 GB before backend deduplication. Selective downloading is intended.', '',
              '## Load a checkpoint', '',
              'Use separate environments: native DOM/KRV uses `requirements-native.txt` (Transformers 5.2.0, LeRobot 0.3.3); '
              'the other adapters use `requirements-adapters.txt` (Transformers 4.53.3, LeRobot 0.4.1). '
              'Matching patched LeRobot source is packaged under `code/runtime-*` and selected by the loader. '
              'CUDA PyTorch installation depends on the target machine. The frozen native source requires LeRobot support; '
              'the adapter implementation uses the recorded LeRobot and Transformers versions. Load one model family per process.', '',
              '```python', 'from tools.checkpoint_tools import load_model',
              'policy, metadata = load_model(".", "krv100-full-best", device="cuda")', '```', '',
              'The helper restores the complete trained state strictly, resolves tokenizer/config assets locally, and uses '
              'the checkpoint-specific source. These are custom research exports; passing an adapter `checkpoint.pt` directly '
              'to a stock LeRobot `from_pretrained()` call is not the supported loading path.', '',
              'Native KRV evaluation gate variants are in `evaluation_configs/`. For an explicit inference-only variant, '
              'pass, for example, `inference_config="evaluation_configs/krv100-full-best/causal.json"` to `load_model`. '
              'The corresponding trained weights remain shared. Default loading preserves the original checkpoint config.', '',
              '## Continue fine-tuning', '',
              'All released production snapshots include optimizer, scheduler, sample cursor, and RNG state. Native models '
              'store these in `training_state.pt`; adapter models store them inside `checkpoint.pt`.', '',
              '```python', 'from tools.checkpoint_tools import load_model, training_state',
              'policy, metadata = load_model(".", "krv100-full-best", device="cuda", train=True)',
              'state = training_state(".", "krv100-full-best")',
              '# Construct the optimizer with the original parameter groups before restoring:',
              '# optimizer.load_state_dict(state["optimizer"])',
              '# scheduler.load_state_dict(state["scheduler"])', '```', '',
              'The packaged `code/native-*/core/dom_distributed_training.py` and `code/adapter-trainer/train.py` retain '
              'the original optimizer, sampler, RNG-restoration, and training-loss implementations. The native trainer uses '
              '`core.dom_training.make_optimizer`/`make_scheduler`; adapter training uses its `optimizer_for` function. '
              'Distributed RNG state and sample cursors require the recorded rank/world-size conventions for exact continuation. '
              'For a new fine-tuning experiment, load the trained model and create a new optimizer/scheduler instead of claiming exact continuation.', '',
              'Training launchers contain original site-specific paths, admission checks, and experiment fingerprints. '
              'Relocate dataset/checkpoint paths and deliberately configure the new machine before using those launchers. '
              'The portable loader handles weight restoration; it does not bypass those training controls.', '',
              'Training batches must retain the saved action semantics and normalization. Native actions are 20-step chunks '
              'with physical delta pose/gripper targets; use the packaged native dataset preparation/tokenization. Adapter '
              'batches contain the two RGB views, six-dimensional robot state, language task, raw action chunk, padding mask, '
              'current three-dimensional object velocity, and offset. The wrapper constructs residual targets and restores drift. '
              'Do not add KR displacement or the reference pose a second time.', '',
              'For DOM environment collection, the dataset identifier is `hzxie/DOM`, revision '
              '`bbf6baa15f7253746d271a6e958e037afa537b1e`. Exact episode selections, validation anchors, normalization '
              'statistics, velocity statistics, recipes, and dataset fingerprints are included under `training/` and '
              'the per-model `code/adapters/` directories. Raw dataset videos and simulator assets are not duplicated here.', '',
              '## Evaluation compatibility', '',
              'Closed-loop evaluation additionally requires the benchmark simulator and assets. The recorded workstation '
              'runtime was Isaac Sim 4.5.0 and Isaac Lab 2.2.1. Preserve camera views, units, action normalization, '
              '20-step chunks, 25 Hz action time, velocity frame, phase-gate configuration, and streaming timing when reproducing results. '
              'The large-scale checkpoints and smaller-split models have different training histories and should remain separately labeled.', '',
              '## Provenance and licenses', '',
              'Original provenance, recipes, update counters, and source are preserved. `files.json` records file sizes and '
              'SHA-256 checksums; `model-index.json` records canonical checkpoint paths and aliases. Duplicate evaluation '
              'copies, untrained initializations, smoke tests, and deleted retired weights are excluded.', '',
              'Applicable upstream licenses remain in force. DynamicVLA source and derived weights carry S-Lab non-commercial '
              'terms; π0.5/PaliGemma-derived components carry Gemma terms and use restrictions. SmolVLA/SmolVLM and '
              'LeRobot components retain their upstream Apache notices where applicable. See `licenses/NOTICES.md`. '
              'This collection does not relicense all components under a single permissive license.']
    (STAGE/'README.md').write_text('\n'.join(lines)+'\n')
    (STAGE/'licenses/NOTICES.md').write_text('''# Upstream notices

DynamicVLA: https://github.com/hzxie/DynamicVLA and https://huggingface.co/hzxie/dynamic-vla-DOM .
Retain `S-Lab-LICENSE`, including its non-commercial redistribution condition.

Gemma is provided under and subject to the Gemma Terms of Use found at
https://ai.google.dev/gemma/terms . π0.5/PaliGemma derivatives remain subject to
those terms and https://ai.google.dev/gemma/prohibited_use_policy . Copies of
both pages accompany this release. Upstream π0.5 metadata identifies its license
as `gemma`: https://huggingface.co/lerobot/pi05_base .

LeRobot and SmolVLA: https://github.com/huggingface/lerobot and
https://huggingface.co/lerobot/smolvla_base (Apache-2.0).
SmolVLM2: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct (Apache-2.0).
Preserve source copyright headers and upstream license notices. VLASH's frozen
implementation is included as source for reproducibility; its upstream component
licenses and Gemma-derived weight restrictions remain applicable.

No simulator assets or dataset videos are redistributed in this checkpoint collection.
''')
    (STAGE/'.gitattributes').write_text('*.pt filter=lfs diff=lfs merge=lfs -text\n*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    (STAGE/'upload-status.json').write_text(json.dumps({'state':'prepared','complete':False,'checkpoint_count':len(catalog['checkpoints'])},indent=2)+'\n')
    print('Public model card and usage documentation prepared')


if __name__=='__main__':main()
