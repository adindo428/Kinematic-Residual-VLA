# Kinematic Residual VLA

Training and closed-loop evaluation code for DynamicVLA and our kinematic residual variants. This repository includes the velocity-conditioned Coordinate model, the matched tracking experiments, and the current six-variant DOM training implementation.

**For the lab computer, start with [Local DOM evaluation](evaluation/official_dom/README.md).** It evaluates the released DynamicVLA DOM checkpoint on the actual manipulation benchmark without retraining. The launcher runs on Linux with a local Isaac Lab installation; it does not require Slurm.

| Workflow | Code and instructions |
|---|---|
| Released DynamicVLA checkpoint, original DOM manipulation benchmark | [evaluation/official_dom](evaluation/official_dom/README.md) |
| Four trained models, closed-loop hover tracking and nine stress dimensions | [experiments/tracking75](experiments/tracking75/README.md) |
| Tracking training: baseline, velocity, Coordinate, Coordinate+Velocity | [Training guide](docs/TRAINING.md#tracking-training-75-epochs) |
| DOM training: six matched variants, single GPU and 4/8 GPU DDP | [Training guide](docs/TRAINING.md#dom-training-100-epochs) and [DOM100 details](third_party/DynamicVLA/DOM100.md) |
| Dataset collection and offline checkpoint evaluation | [Collector](third_party/Data_Collection_Dynamic_Objects/README.md) |
| Real-robot CPU tracking, shadow control, and SO101-to-ALOHA collection plan | [Real-robot prototype](real_robot/README.md) |
| Palmetto and DeltaAI deployment | [Cluster guide](docs/CLUSTERS.md) |

**For the Linux RTX PRO 6000 machine, use the [workstation setup and VS Code guide](workstation/README.md).** It selects CUDA 12.8 for Blackwell, keeps policy/simulator environments separate, and includes [the research handoff](workstation/HANDOFF.md) and portable grasp diagnostics.

## Get the code

```bash
git clone https://github.com/adindo428/Kinematic-Residual-VLA.git
cd Kinematic-Residual-VLA
export KRVLA_REPO="$PWD"
```

Two source trees are intentional. `third_party/DynamicVLA` contains current DOM training development. `experiments/tracking75/source` preserves the tracking model/trainer used for the 75-epoch comparison. Load tracking weights with the latter. The official manipulation evaluator fetches a separate pinned, unmodified upstream checkout; it does not use either research fork as the baseline.

## What is included

Model implementations, trainers, dataset adapters, configs/splits, checkpoint loading, simulator interfaces, metrics, plots/video helpers, tests, and cluster launchers. The current Coordinate+Velocity variant conditions its neural residual policy on current object velocity as well as using the analytic velocity displacement. Residual independence from velocity is not assumed in the implemented model.

Large datasets, scene assets, videos, environments, credentials, and trained weights are downloaded or copied separately. The official evaluation guide downloads its public checkpoint. Our trained tracking checkpoints must be copied from Palmetto with their config and verification marker; see the tracking README.

## Metric definitions

The **official DOM manipulation benchmark** reports actual task completion: successful pick/place trials divided by trials, plus mean simulation time including failed trials. The full suite is 90 released configurations × 20 repeats = 1,800 trials.

The **tracking adaptation** reports the mean per-episode percentage of time within **3 cm of the moving hover target**, including the initial second. Early termination earns zero credit for the unexecuted episode remainder. Completion is reported separately. Its nine dimensions are tracking proxies in one room, not reproduction of the original manipulation benchmark. Four models × 1,800 trials = 7,200 rollouts.

## Hardware and verification

Evaluation uses one simulated environment and separate policy/simulation Python environments. An RTX 3080 should be tested with the documented pilot first; successful code installation does not establish that the simulator and policy fit together in VRAM. Headless evaluation still renders the cameras. GPU memory and runtime information are saved by the local DOM launcher. Keep camera resolution, timing, checkpoint, and streaming settings fixed for benchmark comparisons.

CPU verification instructions and the limits of the workstation validation are in [Verification](docs/VERIFICATION.md). Full DOM training is intended for larger GPUs; it is not a recommended workload for the 3080.

## Attribution and research history

The base implementation is [DynamicVLA](https://github.com/hzxie/DynamicVLA), associated with [the paper](https://arxiv.org/abs/2601.22153), [DOM](https://huggingface.co/datasets/hzxie/DOM), and [released weights](https://huggingface.co/hzxie/dynamic-vla-DOM). Retain the upstream [license](third_party/DynamicVLA/LICENSE); included forks keep their original license files. Data/assets/model distributions have their own terms.

The paper draft and [knowledge base](KNOWLEDGE_BASE.md) are research material, not a claim of reproduced published results. The previous recovery-oriented README is retained as [historical context](docs/RESEARCH_HISTORY.md).
