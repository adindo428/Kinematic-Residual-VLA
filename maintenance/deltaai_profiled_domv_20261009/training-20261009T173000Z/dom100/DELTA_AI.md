# DeltaAI staging

DeltaAI is an ARM64, four-GH200-per-node system. Its documented Slurm GPU
partition is `ghx4`; ACCESS allocations must explicitly include DeltaAI.
Do not submit jobs until the ACCESS exchange, NCSA account, GPU access, storage
quota, and Python/TorchCodec compatibility have been verified on that system.
The confirmed login address is `gh-login01.delta.ncsa.illinois.edu`. The
current user is `aparameshwaran`; authenticate with NCSA Kerberos and Duo.
Do not assume an account or project from another user. Run `accounts` after
login, set `DVLA_SLURM_ACCOUNT` to the actual billing account, and set
`DVLA_ACCESS_PROJECT` to the project that owns the writable `/work` storage.
A source snapshot should be staged at `/u/aparameshwaran/Kinematic-Residual-VLA`.
It is not a Git clone and has not been pushed through Palmetto. The prior
staging history below belongs to the earlier account and is not a substitute
for validating the new account's Slurm access, writable project, quotas, and
storage.

On DeltaAI, identify your real project serial code and confirm the account
(typically `<project-serial-code>-dtai-gh`),
partition, quota, and job limits with `accounts`, `sinfo`, and `sacctmgr` or
the site support team. Never substitute an ACCESS credit balance for the Slurm
account name. Set up an ARM-compatible Python environment on a compute node:
the Rocky 8/Palmetto environment and x86 wheels are not portable. Validate
`torch`, GPU visibility, `torchcodec.decoders`, LeRobot, video decode,
pretrained checkpoint loading, and one small training step before a full run.
The provided `deltaai_env.sh` loads the available
`python/miniforge3_pytorch/2.12.0` module, then activates a
venv created with a compatible module. The shared module has Torch/W&B but
does not have `lerobot`, `torchcodec`, `easydict`, or `draccus`. Build and
verify a separate project venv on an allocated compute node. DeltaAI rejects
CPU-only Slurm jobs, so use a short `ghx4` one-GPU allocation for that work.
Do not install
`requirements_rocky8.txt` unchanged: its PyTorch pin and binary packages may
conflict with the ARM module. Resolve dependencies on an allocated compute node.

The source contains the 0.10/0.18 m/s split JSON. The DOM episodes and camera
videos need separate prefetching into the DeltaAI `/work/nvme` project path,
with an allowance for storage and time. Do not copy the entire held-out
`ood_test` split merely to start training. The normal training-time metrics are
`iid_eval` (overlaps training) and `ood_val`; neither is closed-loop task
success. `ood_test` is reserved for final evaluation.

From the staged `third_party/DynamicVLA` directory, after a compute-node
smoke test and Hub authentication:

```bash
export DVLA_ACCESS_PROJECT=<writable-project-name>
export DVLA_SLURM_ACCOUNT=<slurm-billing-account>
export DVLA_PYTHON_ENV=/work/nvme/$DVLA_ACCESS_PROJECT/$USER/dynamicvla/venv-dynamicvla-torch212
export DVLA_HF_CHECKPOINT_REPO=adi2440/dom-vla-training
export DVLA_WANDB_ENTITY=<your-wandb-username-or-entity>
export DVLA_SECRETS_FILE=/u/$USER/.config/kinematic-residual-vla/secrets.env
mkdir -p logs
sbatch --export=ALL slurm/deltaai_full_finetune.sbatch
```

Create the secrets file outside the repository with `umask 077` and
`chmod 600`, containing `WANDB_API_KEY=...` and `HF_TOKEN=...` (no `export`
required). Do not paste these values in chat or commit them. The job loads
them only after checking the file's permissions; W&B uses the project key,
`WANDB_CONFIG_DIR` points to a separate project directory, and
`DVLA_WANDB_ENTITY` overrides the legacy YAML entity. The existing account's
`~/.netrc` is not modified. Obtain a Hugging Face fine-grained token with
write permission to the model repo; never run a global `hf auth login` for
this shared login.

The template requests one four-GH200 node for 24 hours. It passed
`sbatch --test-only` but was **not submitted**. Check the actual
`ghx4` availability and allocation charge before submission; batch size 8
and accumulation 32 are conservative guesses, not tested GH200 settings.
Submit from the project root so relative Slurm log paths resolve. Source
`scripts/deltaai_env.sh` only after setting the project and Python environment.
Set `DVLA_ENV_SCRIPT` to that file when using the generic paired launcher.

Create a Hub model repository before running the pair; verify that the
fine-grained token can read and write it. Hub checkpoint sync
is **opt-in** via `DVLA_HF_CHECKPOINT_REPO`. A public repository exposes the
model and optimizer state to everyone; use a private repository if that is
not intended. The model repository stores `baseline/epoch-NNNN` and `residual/epoch-NNNN`, each
with `config.json`, `model.safetensors`, and `training_state.pt`; the small
`latest.json` marker is written only after those three uploads complete.
At startup the paired launcher restores each variant's latest checkpoint,
or keeps a local checkpoint if it is at least as new. It fails closed on
network/authentication errors or an older/incomplete local checkpoint rather
than silently starting over. Local checkpoints are still used for training.
Keep the repo private: optimizer states can be huge, and every epoch consumes
Hub storage and outbound network bandwidth. Ensure Hub access from the
compute nodes before enabling sync. A different GPU count or batch geometry
is not a bitwise-identical resume; keep hyperparameters aligned when comparing
baseline and residual.
