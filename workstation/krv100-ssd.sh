#!/usr/bin/env bash
set -euo pipefail
status_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$status_repo/workstation/env.local.sh"
"$POLICY_PYTHON" - <<'PY'
import json, os, re
from pathlib import Path
work = Path(os.environ['KRVLA_WORK'])
def read(path):
    return json.loads(path.read_text()) if path.exists() else {}
download = read(work/'training/dom-full/download_status.json')
training = read(work/'training/krv100-full-20261010/training_status.json')
profiling = read(work/'training/krv100-full-20261010/profiling/profiling_status.json')
transfer = read(work/'training/dom-full/palmetto_transfer_status.json')
backfill = read(work/'training/dom-full/palmetto_backfill_status.json')
exclusions = read(work/'training/krv100-full-20261010/approved_exclusions.json')
if transfer.get('phase') in ('transferring','verifying_copied_files'):
    print('DOM data: importing from Palmetto; checksum counters update during verification')
else:
    print('DOM download:', download.get('phase', 'not started'))
if transfer:
    import_phase = transfer.get('phase','unknown')
    if import_phase == 'finishing_missing_files_from_hub':
        import_phase = 'import complete; HF gaps filled' if download.get('complete') else 'import complete; filling HF gaps'
    print('Palmetto import:', import_phase)
if download.get('selected_cameras'):
    print('Required cameras:', ', '.join(download['selected_cameras']))
if backfill:
    print('Palmetto completion:',backfill.get('phase','unknown'))
    if backfill.get('phase','').startswith('verifying_palmetto_checksums'):
        log_path = work/'logs/dom-palmetto-backfill.log'
        if log_path.exists():
            with log_path.open('rb') as log:
                log.seek(max(0, log_path.stat().st_size - 128*1024))
                lines = log.read().decode(errors='replace').splitlines()
            for line in reversed(lines):
                try:
                    progress = json.loads(line)
                except ValueError:
                    continue
                if progress.get('phase') == 'remote_checksum_verification':
                    checked = progress['files_checked']
                    total = backfill.get('required_files', download.get('files', 0))
                    print(f"Palmetto checksums: {checked:,}/{total:,} files checked; "
                          f"{progress['verified_files']:,} matched")
                    break
if download.get('inventory_complete'):
    print(f"Checksum-verified: {download['verified_files']:,}/{download['files']:,} files; "
          f"{download['verified_bytes']/1e9:.2f}/{download['bytes']/1e9:.2f} GB")
else:
    print(f"Inventory: {download.get('listed_directories',0)}/{download.get('total_directories',0)} directories")
print('KRVLA + Velocity:', training.get('phase', 'not queued'))
control=read(work/'training/krv100-full-20261010/training_control.json')
if control:
    print('Requested training budget:', control['stop_after_completed_epochs'], 'completed epoch(s)')
evaluation_request=read(work/'training/krv100-full-20261010/epoch1_table1_request.json')
if evaluation_request:
    evaluation_status=read(Path(evaluation_request['results'])/'suite_status.json')
    print('Epoch-one Table 1 evaluation:', evaluation_status.get('state','queued'))
    print('Epoch-one Table 1 results:', evaluation_request['results'])
if training.get('phase') == 'freezing_dataset_and_training_statistics':
    preparation_log = work/'training/krv100-full-20261010/dataset-preparation.log'
    if preparation_log.exists():
        with preparation_log.open('rb') as log:
            log.seek(max(0, preparation_log.stat().st_size-16*1024))
            lines = log.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:
                progress = json.loads(line)
            except json.JSONDecodeError:
                continue
            if progress.get('phase') in ('reusing_phase_cache', 'recalculating_training_statistics'):
                print(f"Preparation: {progress['phase']}; {progress['episodes']:,}/{progress['total']:,} episodes")
                break
if training.get('error'):
    print('Training error:', training['error'])
if training.get('exclusion_approval_pending'):
    print('Awaiting approval for additional short-video episodes:',
          ', '.join(map(str, training['proposed_additional_exclusions'])))
if training.get('phase') in ('training', 'failed'):
    progress_log = work/'training/krv100-full-20261010/training.log'
    checkpoint = read(work/'training/krv100-full-20261010/family/runs/krv100/latest.json')
    if progress_log.exists():
        with progress_log.open('rb') as log:
            log.seek(max(0, progress_log.stat().st_size-128*1024))
            recent = log.read().decode(errors='replace')
        updates = re.findall(r'krv100 ranks=(\d+) epoch=([\d.]+) updates=(\d+) samples=(\d+) loss=(\S+) rate=([\d.]+)', recent)
        if updates:
            ranks, epoch, count, samples, loss, rate = updates[-1]
            print(f"Latest logged optimizer progress: {int(count):,} updates; {int(samples):,} samples; "
                  f"epoch {epoch}; loss {loss}; {rate} samples/s; {ranks} GPUs")
        elif checkpoint and training.get('phase') == 'training':
            print('Training startup: W&B and initial checkpoint ready; loading data-loader workers (no optimizer update logged yet)')
        elif training.get('phase') == 'training':
            print('Training startup: loading datasets/model and initializing W&B (no optimizer update logged yet)')
    if checkpoint:
        saved = checkpoint.get('progress', {})
        print(f"Latest saved checkpoint: {saved.get('updates',0):,} updates; {saved.get('samples',0):,} samples")
if exclusions:
    print(f"Training exclusions: {len(exclusions['episodes'])} user-approved invalid-video episodes; "
          f"{exclusions['excluded_frames']:,} frames")
    print(f"Retained training dataset: {exclusions['retained_episodes']:,} episodes; "
          f"{exclusions['retained_frames']:,} frames across all splits")
for item in training.get('required', []):
    print('Waiting for:', item)
print('Profiling:', profiling.get('phase', 'pending training prerequisites'))
if profiling.get('trials_recorded'):
    print('Profile trials:', profiling['trials_recorded'])
if profiling.get('cached_trials_reused'):
    print('Completed profile trials reused:', profiling['cached_trials_reused'])
if profiling.get('selected_runtime'):
    print('Selected runtime:', json.dumps(profiling['selected_runtime'],sort_keys=True))
    print('Measured samples/s:', profiling['samples_per_second'])
print('Training folder:', work/'training/krv100-full-20261010')
PY
