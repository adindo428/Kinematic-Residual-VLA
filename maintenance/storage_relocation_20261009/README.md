# Home quota relief and durable evaluation archive

Private archive: https://huggingface.co/datasets/adi2440/krvla-evaluation-results

Home usage before work: **220.73 GiB / 250 GiB**, reported by `checkquota` on 2026-10-09. Scratch usage: **1.34 TiB / 10 TiB**. Initial candidates: pip cache 8.1 GB, uv cache 29 GB; current evaluation result folders about 26 GB. Logical directory sizes are not a guarantee of exact quota savings because package files may be hardlinked to installed environments.

Latest checked home quota: **211.43 GiB / 250 GiB** at 2026-10-09 18:53 UTC, down **9.30 GiB** from the initial check. Pip relocation and Coordinate+Velocity archival are verified complete; uv and remaining completed-result archives continue in their CPU jobs. `checkquota` reads `/hpc/ondemand/vast_quotas.csv`, so reports reflect periodic server data rather than an immediate rescan.

## What moves

- `/home/aparame/.cache/pip` → `/scratch/aparame/storage_relocation_20261009/caches/pip`
- `/home/aparame/.cache/uv` → `/scratch/aparame/storage_relocation_20261009/caches/uv`
- Completed verified folders from Coordinate+Velocity comparison run 16787490, tracking Table 1 adaptation 20261008, and official DOM manipulation evaluation 20261009 → the same result-relative paths under `/scratch/aparame/storage_relocation_20261009/results/`

The old home directory paths become symlinks after an rsync checksum comparison. Installed conda environments stay in home. Package caches can be downloaded again if purged. W&B artifact cache, source code, unrelated projects, unverified results, and actively running trial folders are not moved by these scripts.

Completed result folders are first packed into `.tar.gz` files with per-file SHA256 manifests. Every archive member is read back and checked. The private Hugging Face commit's archive SHA256 and size must match before the home working copy is relocated. The source is checked again against the archive before relocation. Failed uploads leave the original home data intact.

## Jobs

- Cache relocation: `16886459`
- Initial archive and completed-result relocation: `16886460`
- Final archive after the initial pass and both benchmark arrays end: `16886461`

The final job waits for benchmark arrays 16830026 and 16868047. It also waits up to two hours for summary jobs 16830027 and 16868048 before refreshing browsable summary metadata. If summaries fail or remain pending longer, raw completed results are still archived and the status records absent final summaries. Partial/unverified trial folders remain in home.

## Status and restoration

Detailed logs: `/scratch/aparame/storage_relocation_20261009/logs/`

Relocation receipts: `/scratch/aparame/storage_relocation_20261009/journal/`

Cloud verification receipts: `/scratch/aparame/storage_relocation_20261009/cloud_journal/`

Archive inventory and browsable plots/metrics: private HF repository `archive_index.json`, plus each suite's metadata directory. `latest_archive_status.json` in the scratch relocation root reports failures and whether final summaries exist. The small launch/code records remain in this home directory.

Download an archive to the lab computer with your authorized Hugging Face login:

```python
from huggingface_hub import hf_hub_download
p = hf_hub_download('adi2440/krvla-evaluation-results',
    'coordinate_velocity/16787490.tar.gz', repo_type='dataset')
print(p)
```

Extract into a fresh directory. Check the downloaded archive hash against `archive_index.json`; the internal/external `ARCHIVE_MANIFEST.json` records each stored file's hash and size. Cloud paths for tracking are `tracking_table1/20261008/full/000.tar.gz` etc.; DOM paths are `official_dom_table1/20261009/full/00_CR.tar.gz` etc.

## Scratch retention

Scratch is unbacked temporary space. Clemson purges files if access, modification and metadata-change timestamps are all older than 30 days. Artificial touching/copying solely to avoid purge is prohibited. The permanent copy for results is the private HF archive, not the home symlink or scratch archive. Do not assume reading a parent directory refreshes all its files. Keep important future result directories archived after completion, and restore needed working copies from HF if scratch has purged them.

Policy: https://docs.rcd.clemson.edu/palmetto/storage/spaces/scratch/

Clemson Box was not configured (`rclone` and a Box remote were absent). HF was chosen because an existing write-authorized account is available. Box can be added later if preferred; no Box login or credentials were copied.
