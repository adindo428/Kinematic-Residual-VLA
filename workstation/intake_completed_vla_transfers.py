#!/usr/bin/env python3
"""Verify completed transfers and atomically organize them on the same SSD."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / 'scratch/dom45/training/palmetto-updated-20261102'
ORIGINAL = REPO / 'scratch/dom45/evaluations/multivla20k-20261029/catalog.json'
MODELS = ['vlash_kr', 'vlash_baseline', 'pi05_baseline']
ELIGIBLE = ['vlash_kr', 'pi05_baseline']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    import torch
    os.nice(10)
    original = json.loads(ORIGINAL.read_text())
    ROOT.mkdir(parents=True, exist_ok=True)
    observed = {}
    records = {}
    while True:
        for name in MODELS:
            folder = ROOT / name
            target = folder / 'latest.pt'
            receipt = folder / 'transfer_receipt.json'
            source = REPO / 'scratch' / (name + '_latest.pt')
            if receipt.exists():
                record = json.loads(receipt.read_text())
                assert target.exists() and target.stat().st_size == record['bytes']
                records[name] = record
                continue
            if not source.exists():
                records[name] = {'state': 'waiting_for_transfer', 'source': str(source)}
                continue
            before = source.stat()
            signature = (before.st_size, before.st_mtime_ns)
            if observed.get(name) != signature:
                observed[name] = signature
                records[name] = {'state': 'waiting_for_stable_transfer', 'source': str(source), 'bytes': before.st_size}
                continue
            try:
                with zipfile.ZipFile(source) as archive:
                    assert archive.namelist(), 'Empty checkpoint archive'
                saved = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
                identity, progress = saved['identity'], saved['progress']
                assert identity == original['models'][name]['identity'], 'Checkpoint model identity mismatch'
                assert saved['model'], 'Checkpoint has no model state'
                del saved
            except (RuntimeError, zipfile.BadZipFile, EOFError) as error:
                records[name] = {'state': 'waiting_for_complete_archive', 'source': str(source),
                                 'bytes': source.stat().st_size, 'reason': str(error)}
                continue
            assert not target.exists(), 'Preserve existing target for inspection: ' + str(target)
            folder.mkdir(parents=True, exist_ok=True)
            assert before.st_dev == folder.stat().st_dev, 'Expected an atomic move on the same SSD'
            print('Hashing completed transfer:', name, before.st_size, flush=True)
            digest = sha256(source)
            after = source.stat()
            if (after.st_size, after.st_mtime_ns) != signature:
                records[name] = {'state': 'waiting_for_stable_transfer', 'source': str(source), 'bytes': after.st_size}
                continue
            source.rename(target)
            moved = target.stat()
            assert (moved.st_dev, moved.st_ino, moved.st_size) == (after.st_dev, after.st_ino, after.st_size)
            complete = progress['epoch'] >= 10 and progress['cursor'] == 0
            record = {'state': 'organized', 'source': str(source), 'checkpoint': str(target),
                      'bytes': moved.st_size, 'mtime_ns': moved.st_mtime_ns, 'sha256': digest,
                      'identity': identity, 'progress': progress, 'same_ssd_atomic_move': True,
                      'source_removed': not source.exists(), 'complete_epoch10': complete,
                      'evaluation_eligible': name in ELIGIBLE and complete,
                      'hold_reason': None if name in ELIGIBLE and complete else 'User says training unfinished; await a final checkpoint before evaluation',
                      'verified_utc': datetime.now(timezone.utc).isoformat()}
            write(folder / 'latest.json', {'progress': progress, 'identity': identity, 'file': str(target)})
            write(receipt, record)
            records[name] = record
            print('Organized:', name, progress, 'evaluation eligible:', record['evaluation_eligible'], flush=True)
        done = all(records.get(name, {}).get('state') == 'organized' for name in MODELS)
        write(ROOT / 'transfer_status.json', {'state': 'complete' if done else 'waiting_for_transfers',
                                             'updated_utc': datetime.now(timezone.utc).isoformat(),
                                             'models': records, 'evaluation_order': ELIGIBLE,
                                             'held_models': ['vlash_baseline', 'pi05_kr']})
        if done or not args.watch:
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
