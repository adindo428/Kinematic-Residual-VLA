#!/usr/bin/env python3
"""Keep the existing W&B evaluation run synchronized with both Table 1 displays."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from table1_display_state import LOW, FULL


def write(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def parse_table(text):
    lines = [line for line in text.splitlines() if line.startswith('|')]
    if len(lines) < 3:
        raise ValueError('Comparison table is missing its header or rows')
    columns = [cell.strip() for cell in lines[0].strip('|').split('|')]
    if columns[:2] != ['Method', 'Status']:
        raise ValueError('Unexpected comparison table columns')
    rows = [[cell.strip() for cell in line.strip('|').split('|')] for line in lines[2:]]
    if any(len(row) != len(columns) for row in rows):
        raise ValueError('Incomplete comparison table write; retry next poll')
    if not rows:
        raise ValueError('Comparison table has no methods')
    return {'columns': columns, 'data': rows}


def snapshot():
    texts = {name: (root / 'table1_comparison.md').read_text()
             for name, root in [('low_clutter', LOW), ('full_scale', FULL)]}
    tables = {name: parse_table(text) for name, text in texts.items()}
    signature = hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest()
    return texts, tables, signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error('--interval must be at least 5 seconds')
    if args.preview_only:
        _, tables, signature = snapshot()
        print(json.dumps({'signature': signature, 'tables': tables}, indent=2))
        return
    state = LOW / 'wandb-comparisons'
    state.mkdir(exist_ok=True)
    with (state / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import wandb
        record_path = LOW / 'wandb_live_run.json'
        record = json.loads(record_path.read_text())
        settings = wandb.Settings(disable_git=True, disable_code=True, save_code=False,
                                  disable_job_creation=True, x_disable_meta=True,
                                  x_disable_stats=True, x_disable_machine_info=True, console='off')
        run = wandb.init(entity=record['entity'], project=record['project'], id=record['id'],
                         name=record['name'], resume='allow', job_type='evaluation',
                         mode='online', settings=settings)
        previous = None
        try:
            while True:
                try:
                    texts, tables, signature = snapshot()
                except (OSError, ValueError) as error:
                    if args.once:
                        raise
                    print(f'Waiting for complete comparison tables: {error}', flush=True)
                    time.sleep(args.interval)
                    continue
                now = datetime.now(timezone.utc).isoformat()
                if signature != previous:
                    metrics = {f'Table1/{name}': wandb.Table(**table) for name, table in tables.items()}
                    # Preserve the original dashboard key for existing panels.
                    metrics['Table1/results'] = wandb.Table(**tables['low_clutter'])
                    run.log(metrics)
                    for name, table in tables.items():
                        run.summary[f'Table1/{name}_rows'] = len(table['data'])
                        run.summary[f'Table1/{name}_status_counts'] = {
                            status: sum(row[1] == status for row in table['data'])
                            for status in sorted({row[1] for row in table['data']})}
                        folder = state / 'reports' / name
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / 'table1_comparison.md'
                        path.write_text(texts[name])
                        run.save(str(path), base_path=str(state), policy='now')
                    run.summary['Table1/last_changed_utc'] = now
                    run.summary['Table1/content_sha256'] = signature
                    previous = signature
                    print(f'{now} Published low-clutter and full-scale tables ({signature[:12]})', flush=True)
                run.summary['Table1/last_checked_utc'] = now
                run.summary['Table1/publisher_state'] = 'watching' if not args.once else 'synced_once'
                run.summary['Table1/refresh_interval_seconds'] = args.interval
                run.summary['Table1/source'] = 'Current local comparison displays; provisional cells retain denominators.'
                record.update(state='watching_comparisons' if not args.once else 'synced_comparisons',
                              updated_utc=now, tables=['low_clutter', 'full_scale'],
                              payload_scope='Aggregate tables and Markdown reports, including provisional labels; no raw trial or video uploads.')
                write(record_path, record)
                write(state / 'status.json', {'state': record['state'], 'updated_utc': now,
                      'signature': signature, 'url': run.url, 'interval_seconds': args.interval})
                if args.once:
                    break
                time.sleep(args.interval)
        finally:
            run.finish()


if __name__ == '__main__':
    main()
