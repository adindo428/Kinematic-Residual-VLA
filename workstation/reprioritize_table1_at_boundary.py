#!/usr/bin/env python3
"""Preserve active scored phases, then reload the explicitly prioritized queue."""
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import time

import run_low_clutter_table1 as original
import upgrade_table1_latency_guard as maintenance
from render_table1_live_status import render

ROOT = original.ROOT
OUT = ROOT / 'checkpoint_audit'


def main():
    catalog = original.read(ROOT / 'catalog.json')
    original.validate(catalog)
    assert catalog['priority_checkpoint'] == catalog['queue_order'][0] == 'old_krv100'
    parent = int(maintenance.prop('MainPID'))
    assert parent > 1 and maintenance.prop('ActiveState') == 'active'
    frozen = False
    targets = []

    def interrupted(signum, frame):
        raise InterruptedError('Priority reload interrupted')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # Freeze the controller only; policy and simulation children finish their
        # current scored phases autonomously. Waiting lanes cannot start work.
        os.kill(parent, signal.SIGSTOP)
        frozen = True
        state = original.read(ROOT / 'status.json')
        for row in state.get('active_evaluations', []):
            assert row['phase'] == 'full' and row['state'] == 'evaluating'
            folder = ROOT / 'results' / row['model'] / 'full' / f"{original.DIMS.index(row['dimension']):02d}_{row['dimension']}"
            targets.append(dict(row, folder=str(folder), client_pid=maintenance.child_client(parent, row['image_port'])))
        assert targets, 'No active phase identified; resume controller and inspect boundary'
        original.write(OUT / 'priority_reload_status.json', {
            'state': 'draining_current_phase', 'priority_checkpoint': 'old_krv100',
            'requested_utc': datetime.now(timezone.utc).isoformat(), 'controller_pid': parent,
            'targets': targets, 'preserve_current_phase': True})
        started = time.monotonic()
        while any(maintenance.process_state(t['client_pid']) not in ['Z', 'gone'] for t in targets):
            assert int(maintenance.prop('MainPID')) == parent, 'Controller changed'
            live = []
            for target in targets:
                if maintenance.process_state(target['client_pid']) in ['Z', 'gone']:
                    continue
                records = Path(target['folder']) / 'trial_results.jsonl'
                count = len(records.read_text().splitlines()) if records.exists() else 0
                live.append({**{k: target[k] for k in ['lane', 'model', 'phase', 'dimension', 'image_port']},
                    'state': 'evaluating', 'completed_records_in_phase': count,
                    'phase_trial_target': 200, 'checkpoint_trial_target': 1800})
            primary = live[0] if live else {}
            original.status('evaluating', **{k:v for k,v in primary.items() if k != 'state'},
                execution_mode='draining_for_oldkrv_priority', concurrency=len(live),
                planned_maintenance=True, active_evaluations=live, lanes=live,
                priority_checkpoint='old_krv100')
            render(ROOT, True)
            if time.monotonic() - started > 14400:
                raise TimeoutError('Current phase did not drain in four hours')
            time.sleep(2)
        for target in targets:
            original.verify(Path(target['folder']), ROOT / 'results' / target['model'],
                ROOT / 'results' / target['model'] / 'test_lists' / (target['dimension'] + '.txt'), 20)
        subprocess.run(['systemctl', '--user', 'stop', '--no-block', maintenance.SERVICE], check=True)
        time.sleep(1)
        os.kill(parent, signal.SIGCONT)
        frozen = False
        subprocess.run(['systemctl', '--user', 'stop', maintenance.SERVICE], check=True)
        maintenance.ensure_sidecar(maintenance.SERVICE.removesuffix('.service'),
            'exec /bin/bash workstation/run-parallel-low-clutter-table1.sh', ROOT / 'parallel-controller.log')
        prefix = 'source workstation/env.local.sh; exec "$KRVLA_WORK/envs/eval20k/bin/python" -u '
        for unit, script, log in [
            ('krvla-table1-runtime-metrics-20261104', prefix+'workstation/capture_table1_runtime.py', ROOT/'runtime_metrics/collector.log'),
            ('krvla-latency-monitor-20261104', prefix+'workstation/monitor_table1_latency.py', ROOT/'runtime_health/monitor.log'),
            ('krvla-table1-live-wandb-20261102', 'exec /bin/bash workstation/run-table1-live-wandb.sh', ROOT/'live-wandb.log')]:
            maintenance.ensure_sidecar(unit, script, log)
        original.write(OUT / 'priority_reload_status.json', {'state': 'reloaded',
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'priority_checkpoint': 'old_krv100',
            'verified_preserved_phases': targets})
    except BaseException as error:
        if frozen:
            try:
                os.kill(parent, signal.SIGCONT)
            except ProcessLookupError:
                pass
        original.write(OUT / 'priority_reload_failure.json', {'error': repr(error), 'controller_unfrozen': True})
        raise


if __name__ == '__main__':
    main()
