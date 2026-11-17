"""Keep archived reference rows visible without restarting an active evaluator."""
import argparse
import ctypes
import fcntl
import os
import re
from pathlib import Path
import select
import struct

ROOT = Path(__file__).resolve().parents[1] / 'scratch/dom45/evaluations/krv-full-dom-ablation-20261115'
MARKER = '<!-- historical-krv-results -->'


def format_current_rows(text):
    lines = []
    for line in text.splitlines():
        if line.startswith('|'):
            cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
            if cells[0] in ('full_vpd', 'full_vpd_vi'):
                continue
            if cells[0].lower() == 'dom':
                cells[0] = 'DOM'
                for i in range(1, len(cells)):
                    match = re.fullmatch(r'(\d+)/(\d+)', cells[i])
                    if match:
                        successes, trials = map(int, match.groups())
                        rate = f'{100*successes/trials:.2f}%' if trials else '—'
                        cells[i] = f'{rate} ({successes}/{trials})'
                line = '| ' + ' | '.join(cells) + ' |'
        if line.startswith('Cells are successes/completed trials;'):
            line = 'DOM cells show success percentage (successes/completed trials); each category targets 100 trials. — means no completed trials.'
        lines.append(line)
    return '\n'.join(lines)


def append_history(target=None):
    target = Path(target) if target else ROOT / 'live_results.md'
    section = ROOT / 'history-audit/live-history-section.md'
    if not target.exists() or not section.exists():
        return
    old = target.read_text()
    new = format_current_rows(old.split(MARKER, 1)[0]).rstrip() + '\n\n' + section.read_text().strip() + '\n'
    if old != new:
        target.write_text(new)


def watch():
    with (ROOT / 'history-report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1')
        try:
            # Old in-memory renderer closes the report after each update.
            wd = libc.inotify_add_watch(fd, os.fsencode(ROOT), 0x8 | 0x80)
            if wd < 0:
                raise OSError(ctypes.get_errno(), 'inotify_add_watch')
            append_history()
            while True:
                ready, _, _ = select.select([fd], [], [], 30)
                if ready:
                    data = os.read(fd, 65536)
                    offset = 0
                    while offset < len(data):
                        _, _, _, size = struct.unpack_from('iIII', data, offset)
                        name = data[offset+16:offset+16+size].split(b'\0', 1)[0]
                        offset += 16 + size
                        if name == b'live_results.md':
                            append_history()
                pidfile = ROOT / 'runner-MG-then-DOM.pid'
                if pidfile.exists():
                    try:
                        os.kill(int(pidfile.read_text()), 0)
                    except ProcessLookupError:
                        append_history()
                        return
        finally:
            os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    watch() if args.watch else append_history()
