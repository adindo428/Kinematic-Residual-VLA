#!/usr/bin/env python3
"""Keep requested row replacements stable while an older evaluator publishes."""
from pathlib import Path
import signal
import threading

from table1_display_state import FULL,DOMV,read,write_if_changed

PREFIX='| KRV100_Full epoch 3 (causal gate) |'
OLD_LATEST='KRV100_Full latest (4 epochs + partial epoch 5, gated)'
CAUSAL_LATEST='KRV100_Full latest (4 epochs + partial epoch 5, causal gate)'
CONSTANT_LATEST='KRV100_Full latest (4 epochs + partial epoch 5, constant gate)'


def remove_row(paths):
    for path in paths:
        if not path.exists():continue
        text=path.read_text()
        updated=''.join(line for line in text.splitlines(keepends=True) if not line.startswith(PREFIX))
        updated=updated.replace('| '+OLD_LATEST+' |','| '+CAUSAL_LATEST+' |')
        updated=updated.replace(
            'Constant gate retains KR throughout the episode; the latest checkpoint uses causal grasp gating '
            'and is four completed epochs plus partial epoch five.',
            'Constant gate retains KR throughout the episode. The latest constant- and causal-gate variants '
            'share the same weights: four completed epochs plus partial epoch five. '
            'Latest constant gate is queued for 100 trials per column (900 total); its remaining 900 are deferred.')
        if '| '+CONSTANT_LATEST+' |' not in updated:
            lines=updated.splitlines(keepends=True)
            for i,line in enumerate(lines):
                if line.startswith('| '+CAUSAL_LATEST+' |'):
                    lines.insert(i+1,'| '+CONSTANT_LATEST+' | Pending | '+' | '.join(['pending']*11)+' |\n')
                    updated=''.join(lines);break
        if updated!=text:write_if_changed(path,updated)


def main():
    models=read(FULL/'resident_schedule/ledger.json')['models']
    paths={root/'table1_comparison.md' for root in [FULL,DOMV,*[Path(m['root']) for m in models]]}
    stop_event=threading.Event();signatures={}
    def stop(*_):
        stop_event.set()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not stop_event.is_set():
        for path in paths:
            if not path.exists():continue
            stat=path.stat();signature=(stat.st_ino,stat.st_mtime_ns,stat.st_size)
            if signatures.get(path)!=signature:
                remove_row([path]);stat=path.stat()
                signatures[path]=(stat.st_ino,stat.st_mtime_ns,stat.st_size)
        stop_event.wait(.1)


if __name__=='__main__':main()
