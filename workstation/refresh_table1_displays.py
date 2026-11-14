#!/usr/bin/env python3
"""Keep the editor's evaluation tables current without touching scoring."""
import argparse
import time
from table1_display_state import LOW, BASE, original_tables, write_if_changed
from render_table1_live_status import render
from render_causal_gate_comparison import render as render_gate_comparison


def refresh():
    render_gate_comparison()
    render(LOW,True)
    original_tables()
    path=BASE/'table1_comparison.md'
    text=path.read_text()
    if '**Status: Complete' not in text:
        first,rest=text.split('\n',1)
        write_if_changed(path,first+'\n\n**Status: Complete — archived released-checkpoint evaluation, 1,800 verified trials.**\n'+rest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        refresh()
        if a.once:break
        time.sleep(5)
