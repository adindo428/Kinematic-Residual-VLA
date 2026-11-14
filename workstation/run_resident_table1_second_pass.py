#!/usr/bin/env python3
"""Use the existing qualified scene-first runner for the remaining ten repeats."""
import argparse
from datetime import datetime,timezone
import time

import resident_table1_second_pass as second
import run_resident_table1_queue as queue


def publish(stage,**kw):
    if stage=='first_pass_queue_complete':stage='complete'
    kw.pop('remaining_repeats_deferred',None)
    state={'stage':stage,'pass_index':2,'repeat_range':[11,20],
           'updated_utc':datetime.now(timezone.utc).isoformat(),**kw}
    second.low.write(second.ROOT/'status.json',state)
    second.low.write(second.first.ROOT/'status.json',state)
    from render_resident_table1 import render
    render()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    # The first pass retains ownership of its records and process until complete.
    while True:
        state=second.low.read(second.first.ROOT/'status.json')
        if state.get('stage')=='first_pass_queue_complete' or state.get('pass_index')==2:break
        if state.get('stage') in ('failed','paused'):raise RuntimeError('First pass requires attention before starting pass two')
        if args.prepare_only:raise RuntimeError('The first pass has not completed')
        time.sleep(5)
    second.prepare()
    if args.prepare_only:return
    queue.schedule=second;queue.ROOT=second.ROOT;queue.prepare=second.prepare;queue.publish=publish
    queue.main()


if __name__=='__main__':main()
