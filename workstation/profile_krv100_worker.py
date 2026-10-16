#!/usr/bin/env python3
"""Run the frozen profiling worker with unused metadata removed before spawn."""
from pathlib import Path
import runpy
import sys

from training_metadata import install_metadata_compaction


def main():
    position = sys.argv.index('--source')
    source = Path(sys.argv[position+1]).resolve()
    del sys.argv[position:position+2]
    sys.path.insert(0,str(source))
    import core.dom_distributed_profiling as profiling
    install_metadata_compaction(profiling)
    script = source/'scripts/profile_dom_distributed.py'
    sys.argv[0] = str(script)
    runpy.run_path(str(script),run_name='__main__')


if __name__ == '__main__':
    main()
