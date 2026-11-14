"""Transport arai commands and rsync through the existing CPU allocation."""
import subprocess
import sys
from pathlib import Path
import palmetto_allocated_transport as transport

transport.ALLOCATION = (Path(__file__).resolve().parents[1] /
    'scratch/dom45/training/palmetto-arai-control-20261112/cpu_allocation.json')
run = transport.run
allocated_command = transport.allocated_command

if __name__ == '__main__':
    argv = sys.argv[1:]
    if argv[:1] == ['-l']:
        assert argv[1] == 'arai'
        argv = argv[2:]
    target, *args = argv
    assert target in {'arai@slogin.palmetto.clemson.edu', 'slogin.palmetto.clemson.edu'}
    raise SystemExit(subprocess.call(allocated_command(args)))
