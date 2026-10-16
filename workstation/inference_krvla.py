#!/usr/bin/env python3
"""Run frozen KRVLA inference with its streaming dummy-warmup compatibility fix."""
import os
from pathlib import Path
import runpy
import sys


def main():
    position=sys.argv.index('--source')
    source=Path(sys.argv[position+1]).resolve()
    del sys.argv[position:position+2]
    os.environ['KRVLA_POLICY_SOURCE']=str(source)
    sys.path.insert(0,str(source))
    from policies.dynamicvla.modeling_dynamicvla import DynamicVLAPolicy
    from krvla_policy_runtime import inference_loop
    DynamicVLAPolicy._inference_loop=staticmethod(inference_loop)
    script=source/'scripts/inference.py'
    sys.argv[0]=str(script)
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':
    main()
