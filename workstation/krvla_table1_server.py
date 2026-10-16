#!/usr/bin/env python3
"""Official paper-4.5 simulator with current robot-frame object state attached."""
import os
from pathlib import Path
import sys


def main():
    scripts = Path(__file__).resolve().parents[1]/'evaluation/official_dom/scripts'
    sys.path.insert(0,str(scripts))
    from server_observer import instrument_and_run
    source = Path(os.environ['DOM_OFFICIAL_SOURCE'])/'simulations/evaluate.py'
    text = source.read_text()
    images = '                **{"observation.images.%s" % k: v["rgb"] for k, v in cam_view.items()},'
    entry = '    main(app_launcher.app, args)'
    if text.count(images)!=1 or text.count(entry)!=1:
        raise ValueError('Pinned simulator observation entry points changed')
    # get_curr_state already computes these quantities for the original loop.
    # Extend only the observation payload, keeping physics/actions/scoring intact.
    extra = ('                "observation.environment_state": {"object": {\n'
             '                    k: v.cpu().numpy() for k, v in curr_state["object"].items()}},\n')
    text = text.replace(images,extra+images)
    text = text.replace(entry,'    _instrument_and_run(main, app_launcher.app, args, globals())')
    sys.path.insert(0,str(source.parent.parent))
    namespace = {'__name__':'__main__','__file__':str(source),'_instrument_and_run':instrument_and_run}
    exec(compile(text,str(source),'exec'),namespace)


if __name__=='__main__':
    main()
