#!/usr/bin/env python3
"""Run a pinned DOM server with only its post-episode video branch sampled."""
import os
from pathlib import Path
import runpy
import sys
from table1_video_sampling import patch_source


def main():
    index=sys.argv.index('--wrapped-server')
    wrapped=Path(sys.argv[index+1]).resolve()
    del sys.argv[index:index+2]
    assert wrapped.parent == Path(__file__).resolve().parent
    assert wrapped.name in ('collection_matched_server.py','low_clutter_table1_server.py')
    target=(Path(os.environ['DOM_OFFICIAL_SOURCE'])/'simulations/evaluate.py').resolve()
    original=Path.read_text
    def read(self,*args,**kwargs):
        text=original(self,*args,**kwargs)
        if self.name=='evaluate.py' and self.resolve()==target:
            Path.read_text=original
            return patch_source(text)
        return text
    Path.read_text=read
    sys.argv[0]=str(wrapped)
    try:
        offset=int(os.environ.get('DOM_REPEAT_OFFSET','0'))
        if wrapped.name=='low_clutter_table1_server.py' and offset:
            text=wrapped.read_text();needle='        return prompts[index]'
            assert text.count(needle)==1 and 0<offset<20
            text=text.replace(needle,f'        return prompts[index + {offset}]')
            exec(compile(text,str(wrapped),'exec'),{'__name__':'__main__','__file__':str(wrapped)})
        else:runpy.run_path(str(wrapped),run_name='__main__')
    finally:Path.read_text=original


if __name__=='__main__':main()
