"""Decode immutable training videos once to lossless uint8 arrays on local SSD."""
import argparse,json,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from torchcodec.decoders import VideoDecoder


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();c=json.loads((root/'launch_contract.json').read_text());data=Path(c['local_verified_dataset']);cache=Path(c['decoded_video_cache']);episodes=[json.loads(l) for l in (data/'meta/episodes.jsonl').open()]
    jobs=[(e,key) for e in episodes for key in ['observation.images.opst_cam','observation.images.wrist_cam']]
    def decode(job):
        e,key=job;ep=e['episode_index'];out=cache/key/f'{ep:06d}.npy';out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():return out.stat().st_size
        video=data/f'videos/chunk-{ep//1000:03d}/{key}/episode_{ep:06d}.mp4'
        decoder=VideoDecoder(str(video),num_ffmpeg_threads=1,seek_mode='approximate')
        values=decoder.get_frames_at(list(range(e['length']))).data.numpy()
        if values.dtype!=np.uint8 or values.shape!=(e['length'],3,384,384):raise ValueError('Unexpected training video shape')
        temp=out.with_suffix('.writing')
        with temp.open('wb') as f:np.save(f,values)
        temp.replace(out);return out.stat().st_size
    with ThreadPoolExecutor(max_workers=4) as pool:
        sizes=[]
        for i,size in enumerate(pool.map(decode,jobs)):
            sizes.append(size)
            if i%20==19:print('Decoded',i+1,'/',len(jobs),flush=True)
    report=dict(complete=True,files=len(sizes),bytes=sum(sizes),lossless_uint8=True,video_reencoding=False,hardware_accessed=False)
    (cache/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
