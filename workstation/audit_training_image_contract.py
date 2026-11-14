"""Compare frozen training video pixels with the live raw-frame preprocessing."""
import argparse,json
from pathlib import Path
import sys
import numpy as np
from torchcodec.decoders import VideoDecoder
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from real_robot.image_preprocessing import transform,load_spec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    campaign=json.load(open(PROJECT/'scratch/real_robot/family-finetuning-v6/campaign.json'));d=Path(campaign['training_recipe']['local_verified_dataset']);spec=load_spec(d/'meta/image_preprocessing.json')
    train=set(json.load(open(d/'meta/splits.json'))['train']);quality=json.load(open(d/'meta/quality_policy.json'))['usable_frames']
    sources=[json.loads(x) for x in (d/'meta/sources.jsonl').read_text().splitlines()];done=set();result=[]
    for source in sources:
        ep=source['episode_index'];name=source['raw_session']
        if ep not in train or name in done:continue
        valid=set(quality[str(ep)]);indices=[x for x in sorted(valid) if max(0,x-2) in valid]
        if not indices:continue
        q=[json.loads(x) for x in (d/f'quality/episode_{ep:06d}.jsonl').read_text().splitlines()]
        for camera,role in [('opst_cam','front'),('wrist_cam','wrist')]:
            selected=next(((i,Path(name)/q[i]['camera_source_files'][camera]) for i in indices if (Path(name)/q[i]['camera_source_files'][camera]).exists()),None)
            if selected is None:
                result.append(dict(session=name,episode=ep,camera=camera,status='raw_file_unavailable'));continue
            i,raw_path=selected
            with np.load(raw_path,allow_pickle=False) as raw:expected=transform(raw['bgr'],spec,role)[:,:,::-1].copy()
            video=d/f'videos/chunk-{ep//1000:03d}/observation.images.{camera}/episode_{ep:06d}.mp4'
            decoder=VideoDecoder(video.read_bytes(),num_ffmpeg_threads=1);decoded=decoder.get_frame_at(i).data.permute(1,2,0).numpy()
            row=dict(session=name,episode=ep,frame=i,camera=camera,expected_shape=list(expected.shape),decoded_shape=list(decoded.shape))
            if decoded.shape==expected.shape:
                diff=abs(decoded.astype(float)-expected.astype(float));swapped=abs(decoded.astype(float)-expected[:,:,::-1].astype(float))
                row.update(status='compared',rgb_mae_0_255=float(diff.mean()),rgb_p99_abs_error=float(np.percentile(diff,99)),red_blue_swapped_mae=float(swapped.mean()))
            else:row.update(status='shape_mismatch')
            result.append(row);del decoder
        done.add(name)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'image_contract.json').write_text(json.dumps(dict(hardware_accessed=False,comparisons=result,note='AV1 is lossy, so small raw/video differences are expected.'),indent=2)+'\n')
    for row in result:print(Path(row['session']).parent.name,row['camera'],row['status'],row.get('rgb_mae_0_255'),flush=True)


if __name__=='__main__':main()
