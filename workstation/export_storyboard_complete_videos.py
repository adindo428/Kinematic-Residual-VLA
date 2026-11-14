"""Export complete, original-timing clips for a scientific manipulation storyboard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[1]


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)],text=True))['streams'][0]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();cfg=json.loads((root/'storyboard_selection.json').read_text());exports=[];policy=[]
    for row in cfg['sources']:
        source=REPO/row['path'];kind='training' if row['kind']=='training_demonstration' else 'dynamicvla_kr'
        dest=root/'videos'/kind/(row['object'].lower().replace(' ','_')+'_complete.mp4');dest.parent.mkdir(parents=True,exist_ok=True)
        original=probe(source)
        if kind=='training':
            if not dest.exists():os.link(source,dest)
            assert sha(dest)==sha(source)
        else:
            x0,y0,x1,y1=row['camera_crop']
            subprocess.run(['ffmpeg','-v','error','-y','-threads','2','-i',str(source),
                '-vf',f'crop={x1-x0}:{y1-y0}:{x0}:{y0}','-an','-c:v','libx264','-preset','fast','-crf','18',
                '-pix_fmt','yuv420p','-vsync','0','-threads','2','-movflags','+faststart',
                '-metadata','title=DynamicVLA + KR: '+row['object']+' to '+row['container'],str(dest)],check=True)
            policy.append(dest)
        result=probe(dest)
        assert result['nb_frames']==original['nb_frames'] and result['avg_frame_rate']==original['avg_frame_rate']
        assert abs(float(result['duration'])-float(original['duration']))<.001
        exports.append({'task':row['object']+' to '+row['container'],'kind':kind,'path':str(dest.relative_to(root)),
            'source':row['path'],'frames':int(result['nb_frames']),'fps':result['avg_frame_rate'],
            'duration_s':float(result['duration']),'complete_untrimmed_episode':True,'sha256':sha(dest),
            'processing':'Exact original file' if kind=='training' else 'Original opposite-view tile; no trimming, frame dropping, interpolation or playback-speed change'})
    listing=root/'videos/concatenate.txt'
    listing.write_text(''.join("file '"+str(path).replace("'","'\\''")+"'\n" for path in policy))
    showreel=root/'dynamicvla_kr_complete_tasks.mp4'
    subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(showreel)],check=True)
    reel=probe(showreel)
    assert int(reel['nb_frames'])==sum(v['frames'] for v in exports if v['kind']=='dynamicvla_kr')
    receipt={'exports':exports,'showreel':{'path':showreel.name,'task_order':[r['object'] for r in cfg['sources'][3:]],
        'frames':int(reel['nb_frames']),'duration_s':float(reel['duration']),'sha256':sha(showreel)},
        'note':'Selected successful complete episodes for qualitative presentation. Neither all-method smoothness nor success-rate performance is inferred from this selection.'}
    (root/'complete_video_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'full_clips':len(exports),'showreel':str(showreel),'duration_s':float(reel['duration'])},indent=2))


if __name__=='__main__':main()
