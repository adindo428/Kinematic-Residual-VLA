"""Front RGB feed for policies with no object-state inputs; no robot-device IO."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .camera import RGBDCamera
from .multiview_track import check_camera_profile


def publish_frame(root,bgr,timing,rig_sha,now):
    stamp=timing.get('host_arrival_monotonic_s')
    if bgr.shape!=(480,640,3) or bgr.dtype!=np.uint8:raise ValueError('Expected native 640x480 uint8 front RGB')
    if stamp is None or not np.isfinite(stamp) or not 0<=now-stamp<=.2:raise ValueError('Front camera frame stale')
    record=dict(schema='front_side_observation_v1',running=True,stale=False,rig_sha256=rig_sha,
        output_frame='right_base_estimate',timing=dict(oldest_host_arrival_monotonic_s=stamp),
        object_inputs_used_by_policy=False,track=None,velocity_report=dict(status='not_used_by_policy'),
        host_published_monotonic_s=now)
    with (root/'policy_front.tmp').open('wb') as stream:np.savez(stream,bgr=bgr,metadata=json.dumps(record))
    (root/'policy_front.tmp').replace(root/'policy_front.npz')
    temp=root/'observation.tmp.json';temp.write_text(json.dumps(record)+'\n');temp.replace(root/'observation.json')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rig',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=720);p.add_argument('--camera-fps',type=int,choices=[30],default=30)
    p.add_argument('--front-color-format',choices=['bgr8','yuyv'],default='bgr8');p.add_argument('--preview',action='store_true')
    a=p.parse_args();rig=json.loads(a.rig.read_text());digest=hashlib.sha256(a.rig.read_bytes()).hexdigest()
    if not np.isfinite(a.seconds) or not 0<a.seconds<=1800:p.error('seconds must be positive and <=1800')
    a.output.mkdir(parents=True,exist_ok=True);camera=None;count=0;error=None;started=time.monotonic();previous=None;domain=None
    (a.output/'manifest.json').write_text(json.dumps(dict(rig=rig,rig_sha256=digest,object_tracking_enabled=False,
        object_inputs_used_by_policy=False,depth_used_for_object_inputs=False,front_camera_fps=a.camera_fps,
        front_color_format=a.front_color_format,hardware_motion_accessed=False),indent=2)+'\n')
    try:
        camera=RGBDCamera(rig['front_serial'],fps=a.camera_fps,color_format=a.front_color_format)
        while time.monotonic()-started<a.seconds and not (a.output/'STOP').exists():
            bgr,_,timing,_=camera.read();check_camera_profile(rig,'front',timing)
            stamp=timing['color_timestamp_s'];new_domain=timing['timestamp_domain']
            if (previous is not None and stamp<=previous) or (domain is not None and domain!=new_domain):
                raise RuntimeError('Front RGB clock repeated or changed')
            previous=stamp;domain=new_domain
            publish_frame(a.output,bgr,timing,digest,time.monotonic());count+=1
            if a.preview:
                view=bgr.copy();cv2.putText(view,'RGB + robot state | no object tracker',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.53,(0,255,0),1)
                cv2.imshow('Policy front RGB',view)
                if cv2.waitKey(1)&255 in [ord('q'),27]:break
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';raise
    finally:
        if camera is not None:camera.close()
        if a.preview:cv2.destroyAllWindows()
        temp=a.output/'observation.tmp.json';temp.write_text(json.dumps(dict(schema='front_side_observation_v1',running=False,
            stale=True,rig_sha256=digest,object_inputs_used_by_policy=False,error=error)));temp.replace(a.output/'observation.json')
        (a.output/'summary.json').write_text(json.dumps(dict(frames=count,error=error,object_tracking_enabled=False,motion_sent=False),indent=2)+'\n')


if __name__=='__main__':main()
