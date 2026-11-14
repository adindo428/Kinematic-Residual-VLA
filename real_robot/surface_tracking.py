"""RGB ball tracking constrained to measured supporting planes; no depth filling."""
import argparse
from collections import deque
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .geometry import validate_transform
from .image_preprocessing import order_quad

SCHEMA='ball_support_surfaces_v1'


def plane_from_poses(rows,board_thickness_m=0.):
    if len(rows)<20:raise ValueError('Need 20 valid board observations while it is stationary')
    poses=[validate_transform(r['T_camera_board']) for r in rows]
    normals=[];offsets=[]
    for pose in poses:
        n=pose[:3,2].copy();p=pose[:3,3]
        if n@p>0:n=-n
        normals.append(n);offsets.append(float(n@p)-board_thickness_m)
    normals=np.asarray(normals);normal=normals.mean(axis=0);normal/=np.linalg.norm(normal)
    angles=np.degrees(np.arccos(np.clip(normals@normal,-1,1)))
    if angles.max()>1. or np.ptp(offsets)>.008:
        raise ValueError('Board plane is unstable; hold it flat and still, then retry S')
    return dict(normal_camera=normal.tolist(),offset_camera_m=float(np.median(offsets)),
                max_normal_deviation_deg=float(angles.max()),offset_range_m=float(np.ptp(offsets)),
                observations=len(rows),max_reprojection_error_px=max(r['ray_rmse_equivalent_px'] for r in rows))


class SurfaceModel:
    def __init__(self,spec):
        if spec.get('schema')!=SCHEMA or not 0<spec.get('ball_diameter_m',0)<=.2:
            raise ValueError('Invalid surface calibration schema or ball diameter')
        self.spec=spec;self.radius=spec['ball_diameter_m']/2;self.surfaces=[]
        if {s['name'] for s in spec['surfaces']}!={'table','ramp'}:raise ValueError('Need table and ramp planes')
        for surface in spec['surfaces']:
            n=np.asarray(surface['normal_camera'],float);d=float(surface['offset_camera_m'])
            if n.shape!=(3,) or not np.isfinite(n).all() or abs(np.linalg.norm(n)-1)>1e-5 or not np.isfinite(d) or d>=-self.radius:
                raise ValueError('Invalid camera-facing supporting plane')
            corners=order_quad(surface['region_xy'])
            if any(x<0 or y<0 or x>=640 or y>=480 for x,y in corners):raise ValueError('Surface region outside image')
            region=np.zeros((480,640),np.uint8);cv2.fillConvexPoly(region,np.asarray(corners,np.int32),1)
            inside=cv2.erode(region,np.ones((3,3),np.uint8))>0
            self.surfaces.append((surface,n,d,region>0,inside))
        self.surfaces.sort(key=lambda x:x[0]['name']!='ramp')

    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))

    def check_camera(self,timing):
        if timing['camera_serial']!=self.spec['camera_serial']:raise ValueError('Surface calibration belongs to another camera')
        original=self.spec['camera_timing']['intrinsics'];current=timing['intrinsics']
        for key in ['width','height','fx','fy','ppx','ppy','coeffs']:
            if not np.allclose(original[key],current[key],rtol=1e-5,atol=1e-5):raise ValueError('Camera intrinsics changed after surface calibration')
        if timing.get('distortion_model')!=self.spec['camera_timing'].get('distortion_model'):
            raise ValueError('Camera distortion model changed after surface calibration')

    def measure(self,mask,deproject):
        base=dict(measurement_method='rgb_support_plane',depth_used=False,ball_radius_m=self.radius,
                  position_semantics='mask-centroid ray intersected with radius-offset supporting plane; assumes rolling contact')
        if mask is None:return None,{**base,'reason':'object_mask_unavailable'}
        if mask.shape!=(480,640):raise ValueError('Surface calibration requires native 640x480 images')
        ys,xs=np.nonzero(mask)
        if len(xs)<20:return None,{**base,'reason':'mask_too_small_for_surface_tracking'}
        if min(xs)==0 or min(ys)==0 or max(xs)==639 or max(ys)==479:
            return None,{**base,'reason':'object_mask_clipped'}
        chosen=None
        for surface,n,d,region,inside in self.surfaces:
            # Ramp has priority over the projected table. Reject straddling boundaries.
            occupied=region[ys,xs]
            if occupied.any():
                if not inside[ys,xs].all():return None,{**base,'reason':'surface_boundary_or_partial_region'}
                chosen=(surface,n,d);break
        if chosen is None:return None,{**base,'reason':'outside_calibrated_surface_regions'}
        surface,n,d=chosen;pixel=[float(np.mean(xs)),float(np.mean(ys))]
        ray=np.asarray(deproject(pixel,1.),float)
        if ray.shape!=(3,) or not np.isfinite(ray).all() or abs(n@ray)/np.linalg.norm(ray)<.05:
            return None,{**base,'reason':'grazing_or_invalid_ray'}
        scale=(d+self.radius)/(n@ray);point=scale*ray
        if scale<=0 or not .05<point[2]<5:return None,{**base,'reason':'surface_intersection_out_of_range'}
        return point,{**base,'reason':None,'pixel':pixel,'surface':surface['name'],
                      'support_assumption':'ball in contact with this plane; no stereo-depth verification'}


def calibrate(args):
    from .collect import PROJECT,resolve
    from .multiview import CameraPair,FRONT_SERIAL,charuco_detector,estimate_board
    from .multiview_track import check_camera_profile,load_rig
    from .recording_lease import LeaseOwner
    if not 0<args.ball_diameter_mm<=200 or args.square_mm<=0 or not 0<args.marker_mm<args.square_mm or args.board_thickness_mm<0:
        raise ValueError('Invalid ball or board dimensions')
    output=resolve(args.output)
    if output.exists():raise ValueError('Calibration exists; use a new output file')
    rig_path=resolve(args.rig);rig,_,_,digest=load_rig(rig_path)
    lock=LeaseOwner(PROJECT/'scratch/real_robot/active-collection.json')
    camera=CameraPair(front_only=True,fps=30,reconnect_attempts=3).start()
    board,detector=charuco_detector(args.square_mm/1000,args.marker_mm/1000)
    window='Surface calibration - S capture board | click 4 region corners | ENTER next/save | Q cancel'
    samples=deque(maxlen=20);points=[];captured=None;surfaces=[];stage=0;message='Board flat on TABLE: wait for 20 valid frames, then S'
    requested=[None];last_timing=None
    cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
    def click(event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        if y<480:
            if len(points)<4:points.append([x,y])
        elif 490<=y<=535:
            requested[0]=ord('s') if x<210 else ord('r') if x<420 else 13
    cv2.setMouseCallback(window,click)
    try:
        while True:
            pair=camera.poll_latest()
            if pair is None:
                if cv2.waitKey(10)&255 in (27,ord('q')):return None
                continue
            frame=pair[0];check_camera_profile(rig,'front',frame['timing']);last_timing=frame['timing']
            found=estimate_board(frame,board,detector)
            if found['valid'] and found['ray_rmse_equivalent_px']<1.:
                samples.append(found)
            else:samples.clear()
            view=frame['bgr'].copy()
            if found.get('pixels'):
                for x,y in found['pixels']:cv2.circle(view,(int(x),int(y)),2,(0,255,255),-1)
            if points:cv2.polylines(view,[np.asarray(points,np.int32)],len(points)==4,(0,255,0),2)
            cv2.putText(view,['TABLE','RAMP'][stage]+f' | valid board frames {len(samples)}/20 | plane captured={captured is not None}',(8,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            footer=np.full((115,640,3),24,np.uint8)
            for x,label in [(5,'S: CAPTURE BOARD'),(215,'R: RESET REGION'),(425,'ENTER: NEXT/SAVE')]:
                cv2.rectangle(footer,(x,10),(x+205,55),(40,95,40),-1)
                cv2.putText(footer,label,(x+5,37),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            cv2.putText(footer,message[:98],(6,79),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1)
            cv2.putText(footer,'Corners in any order; keep the camera and ramp fixed after saving',(6,102),cv2.FONT_HERSHEY_SIMPLEX,.43,(230,230,230),1)
            cv2.imshow(window,np.vstack([view,footer]));key=cv2.waitKey(1)&255
            if requested[0] is not None:key=requested[0];requested[0]=None
            if key in (27,ord('q')):return None
            try:
                if key==ord('s'):
                    captured=plane_from_poses(list(samples),args.board_thickness_mm/1000)
                    captured['board_observations']=list(samples)
                    message='Plane captured. Click four corners of this surface region, then ENTER.'
                    print(message,flush=True)
                elif key==ord('r'):points.clear();message='Click four surface-region corners in any order'
                elif key in (10,13):
                    if captured is None:raise ValueError('Capture the board plane with S first')
                    corners=order_quad(points)
                    surfaces.append(dict(name=['table','ramp'][stage],region_xy=corners,**captured))
                    image_path=output.with_name(output.stem+'-'+surfaces[-1]['name']+'.png')
                    image_path.parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(image_path),view)
                    if stage==0:
                        stage=1;captured=None;points.clear();samples.clear()
                        message='Put board FLAT ON RAMP; wait for 20 valid frames, then S'
                        print(message,flush=True)
                    else:
                        spec=dict(schema=SCHEMA,created_utc=datetime.now(timezone.utc).isoformat(),
                            camera_serial=FRONT_SERIAL,camera_timing=last_timing,rig_sha256=digest,
                            ball_diameter_m=args.ball_diameter_mm/1000,board=dict(squares=[6,6],square_m=args.square_mm/1000,
                            marker_m=args.marker_mm/1000,thickness_m=args.board_thickness_mm/1000,dictionary='DICT_4X4_50'),
                            surfaces=surfaces,depth_used=False,
                            assumptions='Fixed camera and supporting surfaces; spherical ball in rolling contact; boundary-crossing masks rejected')
                        SurfaceModel(spec)
                        with output.open('x') as f:f.write(json.dumps(spec,indent=2)+'\n')
                        print('Surface calibration saved:',output,flush=True)
                        return spec
            except ValueError as exc:message=str(exc);print(message,flush=True)
    finally:
        camera.close();cv2.destroyAllWindows();lock.close()


def main(argv=None):
    from .collect import PROJECT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ball-diameter-mm',type=float,required=True)
    p.add_argument('--square-mm',type=float,required=True)
    p.add_argument('--marker-mm',type=float,default=37.5)
    p.add_argument('--board-thickness-mm',type=float,default=0.,help='Thickness from supporting surface to printed board face; default assumes thin paper')
    p.add_argument('--rig',default=str(PROJECT/'scratch/real_robot/front-side-calibration-20261020-01/rig.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    return 0 if calibrate(a) is not None else 1


if __name__=='__main__':raise SystemExit(main())
