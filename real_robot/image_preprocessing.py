"""Manual front crop and KRVLA-compatible 384px resize/pad from raw RGB-D."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SCHEMA = 'krvla_image_preprocessing_v1'



def order_quad(points):
    """Arrange any four convex corners as TL, TR, BR, BL in image coordinates."""
    p=np.asarray(points,np.float32)
    if p.shape!=(4,2) or not np.isfinite(p).all():
        raise ValueError('Choose four different corners')
    hull=cv2.convexHull(p).reshape(-1,2)
    if len(hull)!=4 or cv2.contourArea(hull)<25:
        raise ValueError('One point is inside the others or corners overlap; press Reset')
    if cv2.contourArea(hull,oriented=True)<0:hull=hull[::-1]
    first=int(np.argmin(hull.sum(axis=1)))
    return np.roll(hull,-first,axis=0).astype(int).tolist()

def validate_spec(spec):
    if (spec.get('schema') != SCHEMA or spec.get('source_size_wh') != [640,480]
            or spec.get('target_size_wh') != [384,384] or spec.get('padding') != 'left_top'
            or spec.get('pad_value') != 0):
        raise ValueError('Expected raw 640x480 to 384x384 KRVLA preprocessing with black left/top padding')
    if spec.get('front_quad_xy') is not None:
        points=np.asarray(spec['front_quad_xy'],dtype=np.float32)
        if points.shape!=(4,2) or not np.isfinite(points).all():
            raise ValueError('Select exactly four finite corners: TL, TR, BR, BL')
        if np.any(points<0) or np.any(points[:,0]>639) or np.any(points[:,1]>479):
            raise ValueError('Perspective corners must lie in the raw image')
        if not cv2.isContourConvex(points) or cv2.contourArea(points,oriented=True)<25:
            raise ValueError('Corners must form a convex clockwise quadrilateral: TL, TR, BR, BL')
        size=spec.get('front_rectified_size_wh')
        if (not isinstance(size,list) or len(size)!=2 or any(type(v) is not int or not 5<=v<=2048 for v in size)):
            raise ValueError('Rectified rectangle must have integer width/height between 5 and 2048')
    else:
        crop = spec.get('front_crop_xywh')
        if (not isinstance(crop,list) or len(crop)!=4 or any(type(v) is not int for v in crop)):
            raise ValueError('Front crop must be integer [x,y,width,height]')
        x,y,w,h=crop
        if x<0 or y<0 or min(w,h)<5 or x+w>640 or y+h>480:
            raise ValueError('Front crop lies outside the raw 640x480 image')
    return spec


def load_spec(path):
    return validate_spec(json.loads(Path(path).read_text()))


def geometry(spec, role):
    validate_spec(spec)
    quad=spec.get('front_quad_xy') if role=='front' else None
    if quad is not None:
        w,h=spec['front_rectified_size_wh'];x=y=0
        source=np.asarray(quad,np.float32)
        target=np.asarray([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float32)
        warp=cv2.getPerspectiveTransform(source,target)
    else:
        x,y,w,h=spec['front_crop_xywh'] if role=='front' else [0,0,640,480]
        warp=np.array([[1,0,-x],[0,1,-y],[0,0,1]],float)
    ratio=max(w/384,h/384)
    rw,rh=int(w/ratio),int(h/ratio)
    px,py=384-rw,384-rh
    sx,sy=rw/w,rh/h
    resized=np.array([[sx,0,sx*.5-.5+px],[0,sy,sy*.5-.5+py],[0,0,1]])
    return dict(crop_xywh=None if quad is not None else [x,y,w,h],
                front_quad_xy=quad,rectified_size_wh=[w,h],
                homography_raw_to_rectified=warp.tolist(),
                resized_size_wh=[rw,rh],padding_ltrb=[px,py,0,0],
                pixel_transform_raw_to_exported=(resized@warp).tolist())


def transform(bgr, spec, role):
    """Match KRVLA resize_with_pad: bilinear align_corners=False, left/top zeros."""
    import torch
    import torch.nn.functional as F
    if bgr.shape != (480,640,3) or bgr.dtype != np.uint8:
        raise ValueError('Preprocessing requires original uint8 640x480 RGB-D frames')
    g=geometry(spec,role);rw,rh=g['resized_size_wh'];px,py,_,_=g['padding_ltrb']
    if g['front_quad_xy'] is not None:
        region=cv2.warpPerspective(bgr,np.asarray(g['homography_raw_to_rectified']),tuple(g['rectified_size_wh']),
                                   flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    else:
        x,y,w,h=g['crop_xywh'];region=bgr[y:y+h,x:x+w]
    image=torch.from_numpy(region.copy()).permute(2,0,1).unsqueeze(0).float()
    with torch.inference_mode():
        image=F.interpolate(image,size=(rh,rw),mode='bilinear',align_corners=False)
        image=F.pad(image,(px,0,py,0),value=0)
    return image.squeeze(0).permute(1,2,0).round().clamp(0,255).to(torch.uint8).numpy()


def choose_crop(sessions, output, perspective=False, rectified_size=None):
    """Inspect sampled raw front frames, select a box, confirm over other frames."""
    import torch
    torch.set_num_threads(4)
    output=Path(output)
    if output.exists(): raise ValueError('Crop settings already exist; choose a new output')
    examples=[]; serials=set()
    for session in map(Path,sessions):
        manifest=json.loads((session/'manifest.json').read_text())
        serials.add(manifest['rig']['front_serial'])
        episodes=[json.loads(p.read_text()) for p in sorted((session/'episodes').glob('*.json'))]
        successful=[(e['start_monotonic_s'],e['end_monotonic_s']) for e in episodes if e['outcome']=='success']
        rows=[json.loads(line) for line in (session/'observations.jsonl').open()]
        for start,end in successful:
            frames=[row for row in rows if start<=row['host_recorded_monotonic_s']<=end
                    and (row.get('rgbd') or {}).get('files',{}).get('front')]
            if not frames: continue
            for index in sorted({int((len(frames)-1)*f) for f in (0,.25,.5,.75,1)}):
                path=session/frames[index]['rgbd']['files']['front']
                if not examples or examples[-1]!=path: examples.append(path)
    if len(serials)!=1 or not examples: raise ValueError('Need raw front frames from one fixed camera')
    spec=dict(schema=SCHEMA,source_size_wh=[640,480],target_size_wh=[384,384],padding='left_top',pad_value=0,
              front_serial=next(iter(serials)),front_crop_xywh=None,
              source_sessions=[str(Path(p).resolve()) for p in sessions],
              interpolation='bilinear align_corners=False; uint8 rounding before AV1 encoding')
    points=[]
    error_text=['']
    requested_key=[None]
    preview_window='Straightened output - 384x384 - click SAVE or press ENTER'
    window=('Front trapezoid - click four corners in ANY order' if perspective else
            'Front crop - N/P browse | C draw crop | ENTER save | Q cancel')
    cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
    def reset():
        points.clear();spec.pop('front_quad_xy',None);error_text[0]=''

    def save_draft():
        output.parent.mkdir(parents=True,exist_ok=True)
        output.with_suffix('.draft.json').write_text(json.dumps({'points':points,'spec':spec},indent=2)+'\n')

    def click(event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        if y>=480:
            if 450<=x<635 and 490<=y<535:requested_key[0]=13
            elif 245<=x<430 and 490<=y<535:reset();save_draft()
            return
        if len(points)>=4:return
        points.append([int(x),int(y)])
        if len(points)==4:
            try:
                ordered=order_quad(points)
                p=np.asarray(ordered,float)
                derived=[max(5,int(round((np.linalg.norm(p[1]-p[0])+np.linalg.norm(p[2]-p[3]))/2))),
                         max(5,int(round((np.linalg.norm(p[3]-p[0])+np.linalg.norm(p[2]-p[1]))/2)))]
                spec['front_quad_xy']=ordered
                spec['front_rectified_size_wh']=list(rectified_size) if rectified_size is not None else derived
                spec['rectified_aspect_note']=('user-specified rectangle' if rectified_size is not None else
                    'Approximate aspect inferred from mean opposite image-edge lengths; physical aspect is not measured')
                validate_spec(spec)
                points[:]=ordered
                error_text[0]=''
            except ValueError as exc:
                error_text[0]=str(exc)
                print(error_text[0],flush=True)
                spec.pop('front_quad_xy',None)
        save_draft()

    def preview_click(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and y>=384:
            requested_key[0]=13

    if perspective:cv2.setMouseCallback(window,click)
    index=len(examples)//2
    print('Click the four trapezoid corners in ANY order. Click SAVE or press ENTER in either image window. Reset/R clears corners; N/P browses; Q cancels.' if perspective else 'Crop selector: N/P browse raw frames; C draws crop; ENTER confirms; Q cancels.',flush=True)
    try:
        while True:
            with np.load(examples[index],allow_pickle=False) as raw: image=raw['bgr'].copy()
            if image.shape!=(480,640,3): raise ValueError('Raw frame is not 640x480')
            view=image.copy()
            if perspective:
                for i,point in enumerate(points):
                    cv2.circle(view,tuple(point),5,(0,255,0),-1)
                    cv2.putText(view,str(i+1),tuple(point),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
                if len(points)>1:cv2.polylines(view,[np.asarray(points,np.int32)],len(points)==4,(0,255,0),2)
            elif spec['front_crop_xywh']:
                x,y,w,h=spec['front_crop_xywh'];cv2.rectangle(view,(x,y),(x+w,y+h),(0,255,0),2)
            if spec.get('front_quad_xy') is not None or spec['front_crop_xywh']:
                preview=transform(image,spec,'front')
                preview_footer=np.full((58,384,3),(35,100,35),np.uint8)
                cv2.putText(preview_footer,'CLICK HERE TO SAVE',(30,36),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
                cv2.imshow(preview_window,np.vstack([preview,preview_footer]))
                cv2.setMouseCallback(preview_window,preview_click)
            footer=np.full((120,640,3),24,np.uint8)
            ready=spec.get('front_quad_xy') is not None or spec['front_crop_xywh'] is not None
            status=error_text[0] or ('READY: inspect preview, then SAVE' if ready else f'Click {4-len(points)} more corner(s), any order')
            cv2.putText(footer,status[:83],(8,78),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,255) if error_text[0] else (0,255,255),1)
            cv2.putText(footer,f'Example {index+1}/{len(examples)} | N/P browse | Q cancel',(8,103),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,230,230),1)
            cv2.rectangle(footer,(245,10),(430,55),(75,75,75),-1)
            cv2.putText(footer,'RESET',(293,41),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            cv2.rectangle(footer,(450,10),(635,55),(25,135,25) if ready else (65,65,65),-1)
            cv2.putText(footer,'SAVE',(505,41),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            cv2.imshow(window,np.vstack([view,footer]))
            key=cv2.waitKey(30)&255
            if requested_key[0] is not None:
                key=requested_key[0];requested_key[0]=None
            if key in (ord('q'),27): return None
            if key==ord('n'): index=(index+1)%len(examples)
            elif key==ord('p'): index=(index-1)%len(examples)
            elif key==ord('r') and perspective:
                reset();save_draft()
            elif key==ord('c') and not perspective:
                box=list(map(int,cv2.selectROI('Draw crop on RAW front image - ENTER confirms box',image,showCrosshair=True,fromCenter=False)))
                cv2.destroyWindow('Draw crop on RAW front image - ENTER confirms box')
                if min(box[2:])>=5:spec['front_crop_xywh']=box
            elif key in (10,13) and (spec.get('front_quad_xy') is not None or spec['front_crop_xywh']):
                validate_spec(spec)
                spec['selection_example']=str(examples[index].resolve())
                spec['selection_example_sha256']=hashlib.sha256(examples[index].read_bytes()).hexdigest()
                output.parent.mkdir(parents=True,exist_ok=True)
                with output.open('x') as f:f.write(json.dumps(spec,indent=2)+'\n')
                cv2.imwrite(str(output.with_suffix('.preview.png')),transform(image,spec,'front'))
                print('Saved front region:',spec.get('front_quad_xy') or spec['front_crop_xywh'],'to',output,flush=True)
                return spec
            elif key in (10,13):
                error_text[0]='Select four valid corners before saving; click RESET to try again'
    finally:cv2.destroyAllWindows()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--perspective',action='store_true',help='Select four corners to rectify a trapezoid')
    p.add_argument('--rectified-size',type=int,nargs=2,metavar=('WIDTH','HEIGHT'),help='Optional rectangle aspect/size before resize and padding')
    p.add_argument('--sessions',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    return 0 if choose_crop(a.sessions,a.output,a.perspective,a.rectified_size) is not None else 1


if __name__=='__main__':raise SystemExit(main())
