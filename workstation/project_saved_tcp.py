"""Project saved robot/policy points into native front RGB; no hardware access."""
import argparse,json
from pathlib import Path
import sys
import cv2
import numpy as np
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from real_robot.recover_marker_session import deprojector
from real_robot.efficienttam_tracker import project_native_pixel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    inputs=sorted((a.session/'exchange/inference_inputs').glob('*.npz'))
    path=inputs[0]
    with np.load(path,allow_pickle=False) as z:
        front=z['front'][-1].copy();state=z['state'][-1];env=z['environment'][-1];actions=z['actions'];meta=json.loads(str(z['metadata']))
    stamp=meta['observation_host_stamps_s']['front'][-1]
    records=[json.loads(x) for x in (a.session/'perception/fusion.jsonl').read_text().splitlines()]
    row=min(records,key=lambda r:abs(r['views']['front']['timing']['host_arrival_monotonic_s']-stamp))
    timing=row['views']['front']['timing'];rig=json.loads((a.session/'perception/manifest.json').read_text())['rig'];inverse=np.linalg.inv(rig['T_base_front_candidate'])
    frame=dict(timing=timing,deproject=deprojector(timing));points={}
    for label,point,color in [('FK TCP',state[:3],(0,255,0)),('Object proxy',env[:3],(255,255,0)),('Policy h0',actions[0,:3],(255,0,255)),('Policy h3',actions[3,:3],(0,255,255)),('Policy h19',actions[19,:3],(0,128,255))]:
        camera=inverse[:3,:3]@point+inverse[:3,3];pixel=project_native_pixel(camera,frame)
        points[label]=dict(base_xyz_m=point.tolist(),pixel=None if pixel is None else pixel.tolist())
        if pixel is not None:
            uv=tuple(np.round(pixel).astype(int));cv2.drawMarker(front,uv,color,cv2.MARKER_CROSS,17,2)
            cv2.putText(front,label,(uv[0]+8,uv[1]-6),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1,cv2.LINE_AA)
    footer=np.zeros((82,640,3),np.uint8)
    cv2.putText(footer,'Green: FK TCP | Cyan: tracked surface | Magenta/yellow/orange: policy',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
    cv2.putText(footer,'Projection uses current camera/base/TCP model; visual diagnostic only.',(8,46),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
    a.output.mkdir(parents=True,exist_ok=True);cv2.imwrite(str(a.output/'tcp_projection.png'),np.vstack([front,footer]))
    (a.output/'tcp_projection.json').write_text(json.dumps(dict(hardware_accessed=False,saved_input=str(path),points=points),indent=2)+'\n')
    print(a.output/'tcp_projection.png')


if __name__=='__main__':main()
