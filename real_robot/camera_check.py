"""Read-only front-camera board/depth check. Never opens robot serial ports."""
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .base_calibration import board_reference,board_depth_agreement,write
from .calibration_board import Chessboard,ChessboardDetector,contact_layout
from .multiview import estimate_board


def intrinsics_dict(intr):
    return {**{key:getattr(intr,key) for key in ('width','height','fx','fy','ppx','ppy','coeffs')},
            'distortion_model':str(intr.model)}


def compare_board(frame,board,detector,native_color_intrinsics,rs):
    """Compare the existing aligned-depth ray path with native color rays."""
    required=contact_layout(board.squares)['fit_ids']+contact_layout(board.squares)['check_ids']
    result={}
    for label,intr in [('aligned_depth_rays',None),('native_color_rays',native_color_intrinsics)]:
        observation=dict(frame)
        if intr is not None:
            observation['deproject']=lambda pixel,z:rs.rs2_deproject_pixel_to_point(intr,pixel,z)
            metadata=intrinsics_dict(intr);model=metadata.pop('distortion_model')
            observation['timing']={**frame['timing'],'intrinsics':metadata,'distortion_model':model}
        detection=estimate_board(observation,board,detector)
        if not detection.get('valid'):raise ValueError(f'Need complete board visible: {detection.get("corner_count",0)} corners, {detection.get("reason")}')
        reference=board_reference(observation,detection,board,required)
        result[label]=dict(agreement=board_depth_agreement(reference),reference=reference)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True,help='Completed chessboard contact session supplying camera/board settings')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=12)
    args=parser.parse_args(argv)
    if not 3<=args.frames<=30:parser.error('frames must be 3..30')
    manifest=json.loads((args.session/'manifest.json').read_text())
    if manifest.get('board_type')!='chessboard':parser.error('This diagnostic requires a plain chessboard session')
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    from .camera import RGBDCamera
    from .recording_lease import LeaseOwner
    cv2.setNumThreads(1)
    board=Chessboard(manifest['board_squares'],manifest['square_mm']/1000)
    detector=ChessboardDetector(board)
    camera=lease=None;records=[];error=None
    report=dict(schema='krvla_camera_board_check_v1',source_session=str(args.session.resolve()),
                square_mm=manifest['square_mm'],board_squares=manifest['board_squares'],
                robot_ports_opened=False,calibration_parameters_changed=False)
    try:
        lease=LeaseOwner(Path(__file__).resolve().parents[1]/'scratch/real_robot/active-collection.json')
        camera=RGBDCamera(manifest['rig']['front_serial'],fps=15)
        native_color=camera.profile.get_stream(camera.rs.stream.color).as_video_stream_profile().get_intrinsics()
        native_depth=camera.profile.get_stream(camera.rs.stream.depth).as_video_stream_profile().get_intrinsics()
        report.update(native_color_intrinsics=intrinsics_dict(native_color),native_depth_intrinsics=intrinsics_dict(native_depth))
        device=camera.profile.get_device()
        report['device']={key:device.get_info(getattr(camera.rs.camera_info,key)) for key in
                          ('name','serial_number','firmware_version') if device.supports(getattr(camera.rs.camera_info,key))}
        print('Camera-only check. Keep the full printed board visible and hands/gripper clear; no robot control is opened.',flush=True)
        deadline=time.monotonic()+30
        while len(records)<args.frames and time.monotonic()<deadline:
            bgr,depth,timing,deproject=camera.read()
            if time.monotonic()-timing['host_arrival_monotonic_s']>.2:continue
            frame=dict(bgr=bgr,depth=depth,timing=timing,deproject=deproject)
            try:comparison=compare_board(frame,board,detector,native_color,camera.rs)
            except ValueError as exc:
                error=str(exc);continue
            error=None
            records.append(dict(timing=timing,comparison=comparison))
            if len(records)==1:
                cv2.imwrite(str(root/'board.jpg'),bgr)
                np.savez(root/'board.npz',bgr=bgr,depth_m=depth)
            print(f'{len(records)}/{args.frames} frames: aligned-ray gap '
                  f'{comparison["aligned_depth_rays"]["agreement"]["max_gap_m"]*1000:.1f} mm; '
                  f'native-color-ray gap {comparison["native_color_rays"]["agreement"]["max_gap_m"]*1000:.1f} mm',flush=True)
        if len(records)<args.frames:
            raise RuntimeError(f'Only {len(records)}/{args.frames} complete board frames; {error or "camera timeout"}')
        report['summary']={label:dict(median_max_gap_mm=float(np.median([
            r['comparison'][label]['agreement']['max_gap_m']*1000 for r in records])))
                          for label in ('aligned_depth_rays','native_color_rays')}
        print(json.dumps(report['summary'],indent=2))
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report.update(frames=records,error=error)
        write(root/'camera_check.json',report)
        if camera is not None:camera.close()
        if lease is not None:lease.close()
        print(f'Camera diagnostic saved: {root/"camera_check.json"}',flush=True)


if __name__=='__main__':main()
