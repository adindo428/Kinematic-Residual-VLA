"""Tape prompt association for the wrist camera; EfficientTAM owns the mask.

Use the current fixed-camera 3D target, not a demonstrated image location, to
choose the initial prompt and recover a lost identity. No actuator access.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from .vx300s_kinematics import forward


def read_hint(path, timing, template):
    try:
        hint = json.loads(Path(path).read_text())
        stamp = timing['host_arrival_monotonic_s']
        age = stamp-hint['host_monotonic_s']
        if not hint.get('active') or not -.1 <= age <= .30:
            return None
        q = np.asarray(hint['q_rad'])+np.asarray(hint['joint_velocity_rad_s'])*max(0.,age)
        camera = forward(q)@np.asarray(template['T_gripper_wrist_camera'])
        point = np.asarray(hint['object_position_m'])+np.asarray(hint['object_velocity_m_s'])*(stamp-hint['object_stamp_s'])
        p = camera[:3,:3].T@(point-camera[:3,3])
        if not np.isfinite(p).all() or p[2] <= .025:
            return {'in_view':False}
        intr = timing['intrinsics']
        pixel = np.array([intr['fx']*p[0]/p[2]+intr['ppx'],intr['fy']*p[1]/p[2]+intr['ppy']])
        return {'in_view':bool(0<=pixel[0]<intr['width'] and 0<=pixel[1]<intr['height']),
                'center_px':pixel.tolist(),'depth_m':float(p[2]),'source':'fixed_camera_projection'}
    except (OSError,ValueError,KeyError,TypeError):
        return None


def tape_prompt(image, depth, timing, template, hint=None):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv,np.array([90,100,25]),np.array([140,255,255]))
    # Close small holes without joining separated blue fixtures to the tape.
    blue = cv2.morphologyEx(blue,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,labels,stats,centers = cv2.connectedComponentsWithStats(blue)
    intr = timing['intrinsics']
    reference = template['wrist_reference']
    diameter = max(reference['bbox_xywh'][2:])*reference['depth_m']/np.sqrt(intr['fx']*intr['fy'])
    if hint is not None and not hint.get('in_view'):
        raise ValueError('Fixed-camera tape projects outside the wrist image; waiting for it to enter')
    candidates = []
    for i in range(1,n):
        x,y,w,h,area = stats[i]
        if area < 120:
            continue
        samples = depth[labels==i]
        valid = np.isfinite(samples)&(samples>=.035)&(samples<=.65)
        if valid.sum()<30 or valid.mean()<.35:
            continue
        z = float(np.median(samples[valid]))
        extent = max(w,h)*z/np.sqrt(intr['fx']*intr['fy'])
        # A small blue marker on the gripper is not a tape-sized surface.
        if not .45*diameter <= extent <= 1.9*diameter:
            continue
        score = -np.log(float(area))
        if hint is not None:
            size_px = diameter*np.sqrt(intr['fx']*intr['fy'])/hint['depth_m']
            distance = float(np.linalg.norm(centers[i]-hint['center_px']))
            if distance > .6*size_px+65 or abs(z-hint['depth_m'])>max(.075,.25*hint['depth_m']):
                continue
            score = distance/max(size_px,1.)+abs(z-hint['depth_m'])/.075
        candidates.append((score,i,z))
    if not candidates:
        raise ValueError('No tape-sized blue surface with usable depth near the wrist target; R marks it manually')
    candidates.sort()
    if hint is None and len(candidates)>1:
        if stats[candidates[0][1],4] < 2*stats[candidates[1][1],4]:
            raise ValueError('Multiple tape-sized blue surfaces; R marks the intended tape')
    _,i,z = candidates[0]
    x,y,w,h,_ = stats[i]
    left,top = max(0,int(x)-8),max(0,int(y)-8)
    right,bottom = min(image.shape[1],int(x+w)+8),min(image.shape[0],int(y+h)+8)
    return [left,top,right-left,bottom-top],(labels==i).astype(np.uint8)*255,{
        'source':'fixed_camera_guided_blue_prompt' if hint is not None else 'tape_size_and_depth_prompt',
        'center_px':centers[i].tolist(),'depth_m':z}


def associated_mask(mask, candidate, hint, timing):
    if mask is None:
        return False,'EfficientTAM mask unavailable'
    if candidate is not None:
        overlap = np.count_nonzero((mask>0)&(candidate>0))/max(1,np.count_nonzero(candidate))
        if overlap < .35:
            return False,'EfficientTAM mask does not cover the associated tape surface'
    if hint is not None and hint.get('in_view'):
        ys,xs = np.where(mask>0)
        if len(xs):
            center = np.array([np.median(xs),np.median(ys)])
            intr = timing['intrinsics']
            radius = max(65.,.045*np.sqrt(intr['fx']*intr['fy'])/hint['depth_m'])
            if np.linalg.norm(center-hint['center_px'])>radius:
                return False,'EfficientTAM mask is away from the fixed-camera tape projection'
    return True,None
