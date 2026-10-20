"""Visible top of the segmented object in the provisional robot base frame."""
import numpy as np

def visible_top(mask,depth,deproject,transform,quality,maximum_samples=256):
    if mask is None or quality.get('reason') is not None:return None
    ys,xs=np.nonzero(mask)
    if len(xs)<20:return None
    stride=max(1,int(np.ceil(len(xs)/maximum_samples)));ys,xs=ys[::stride],xs[::stride]
    zs=depth[ys,xs];median=quality['depth_m'];mad=quality['depth_mad_m']
    valid=np.isfinite(zs)&(zs>0)&(np.abs(zs-median)<=max(3*mad,.006))
    if np.count_nonzero(valid)<12:return None
    points=np.array([deproject([float(x),float(y)],float(z)) for x,y,z in zip(xs[valid],ys[valid],zs[valid])])
    t=np.asarray(transform,float);points=points@t[:3,:3].T+t[:3,3]
    points=points[np.isfinite(points).all(axis=1)]
    if len(points)<12:return None
    top=float(np.quantile(points[:,2],.95))+.003
    return top if .015<top<.35 else None
