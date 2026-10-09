"""USD geometry inspection without importing Isaac or starting simulation."""
import numpy as np


def visual_bounds(stage, root_path):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bounds = []
    # DOM objects may have a points-less Mesh parent containing the actual Mesh.
    # Bounding the parent can return an empty range; inspect drawable leaves.
    for prim in Usd.PrimRange(stage.GetPrimAtPath(str(root_path))):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if points is None or len(points) == 0:
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if not box.IsEmpty():
            bounds.append([np.array(box.GetMin()), np.array(box.GetMax())])
    if not bounds:
        raise ValueError(f"No bounded visible mesh geometry at {root_path}")
    bounds = np.array(bounds)
    return bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
