"""Run in an environment exposing pxr; no GPU or SimulationApp is required."""
import numpy as np
import pytest

from dom_tracking.usd_geometry import visual_bounds


def test_nested_empty_mesh_parent_does_not_hide_geometry():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Object")
    UsdGeom.Mesh.Define(stage, "/Object/geometry")
    mesh = UsdGeom.Mesh.Define(stage, "/Object/geometry/mesh")
    mesh.GetPointsAttr().Set([Gf.Vec3f(-.04, -.03, -.02), Gf.Vec3f(.04, .03, .02)])
    mesh.GetExtentAttr().Set([Gf.Vec3f(-.04, -.03, -.02), Gf.Vec3f(.04, .03, .02)])
    lo, hi = visual_bounds(stage, "/Object")
    np.testing.assert_allclose(lo, [-.04, -.03, -.02])
    np.testing.assert_allclose(hi, [.04, .03, .02])
