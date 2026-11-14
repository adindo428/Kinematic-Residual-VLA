"""Shared telemetry support for grasp annotations and operator confirmation.

These bounds alone do not confirm attachment; wrist visual evidence is required.
An object can keep the jaws well above their empty, fully-closed aperture.
"""
import math

MAX_HOLDING_APERTURE_FRACTION=.675
MAX_HOLDING_GRIPPER_NORMALIZED=.35
MINIMUM_HOLDING_LIFT_M=.020


def supports_visible_holding(measured_aperture,commanded_aperture,lift_m):
    return (all(math.isfinite(x) for x in [measured_aperture,commanded_aperture,lift_m])
        and measured_aperture<=MAX_HOLDING_APERTURE_FRACTION
        and commanded_aperture<=MAX_HOLDING_APERTURE_FRACTION
        and lift_m>=MINIMUM_HOLDING_LIFT_M)
