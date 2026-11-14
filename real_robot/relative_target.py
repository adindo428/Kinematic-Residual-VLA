"""Scalar or nine-motor relative target bounds, shared by collection entrypoints."""
import argparse
import json
import math
from numbers import Real

import numpy as np


def normalize_relative_target(value):
    if isinstance(value,str):
        try:value=json.loads(value)
        except (ValueError,TypeError) as exc:
            raise ValueError('Relative target must be a number or JSON list of nine positive finite numbers') from exc
    if isinstance(value,np.ndarray):value=value.tolist()
    def number(item):
        if isinstance(item,(bool,np.bool_)) or not isinstance(item,Real) or not math.isfinite(item) or item<=0:
            raise ValueError('Relative targets must be positive finite numbers')
        return float(item)
    if isinstance(value,(list,tuple)):
        if len(value)!=9:raise ValueError('Expected nine relative targets in right-arm motor order (gripper last)')
        return [number(item) for item in value]
    return number(value)


def parse_relative_target(value):
    try:return normalize_relative_target(value)
    except ValueError as exc:raise argparse.ArgumentTypeError(str(exc)) from exc
