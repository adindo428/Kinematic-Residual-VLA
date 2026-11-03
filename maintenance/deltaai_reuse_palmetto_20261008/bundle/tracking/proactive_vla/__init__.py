"""Utilities for residual proactive DynamicVLA experiments."""

from .residual_flow import (
    DEFAULT_COVARIANCE_STD,
    apply_residual_drift,
    centered_action_drift,
    extract_dom_dynamic_fields,
    lambda_schedule,
    project_velocity_to_action_drift,
    remove_scheduled_residual_drift,
    residual_flow_target,
    residualize_actions,
    restore_residual_actions,
    scheduled_residual_drift,
    scheduled_residual_flow_target,
)

__all__ = [
    "DEFAULT_COVARIANCE_STD",
    "apply_residual_drift",
    "centered_action_drift",
    "extract_dom_dynamic_fields",
    "lambda_schedule",
    "project_velocity_to_action_drift",
    "remove_scheduled_residual_drift",
    "residual_flow_target",
    "residualize_actions",
    "restore_residual_actions",
    "scheduled_residual_drift",
    "scheduled_residual_flow_target",
]
