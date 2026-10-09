"""Stable tracking methods; DOM+Velocity uses neural conditioning without analytic drift."""
VARIANT_LABELS = {'baseline': 'DOM-VLA', 'velocity': 'DOM-VLA+Velocity',
                  'residual': 'KR-VLA-Field', 'coordinate': 'KR-VLA-Coordinate',
                  'coordinate_velocity': 'KR-VLA-Coordinate+Velocity'}

def checkpoints_from_args(args):
    result = {key: getattr(args, key) for key in VARIANT_LABELS if getattr(args, key, None)}
    if not result: raise ValueError('At least one explicit checkpoint is required')
    return result

def validate_variant(config, variant, require_neural_velocity=False):
    if variant not in VARIANT_LABELS: raise ValueError(f'Unknown variant {variant}')
    if bool(config.enable_residual_flow) != (variant in ('residual','coordinate','coordinate_velocity')):
        raise ValueError(f'{variant}: checkpoint residual enable flag differs')
    neural_velocity = bool(getattr(config,'condition_on_object_velocity',False))
    # Keep old Coordinate checkpoints loadable as the no-velocity ablation.
    # New Coordinate training explicitly requires both neural and analytic velocity.
    if variant != 'coordinate' and neural_velocity != (variant in ('velocity','coordinate_velocity')):
        raise ValueError(f'{variant}: checkpoint velocity conditioning flag differs')
    if require_neural_velocity and not neural_velocity:
        raise ValueError(f'{variant}: explicit neural velocity conditioning is required')
    expected = 'coordinate' if variant in ('coordinate','coordinate_velocity') else 'field'
    actual = getattr(config, 'residual_flow_formulation', 'field')
    if variant in ('residual','coordinate','coordinate_velocity') and actual != expected:
        raise ValueError(f'{variant}: expected {expected} formulation, got {actual}')
