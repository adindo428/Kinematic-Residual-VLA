"""Serializable validation-based stopping, independent of checkpoint selection."""
import math


def validate_rule(rule):
    if not rule or not rule.get('enabled', False):
        return
    if (rule['patience'] < 1 or rule['min_epochs'] < 1 or
            not 0 <= rule['relative_min_delta'] < 1 or not rule['metric']):
        raise ValueError('Invalid validation stopping rule')


def should_stop(progress, rule):
    if not rule or not rule.get('enabled', False):
        return False
    state = progress.get('early_stopping', {})
    return (progress['epoch'] >= rule['min_epochs'] and
            state.get('bad_validations', 0) >= rule['patience'])


def record_validation(progress, metrics, rule):
    validate_rule(rule)
    if not rule or not rule.get('enabled', False):
        return False
    value = float(metrics[rule['metric']])
    if not math.isfinite(value) or value < 0:
        raise ValueError('Validation stopping metric must be finite and nonnegative')
    state = progress.setdefault('early_stopping', {'bad_validations': 0})
    epoch = progress['epoch']
    if epoch < state.get('last_validation_epoch', -1):
        raise ValueError('Validation stopping epochs moved backwards')
    if epoch != state.get('last_validation_epoch'):
        best = state.get('significant_best')
        if best is None or (value < best and value <= best*(1-rule['relative_min_delta'])):
            state.update(significant_best=value, best_epoch=epoch, bad_validations=0)
        else:
            state['bad_validations'] += 1
        state['last_validation_epoch'] = epoch
    return should_stop(progress, rule)
