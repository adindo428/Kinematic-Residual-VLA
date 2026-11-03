"""Ten new subset epochs, preserving each checkpoint's optimization schedule."""
import math
from subset import require


def validate_budget(r, count):
    require(count == r['training_anchors'], 'Subset frame count differs')
    require(r['min_epochs'] >= 10, 'At least ten additional subset epochs required')
    require(r['steps'] == r['parent_updates'] + r['min_epochs'] * (count // r['global_batch']), 'Subset update budget differs')
    require(r['initialization'] == 'own_verified_checkpoint', 'Checkpoint continuation required')
    require(r['schedule_policy'] == 'preserve_parent_schedule', 'Parent LR schedule must be retained')


def lr_factor(step, r):
    if step < 1000:
        return step / 1000 if r['method'] == 'dom' else (step + 1) / 1000
    t = min(1., (step - 1000) / max(1, r['schedule_steps'] - 1000))
    floor = r['decay_lr'] / r['lr']
    return floor + (1-floor)*.5*(1+math.cos(math.pi*t))


def initial_progress(r):
    return {'updates': r['parent_updates'], 'samples': r['parent_samples'], 'epoch': 0, 'cursor': 0}


def validate_progress(p, r):
    end = r['training_anchors'] // r['global_batch'] * r['global_batch']
    updates = p['updates'] - r['parent_updates']
    samples = p['samples'] - r['parent_samples']
    require(samples == updates * r['global_batch'], 'New samples/update counters differ')
    require(samples == p['epoch'] * end + p['cursor'], 'Subset epoch/cursor differs')
    require(0 <= p['epoch'] <= r['min_epochs'] and 0 <= p['cursor'] < end, 'Invalid subset progress')
    require(p['cursor'] % r['global_batch'] == 0 and 0 <= updates <= r['steps']-r['parent_updates'], 'Invalid subset cursor/update')


def advance(p, r):
    p['updates'] += 1
    p['samples'] += r['global_batch']
    p['cursor'] += r['global_batch']
    if p['cursor'] == r['training_anchors'] // r['global_batch'] * r['global_batch']:
        p['epoch'] += 1
        p['cursor'] = 0
    validate_progress(p, r)
