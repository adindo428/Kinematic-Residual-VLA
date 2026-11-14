"""A fresh subset experiment: complete epochs and a schedule sized to its data."""
import math
from subset import require


def validate_budget(recipe, count):
    batch = recipe['global_batch']
    require(count == recipe['training_anchors'] and count >= batch, 'Subset anchor count changed')
    require(recipe['min_epochs'] >= 5, 'At least five complete subset epochs required')
    require(recipe['steps'] == recipe['min_epochs'] * (count // batch), 'Epoch budget differs from actual subset')
    require(recipe['schedule_steps'] == recipe['steps'], 'Cosine schedule must span the subset budget')
    require(0 < recipe['warmup_updates'] < recipe['steps'], 'Invalid subset warmup')
    require(recipe['schedule_policy'] == 'fresh_subset_cosine', 'Wrong experiment schedule')
    require(recipe['initialization'] == 'released_pretrained', 'Full-DOM checkpoints contaminate a 20k-data claim')


def lr_factor(step, recipe):
    warmup = recipe['warmup_updates']
    if step < warmup:
        return (step + 1) / warmup
    t = min(1., (step - warmup) / (recipe['schedule_steps'] - warmup))
    floor = recipe['decay_lr'] / recipe['lr']
    return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * t))


def validate_progress(p, recipe):
    end = recipe['training_anchors'] // recipe['global_batch'] * recipe['global_batch']
    require(p['samples'] == p['updates'] * recipe['global_batch'], 'Samples/update counters differ')
    require(p['samples'] == p['epoch'] * end + p['cursor'], 'Epoch/cursor differs from consumed anchors')
    require(0 <= p['epoch'] <= recipe['min_epochs'], 'Invalid epoch')
    require(0 <= p['cursor'] < end and p['cursor'] % recipe['global_batch'] == 0, 'Invalid cursor')
    require(0 <= p['updates'] <= recipe['steps'], 'Invalid updates')


def advance(p, recipe):
    p['updates'] += 1
    p['samples'] += recipe['global_batch']
    p['cursor'] += recipe['global_batch']
    end = recipe['training_anchors'] // recipe['global_batch'] * recipe['global_batch']
    if p['cursor'] == end:
        p['epoch'] += 1
        p['cursor'] = 0
    validate_progress(p, recipe)
