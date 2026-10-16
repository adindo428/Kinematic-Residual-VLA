"""Stop a frozen trainer at an approved epoch boundary without changing its LR."""
import json
from pathlib import Path


class EpochLimitReached(Exception):
    def __init__(self, result):
        self.result = result


def install_epoch_limit(training, destination, control, control_sha):
    original_save = training.save_distributed
    limit = control['stop_after_completed_epochs']

    def save(root, policy, optimizer, scheduler, progress, fingerprint, wandb_id,
             best=None, milestone=False, device='cuda', final_checkpoint=False):
        reached = progress['epoch'] >= limit and not progress.get('pending_validation', False)
        if reached:
            if progress['epoch'] != limit or progress['cursor'] != 0:
                raise RuntimeError('Requested epoch boundary was exceeded')
            progress.update(completion_reason='user_epoch_limit', user_epoch_limit=limit,
                            epoch_limit_control_sha256=control_sha)
        path = original_save(root, policy, optimizer, scheduler, progress, fingerprint,
                             wandb_id, best, milestone or reached, device, final_checkpoint or reached)
        if reached:
            raise EpochLimitReached({**progress, 'fingerprint':fingerprint, 'best':best,
                                     'checkpoint':str(path), 'manifest':control['manifest'],
                                     'original_scheduler_horizon_epochs':control['scheduler_horizon_epochs']})
        return path

    training.save_distributed = save


def load_control(root, manifest, recipe, file_digest):
    path = Path(root).parent/'training_control.json'
    if not path.exists():
        return None, None
    control = json.loads(path.read_text())
    if (not control.get('user_authorized') or control['manifest'] != manifest['fingerprint']
            or control['recipe_fingerprint'] != recipe['fingerprint']
            or control['scheduler_horizon_epochs'] != recipe['epochs']
            or not 1 <= control['stop_after_completed_epochs'] <= recipe['epochs']):
        raise ValueError('Epoch limit differs from the authorized frozen experiment')
    return control, file_digest(path)
