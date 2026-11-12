"""Current object velocity through the existing language input; no learned layers."""
import torch

VELOCITY_PROMPT = dict(mode='language_prompt_v1', frame='robot_base', units='mm/s',
                       resolution_mm_s=1, source='current_quality_checked_observation',
                       held_policy='zero_when_velocity_not_required')


def velocity_prompt_batch(batch, contract):
    if contract is None:
        return batch
    if contract != VELOCITY_PROMPT:
        raise ValueError('Unknown velocity conditioning contract')
    velocity = torch.as_tensor(batch['velocity']).detach()
    required = torch.as_tensor(batch['velocity_required'], device=velocity.device)
    if velocity.shape != (len(batch['task']), 3) or required.shape != (len(batch['task']),):
        raise ValueError('Expected current object velocity [batch, 3] and required mask [batch]')
    if required.dtype != torch.bool or not torch.isfinite(velocity).all():
        raise ValueError('Object velocity and quality mask must be finite and valid')
    # The dataset uses a neutral velocity after annotated holding; the same
    # quality gate applies to neural conditioning and analytical KR restoration.
    velocity = torch.where(required[:, None], velocity, torch.zeros_like(velocity))
    values = (velocity.float() * 1000).round().to(torch.int64).cpu().tolist()
    result = dict(batch)
    result['task'] = [f'Object base velocity mm/s: {x},{y},{z}; {task.strip()}'
                      for task, (x, y, z) in zip(batch['task'], values)]
    return result


class UntruncatedTokenizer:
    """Reject prompts that would silently discard velocity, instruction or state."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, texts, **kwargs):
        full = self.tokenizer(texts, padding=False, truncation=False)['input_ids']
        if any(len(ids) > kwargs['max_length'] for ids in full):
            raise ValueError('Velocity-conditioned prompt exceeds native token budget; do not truncate inputs')
        return self.tokenizer(texts, **kwargs)
