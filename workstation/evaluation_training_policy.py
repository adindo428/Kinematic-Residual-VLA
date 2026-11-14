"""Apply the user's limit on full-data fine-tuning before the 20k continuation."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLICY = REPO / 'scratch/dom45/evaluations/training_history_policy.json'
EXCLUDED_LABELS = {'old KRV100', 'KRV100 · 8.09 epochs', 'DOM + velocity'}


def load_policy():
    return json.loads(POLICY.read_text())


def require_eligible(name, entry=None, policy=None):
    policy = load_policy() if policy is None else policy
    record = policy['models'].get(name)
    if record is None:
        raise ValueError('Training history has not been audited for ' + name)
    if record['prior_full_data_epochs'] >= policy['prior_full_data_epochs_exclusive_limit']:
        raise ValueError(f"{name} has {record['prior_full_data_epochs']:.4f} prior full-data epochs; limit is strictly less than one")
    if not record['training_complete']:
        raise ValueError(name + ' is held until its final training checkpoint is available')
    if entry is not None and record.get('identity') and entry.get('identity') != record['identity']:
        raise ValueError('Audited model identity differs: ' + name)


def filter_comparison_rows(rows):
    return [row for row in rows if row['Method'] not in EXCLUDED_LABELS]


def comparison_note():
    return ('Eligibility: 20k-split fine-tuning with less than one prior full-data epoch, as authorized by the user. '
            'Fresh DOM/KRV100 have zero additional full-data fine-tuning before the 20k stage. '
            'SmolVLA ± KR have approximately 0.057 prior epochs, VLASH ± KR 0.043, pi0.5 baseline 0.028 and pi0.5 + KR 0.086. '
            'DOM + velocity (3.438 prior epochs) and the older KRV100 run (two prior epochs) are excluded. '
            'VLASH baseline and pi0.5 + KR remain pending final training checkpoints. '
            'This tolerance does not make these continuation checkpoints strictly 20k-only; prior exposure remains disclosed.')
