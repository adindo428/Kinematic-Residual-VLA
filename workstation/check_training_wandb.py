"""Read-only W&B identity/project checks; never emit a credential."""
import datetime
import json
import os
from pathlib import Path

import wandb

record = {'authenticated':False, 'training_account_confirmed':False,
          'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
try:
    api = wandb.Api(timeout=20)
    user = api.viewer
    record.update(authenticated=True, username=user.username, default_entity=api.default_entity, teams=user.teams)
    entity = os.environ.get('DVLA_WANDB_ENTITY') or api.default_entity
    record['training_entity'] = entity
    record['project_visible'] = any(project.name=='Kinematic-Residual-VLA' for project in api.projects(entity=entity))
    record['project_action'] = 'use existing' if record['project_visible'] else 'create with the training run'
    record['training_account_confirmed'] = entity in user.teams or entity==user.username
except Exception as error:
    record['error_type'] = type(error).__name__
path = Path(os.environ['KRVLA_WORK'])/'checks/training-wandb.json'
path.write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
raise SystemExit(0 if record['training_account_confirmed'] else 1)
