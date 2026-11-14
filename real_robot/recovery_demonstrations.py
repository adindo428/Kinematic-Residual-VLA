"""Attach operator-declared recovery provenance to existing expert sessions.

This module never records or commands motion and never creates action labels.
"""
import argparse
import json
from pathlib import Path


def validate_context(context):
    if context.get('schema')!='krvla_expert_recovery_v1':raise ValueError('Wrong recovery schema')
    for key in ['operator','parent_checkpoint','scenario','deviation_source']:
        if not isinstance(context.get(key),str) or not context[key].strip():raise ValueError('Missing '+key)
    if context.get('expert_corrective_commands') is not True:
        raise ValueError('Recovery labels must be actual expert corrective commands')
    if context.get('synthetic_labels') is not False:raise ValueError('Synthetic recovery labels are unsupported')
    if context['deviation_source'] not in ['actual_policy_deviation','expert_recreated_observed_deviation']:
        raise ValueError('Declare actual policy deviation or expert recreation explicitly')
    return dict(context)


def register(session,context):
    session=Path(session);context=validate_context(context)
    path=session/'manifest.json';manifest=json.loads(path.read_text())
    if manifest.get('collector')!='human_leader' or manifest.get('mode')!='execute':
        raise ValueError('Recovery session must contain recorded human leader commands')
    if not list((session/'episodes').glob('*.json')):raise ValueError('No recorded episodes')
    if 'expert_recovery' in manifest:raise ValueError('Recovery provenance already exists; review it before replacing')
    manifest['expert_recovery']=context
    temporary=path.with_suffix('.recovery.tmp');temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(path)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    a=p.parse_args();register(a.session,json.loads(a.context.read_text()))
    print('Recovery provenance saved; original observations/actions unchanged. Export the session to retain provenance in meta/sources.jsonl.')


if __name__=='__main__':main()
