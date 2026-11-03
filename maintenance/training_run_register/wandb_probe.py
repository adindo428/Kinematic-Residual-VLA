"""Read-only W&B queries, executed through an authenticated SSH bridge.

Credentials remain on the bridge host. REQUEST contains approved entity/project
run identities; no run mutation, training command, or credential output occurs.
"""
import base64
import json
import netrc
import os
import urllib.request
from datetime import datetime, timezone


def probe_wandb(requests):
    key = os.environ.get('WANDB_API_KEY')
    if not key:
        auth = netrc.netrc().authenticators('api.wandb.ai')
        key = auth[2] if auth else None
    if not key:
        raise RuntimeError('W&B authentication unavailable')
    headers = {'Content-Type': 'application/json',
               'Authorization': 'Basic ' + base64.b64encode(('api:' + key).encode()).decode()}
    query = '''query($entity:String!, $project:String!, $run:String!) {
      project(name:$project, entityName:$entity) {
        run(name:$run) { name displayName state config summaryMetrics }
      }
    }'''
    results = []
    for identity in requests:
        variables = {k: identity[k] for k in ('entity', 'project', 'run')}
        request = urllib.request.Request('https://api.wandb.ai/graphql',
            data=json.dumps({'query': query, 'variables': variables}).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.load(response)
            run = payload.get('data', {}).get('project', {}).get('run')
            if payload.get('errors') or not run:
                raise ValueError('W&B run unavailable')
            config = json.loads(run['config'])
            results.append({**variables, 'state': run['state'], 'name': run['displayName'],
                'config': {k: v.get('value') for k, v in config.items() if k != '_wandb'},
                'summary': json.loads(run['summaryMetrics']),
                'checked_at': datetime.now(timezone.utc).isoformat()})
        except Exception as exc:
            # Do not expose headers, credentials or signed URLs in error output.
            results.append({**variables, 'error': type(exc).__name__})
    return results


if __name__ == '__main__':
    print(json.dumps(probe_wandb(REQUEST)))
