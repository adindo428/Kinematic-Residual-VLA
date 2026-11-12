"""Download and SHA-verify the six pinned policy snapshots without loading them."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
    campaign=json.loads(a.campaign.read_text());root=a.campaign.parent
    status=root/'downloads.json';state={'revision':campaign['revision'],'models':{}}
    def save():
        temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(status)
    for entry in campaign['entries']:
        name=entry['family'];state['models'][name]={'state':'downloading'};save()
        try:
            path=Path(hf_hub_download(campaign['repo'],entry['checkpoint_file'],revision=campaign['revision'],local_dir=root/'hub'))
            state['models'][name]={'state':'verifying','path':str(path)};save()
            h=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
            if h.hexdigest()!=entry['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
            state['models'][name]={'state':'verified','path':str(path.resolve()),'sha256':h.hexdigest()};save()
            print(name,'verified',flush=True)
        except BaseException as exc:
            state['models'][name]={'state':'failed','error':str(exc)};save();raise
    state['state']='completed';save()


if __name__=='__main__':main()
