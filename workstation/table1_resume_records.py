"""Validate saved prefixes and fresh acknowledged blocks in a resumed column."""
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from report import read_records, verify_folder, verify_records

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def collect(root,listing,repeats,sources):
    protocol=read(Path(root)/'protocol.json');tests={t['name']:t for t in protocol['tests']}
    records=[];receipts=[]
    for item in sources:
        folder=Path(item['folder']);rows=read_records(folder)
        if item.get('prefix'):
            assert sha(folder/'trial_results.jsonl')==item['records_sha256']
            groups=defaultdict(list)
            for r in rows:groups[r['env_name']].append(r)
            assert {k:len(v) for k,v in groups.items()}==item['counts']
            for name,rs in groups.items():
                verify_records(rs,[name],len(rs),{name:tests[name]['horizon_seconds']})
            # This is a durable completed-trial prefix, not a claim that its
            # interrupted client acknowledged a complete suite.
        else:
            verify_folder(folder,root,Path(item['listing']),item['repeats'])
        offset=item.get('repeat_offset',0)
        records.extend({**r,'trial':r['trial']+offset} for r in rows)
        receipts.append({'folder':str(folder),'records_sha256':sha(folder/'trial_results.jsonl'),
                         'client_log_sha256':sha(folder/'client.log') if (folder/'client.log').exists() else None})
    names=[Path(l).stem for l in Path(listing).read_text().splitlines() if l.strip()]
    proof=verify_records(records,names,repeats,{n:tests[n]['horizon_seconds'] for n in names})
    return records,proof,receipts

def compose(folder,root,listing,repeats,sources):
    records,_,receipts=collect(root,listing,repeats,sources)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'trial_results.jsonl').exists()
    (folder/'trial_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (folder/'composition.json').write_text(json.dumps({'kind':'resume_prefix','sources':sources,'source_receipts':receipts},indent=2)+'\n')
    return verify(folder,root,listing,repeats)

def verify(folder,root,listing,repeats):
    receipt=read(Path(folder)/'composition.json')
    expected,proof,current=collect(root,listing,repeats,receipt['sources'])
    assert current==receipt['source_receipts'],'Resume source changed'
    key=lambda r:(r['env_name'],r['trial'])
    assert sorted(read_records(folder),key=key)==sorted(expected,key=key),'Resume aggregate changed'
    return {**proof,'aggregation':'verified saved prefix plus acknowledged fresh blocks',
            'source_folders':[s['folder'] for s in receipt['sources']]}
