"""Combine disjoint verified configuration blocks without inventing client ACKs."""
import hashlib
import json
from pathlib import Path

from report import read_records, verify_folder, verify_records


def read(path): return json.loads(Path(path).read_text())


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_records(root, sources, listing, repeats):
    target=read(Path(root)/'protocol.json')
    tests={t['name']:t for t in target['tests']}
    config=read(Path(target['trained_checkpoint'])/'config.json')
    records=[];covered=set();receipts=[]
    for item in sources:
        source_root=Path(item['root']);folder=Path(item['folder']);source_listing=Path(item['listing'])
        protocol=read(source_root/'protocol.json')
        assert protocol['checkpoint_weights_sha256']==target['checkpoint_weights_sha256']
        assert protocol['policy_source']==target['policy_source']
        assert read(Path(protocol['trained_checkpoint'])/'config.json')==config
        for key in ('physics_time_step_s','rotation','delta_actions','continuous_inference_and_latency_aware_streaming','requested_runtime','seed_policy'):
            assert protocol[key]==target[key],key
        source_tests={t['name']:t for t in protocol['tests']}
        names={Path(n).stem for n in source_listing.read_text().splitlines() if n.strip()}
        assert not covered.intersection(names),'Duplicate configurations in reused blocks'
        for name in names:
            assert source_tests[name]['config_sha256']==tests[name]['config_sha256']
            assert source_tests[name]['seed']==tests[name]['seed']
        covered.update(names)
        proof=verify_folder(folder,source_root,source_listing,repeats)
        records.extend(read_records(folder))
        receipts.append({**item,'verification':proof,'records_sha256':sha(folder/'trial_results.jsonl'),
                         'protocol_sha256':sha(source_root/'protocol.json'),'client_log_sha256':sha(folder/'client.log')})
    names=[Path(n).stem for n in Path(listing).read_text().splitlines() if n.strip()]
    assert covered==set(names),'Combined blocks must cover every requested configuration exactly once'
    proof=verify_records(records,names,repeats,{n:tests[n]['horizon_seconds'] for n in names})
    return records,proof,receipts


def compose(folder,root,listing,repeats,sources):
    folder=Path(folder)
    records,_,receipts=source_records(root,sources,listing,repeats)
    folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'trial_results.jsonl').exists(),'Do not overwrite an existing aggregate'
    (folder/'trial_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (folder/'composition.json').write_text(json.dumps({'sources':sources,'source_receipts':receipts,
        'scored_once':True,'all_outcomes_reused_without_selection':True},indent=2)+'\n')
    return verify_composed(folder,root,listing,repeats)


def verify_composed(folder,root,listing,repeats):
    folder=Path(folder);receipt=read(folder/'composition.json')
    expected,proof,receipts=source_records(root,receipt['sources'],listing,repeats)
    assert receipts==receipt['source_receipts'],'Reused source evidence changed'
    order=lambda r:(r['env_name'],r['trial'])
    assert sorted(read_records(folder),key=order)==sorted(expected,key=order),'Aggregate differs from verified source records'
    return {**proof,'aggregation':'disjoint verified configuration blocks; source client ACKs retained',
            'source_folders':[s['folder'] for s in receipt['sources']],
            'reused_trials':receipts[0]['verification']['trials']}
