"""Retain source ACKs and reject overlaps or changed evidence in trial reuse."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_low_clutter_table1  # Installs the pinned report module path.
from table1_trial_reuse import compose, verify_composed, source_records


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    checkpoint=tmp_path/'checkpoint';write(checkpoint/'config.json',{'gate':'causal'})
    tests=[{'name':n,'config_sha256':n,'seed':i,'horizon_seconds':12} for i,n in enumerate(['a','b'])]
    protocol={'trained_checkpoint':str(checkpoint),'tests':tests,'checkpoint_weights_sha256':'weights',
              'policy_source':'source',**{k:True for k in ('physics_time_step_s','rotation','delta_actions',
              'continuous_inference_and_latency_aware_streaming','requested_runtime','seed_policy')}}
    root=tmp_path/'target';write(root/'protocol.json',protocol)
    listing=root/'all.txt';listing.write_text('a.json\nb.json\n')
    sources=[]
    for name in ('a','b'):
        r=tmp_path/name;write(r/'protocol.json',protocol)
        f=r/'full';f.mkdir();ls=r/'cases.txt';ls.write_text(name+'.json\n')
        records=[{'env_name':name,'trial':i,'status':i%2,'success':i%2==0,'n_steps':10,
                  'step_dt_s':.04,'task_time_s':.4} for i in (1,2)]
        (f/'trial_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (f/'client.log').write_text('Evaluation completed. Total tests run: 2\nTest suite done with success rates:\n')
        sources.append({'root':str(r),'folder':str(f),'listing':str(ls)})
    return root,listing,sources


def test_composition_revalidates_sources_without_fabricated_aggregate_ack(tmp_path):
    root,listing,sources=fixture(tmp_path);out=root/'aggregate'
    v=compose(out,root,listing,2,sources)
    assert v['trials']==4 and v['successes']==2 and v['reused_trials']==2
    assert not (out/'client.log').exists()
    assert verify_composed(out,root,listing,2)['verified']
    p=Path(sources[0]['folder'])/'client.log';p.write_text(p.read_text()+'changed evidence\n')
    with pytest.raises(AssertionError):verify_composed(out,root,listing,2)


def test_duplicate_configuration_rejected(tmp_path):
    root,listing,sources=fixture(tmp_path)
    with pytest.raises(AssertionError,match='Duplicate configurations'):
        source_records(root,[sources[0],sources[0]],listing,2)


def test_changed_checkpoint_protocol_rejected(tmp_path):
    root,listing,sources=fixture(tmp_path)
    path=Path(sources[0]['root'])/'protocol.json';p=json.loads(path.read_text());p['checkpoint_weights_sha256']='other';write(path,p)
    with pytest.raises(AssertionError):source_records(root,sources,listing,2)
