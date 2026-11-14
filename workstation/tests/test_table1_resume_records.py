"""Resume prefixes once; remap fresh tail IDs without losing failures."""
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_low_clutter_table1
from table1_resume_records import compose,collect,verify,sha

def row(name,trial,success):
    return {'env_name':name,'trial':trial,'status':0 if success else 1,'success':success,
            'n_steps':10,'step_dt_s':.04,'task_time_s':.4}

def fixture(tmp_path):
    root=tmp_path/'root';root.mkdir()
    (root/'protocol.json').write_text(json.dumps({'tests':[{'name':n,'horizon_seconds':12} for n in ['a','b']]}))
    listing=root/'all.txt';listing.write_text('a.json\nb.json\n')
    sources=[]
    for key,rows,n,offset in [('prefix',[row('a',1,False)],0,0),('a_tail',[row('a',1,True)],1,1),('b',[row('b',1,False),row('b',2,True)],2,0)]:
        folder=tmp_path/key;folder.mkdir()
        (folder/'trial_results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (folder/'client.log').write_text(f'Evaluation completed. Total tests run: {n}\nTest suite done with success rates:\n')
        if key=='prefix':sources.append({'folder':str(folder),'prefix':True,'counts':{'a':1},'records_sha256':sha(folder/'trial_results.jsonl')})
        else:
            ls=folder/'cases.txt';ls.write_text(rows[0]['env_name']+'.json\n')
            sources.append({'folder':str(folder),'listing':str(ls),'repeats':n,'repeat_offset':offset})
    return root,listing,sources

def test_prefix_and_offset_tail_cover_every_trial_once(tmp_path):
    root,listing,sources=fixture(tmp_path);out=root/'aggregate'
    v=compose(out,root,listing,2,sources)
    assert v['verified'] and v['trials']==4 and v['successes']==2
    assert not (out/'client.log').exists()
    assert verify(out,root,listing,2)['verified']

def test_duplicate_tail_ids_rejected(tmp_path):
    root,listing,sources=fixture(tmp_path);sources[1]['repeat_offset']=0
    with pytest.raises(AssertionError):collect(root,listing,2,sources)

def test_changed_saved_prefix_rejected(tmp_path):
    root,listing,sources=fixture(tmp_path)
    (Path(sources[0]['folder'])/'trial_results.jsonl').write_text(json.dumps(row('a',1,True))+'\n')
    with pytest.raises(AssertionError):collect(root,listing,2,sources)
