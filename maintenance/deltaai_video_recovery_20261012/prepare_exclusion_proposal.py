"""Prepare, but do not apply, the six-episode correction to the frozen study."""
import hashlib,json
from pathlib import Path
root=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/domv-bnfix-ddp-20261011')
out=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-bnfix-domv-20261011/data-correction-proposal')
out.mkdir(exist_ok=True)
ids=[194633,194760,195154,195405,195417,195603]
s=json.loads((root/'split.json').read_text());e=json.loads((root/'episodes.json').read_text());v=json.loads((root/'video_hashes.json').read_text());m=json.loads((root/'manifest.json').read_text())
assert all(ep in s['splits']['train'] for ep in ids)
rows=[row for row in e if row['episode_index'] in ids]
assert len(rows)==6
before=len(s['splits']['train']);s['splits']['train']=[ep for ep in s['splits']['train'] if ep not in ids]
report={'status':'proposal_only_not_applied','reason':'Six official pinned MP4 files are 48-byte empty containers; each lacks one required camera','episode_ids':ids,'train_episodes_before':before,'train_episodes_after':len(s['splits']['train']),'excluded_train_frames':sum(row['length'] for row in rows),'train_frames_before':m['train_frames'],'train_frames_after':m['train_frames']-sum(row['length'] for row in rows),'validation_test_membership':'unchanged','normalization_and_initialization':'preserve frozen paired initialization and normalization','must_apply_to_both_servers_and_all_study_methods':True,'original_split_sha256':m['split_sha256'],'affected_videos':[{"path":k,**val} for k,val in v.items() if val['bytes']<=128]}
# Keep the proposal separate; no frozen family or dataset file is modified.
(out/'split.proposed.json').write_text(json.dumps(s,indent=2)+'\n')
report['proposed_split_sha256']=hashlib.sha256((out/'split.proposed.json').read_bytes()).hexdigest()
(out/'proposal.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
