"""Freeze the user-authorized exclusions from a completed exhaustive audit."""
import json,sys
from pathlib import Path
old,root,source=map(Path,sys.argv[1:4])
sys.path.insert(0,str(source))
from utils.dom_experiment import atomic_json,digest,file_digest,source_digest
audit=Path(sys.argv[4])
reports=[json.loads(p.read_text()) for p in sorted(audit.glob('status-*.json'))]
assert len(reports)==16 and all(r['state']=='complete' for r in reports)
assert sum(r['completed_episodes'] for r in reports)==201227
assert not any(r['metadata_errors'] for r in reports)
failures=[]
for p in audit.glob('episodes-*.jsonl'):
 for line in p.open():
  record=json.loads(line)
  if record['errors']:failures.append(record)
assert all(all('video has fewer frames than labels' in e for e in r['errors']) for r in failures), failures
ids={r['episode'] for r in failures}
assert ids, 'No exclusions needed'

assert not root.exists()
original=json.loads((old/'manifest.json').read_text());check=dict(original);assert digest({k:v for k,v in check.items() if k!='fingerprint'})==check['fingerprint']
for name,value in original['artifacts'].items():assert file_digest(old/name)==value,name
root.mkdir()
for name in original['artifacts']:
 path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.symlink_to((old/name).resolve())
for name in ('phase_cache','initialization'):
 if not (root/name).exists():(root/name).symlink_to((old/name).resolve(),target_is_directory=True)
if not (root/'initialization/model.safetensors').exists():(root/'initialization/model.safetensors').symlink_to((old/'initialization/model.safetensors').resolve())

def write(name,value):
 p=root/name
 if p.is_symlink():p.unlink()
 atomic_json(p,value)

split=json.loads((old/'split.json').read_text());assert all(ep in split['splits']['train'] for ep in ids)
split['splits']['train']=[ep for ep in split['splits']['train'] if ep not in ids];write('split.json',split)
episodes=json.loads((old/'episodes.json').read_text());removed=[e for e in episodes if e['episode_index'] in ids]
assert len(removed)==len(ids)
removed_frames=sum(e['length'] for e in removed)
write('episodes.json',[e for e in episodes if e['episode_index'] not in ids])
videos=json.loads((old/'video_hashes.json').read_text())
bad=[{'path':k,**v} for k,v in videos.items() if int(Path(k).stem.split('_')[-1]) in ids]
for name in ('parquet_hashes.json','phase_hashes.json'):
 values=json.loads((old/name).read_text());write(name,{k:v for k,v in values.items() if int(k) not in ids})
write('video_hashes.json',{k:v for k,v in videos.items() if int(Path(k).stem.split('_')[-1]) not in ids})
coverage=json.loads((old/'dataset_coverage.json').read_text())
coverage['retained_episodes']-=len(ids);coverage['excluded_episodes']+=len(ids)
coverage['retained_frames']-=removed_frames;coverage['excluded_frames']+=removed_frames
coverage['missing_or_empty_files']+=len(ids)
coverage['episode_fraction']=coverage['retained_episodes']/coverage['published_episodes']
coverage['frame_fraction']=coverage['retained_frames']/coverage['published_frames']
for e in removed:
 k=json.dumps([e['task'],e['speed_bin']]);coverage['task_speed_coverage'][k]['retained']-=1;coverage['task_speed_coverage'][k]['excluded']+=1
write('dataset_coverage.json',coverage)
selection=json.loads((old/'dataset_selection.json').read_text())
selection['episode_ids']=[ep for ep in selection['episode_ids'] if ep not in ids]
selection['excluded'].extend({'episode_index':e['episode_index'],'length':e['length'],
 'missing_or_empty_files':[v['path'] for v in bad if int(Path(v['path']).stem.split('_')[-1])==e['episode_index']],
 'reason':'Exhaustive audit found a video shorter than its action labels'} for e in sorted(removed,key=lambda e:e['episode_index']))
write('dataset_selection.json',selection)
correction={'user_approved':True,'approval_date':'2026-10-12','excluded_episode_ids':sorted(ids),'excluded_train_frames':removed_frames,
 'reason':'Completed Palmetto exhaustive decode audit; user authorized clean dataset deployment on both clusters',
 'original_split_sha256':original['split_sha256'],'split_sha256':file_digest(root/'split.json'),
 'validation_test_membership':'unchanged','normalization':'original frozen paired statistics retained','videos':bad,'audit_failures':failures,'parent_exclusions':original['data_correction']['excluded_episode_ids']}
write('data_correction.json',correction)
manifest=dict(original);manifest.pop('fingerprint')
manifest.update(source_sha256=source_digest(source),initialization=str(root/'initialization'),split_sha256=correction['split_sha256'],
 coverage=coverage,frames=coverage['retained_frames'],episodes=len(episodes)-len(ids),train_frames=original['train_frames']-removed_frames,counts={k:len(v) for k,v in split['splits'].items()},data_correction=correction,
 runtime_adaptation_parent=original['fingerprint'])
manifest['parquets_sha256']=digest(json.loads((root/'parquet_hashes.json').read_text()))
manifest['artifacts']={name:file_digest(root/name) for name in list(original['artifacts'])+['data_correction.json']}
manifest['fingerprint']=digest(manifest);atomic_json(root/'manifest.json',manifest)
recipe=json.loads((old/'recipe.json').read_text());recipe.pop('fingerprint');recipe['manifest']=manifest['fingerprint'];recipe['fingerprint']=digest(recipe);atomic_json(root/'recipe.json',recipe)
(root/'provenance').mkdir();atomic_json(root/'provenance/completed-audit.json',{'shards':reports,'failures':failures,'retained_clean_episodes':len(episodes)-len(ids)});atomic_json(root/'provenance/parent-manifest.json',original);(root/'logs').mkdir()
for name in ('evaluation_plan.json',):
 if (old/name).is_file() and not (root/name).exists():(root/name).symlink_to((old/name).resolve())
report={'family':str(root),'source':str(source),'manifest':manifest['fingerprint'],'source_sha256':manifest['source_sha256'],'recipe':recipe['fingerprint'],
 'split_sha256':manifest['split_sha256'],'train_frames':manifest['train_frames'],'counts':manifest['counts'],'retained_frames':manifest['frames'],'retained_episodes':coverage['retained_episodes']}
atomic_json(root/'provenance/data-correction-deployment.json',report);print(json.dumps(report),flush=True)
