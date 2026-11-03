import json
from pathlib import Path
from huggingface_hub import get_hf_file_metadata,hf_hub_url,HfApi
root=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/domv-bnfix-ddp-20261011')
m=json.loads((root/'manifest.json').read_text());data=Path(m['dataset_root'])
v=json.loads((root/'video_hashes.json').read_text())
small=[{'path':key,**value} for key,value in v.items() if value['bytes']<=128]
print(json.dumps({'header_only_retained_videos':small}),flush=True)
path='videos/chunk-195/observation.images.opst_cam/episode_195603.mp4'
for revision in (m['dataset_revision'],'main'):
 try:
  f=get_hf_file_metadata(hf_hub_url('hzxie/DOM',path,repo_type='dataset',revision=revision))
  print(json.dumps({'revision':revision,'commit':f.commit_hash,'bytes':f.size,'etag':f.etag}),flush=True)
 except Exception as exc:print(json.dumps({'revision':revision,'error_type':type(exc).__name__}),flush=True)
with (data/'meta/camera.jsonl').open() as stream:
 for line in stream:
  row=json.loads(line)
  if row.get('episode_index')==195603:
   print(json.dumps({'camera_metadata':row}),flush=True);break
for repo in HfApi().list_datasets(author='hzxie'):
 if any(k in repo.id.lower() for k in ('dom','dynamic')):print(json.dumps({'possible_raw_dataset':repo.id}),flush=True)
