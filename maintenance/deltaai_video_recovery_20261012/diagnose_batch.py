"""Check the exact rank-three batches around the first production decoder error."""
import json,hashlib,time
from pathlib import Path
import numpy as np
import torchcodec.decoders
root=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/domv-bnfix-ddp-20261011')
manifest=json.loads((root/'manifest.json').read_text());dataset=Path(manifest['dataset_root'])
episodes={e['episode_index']:e for e in json.loads((root/'episodes.json').read_text())}
train=json.loads((root/'split.json').read_text())['splits']['train']
lengths=np.array([episodes[e]['length'] for e in train],dtype=np.int64)
ends=np.cumsum(lengths);order=np.random.default_rng(42).permutation(int(ends[-1]))
video_hashes=json.loads((root/'video_hashes.json').read_text())
info=json.loads((dataset/'meta/info.json').read_text())
keys=['observation.images.opst_cam','observation.images.wrist_cam']
records=[]
for update in (81,82):
 indices=order[(update-1)*352+264:(update-1)*352+352]
 for ep in dict.fromkeys(train[i] for i in np.searchsorted(ends,indices,side='right')):
  for camera in keys:
   rel=info['video_path'].format(episode_chunk=ep//info['chunks_size'],video_key=camera,episode_index=ep)
   path=dataset/rel;raw=path.read_bytes();expected=video_hashes[rel]
   record={'update':update,'episode':ep,'camera':camera,'path':str(path),'bytes':len(raw),'header_hex':raw[:32].hex(),'sha256':hashlib.sha256(raw).hexdigest(),'frozen_sha256':expected['sha256']}
   try:
    decoder=torchcodec.decoders.VideoDecoder(raw,seek_mode='approximate',num_ffmpeg_threads=1)
    decoder.get_frames_at(indices=[0])
    record['decode']='okay'
   except Exception as exc:
    record['decode']='failed';record['error']=str(exc)
    print(json.dumps(record),flush=True)
   records.append(record)
report={'records':records,'checked':len(records),'failed':sum(r['decode']=='failed' for r in records),'all_hashes_match':all(r['sha256']==r['frozen_sha256'] for r in records)}
out=Path('/work/nvme/bivv/aparameshwaran/dynamicvla/setup-bnfix-domv-20261011/video-failure-diagnostic.json')
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
