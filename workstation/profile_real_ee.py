import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
root=Path('scratch/real_robot/training_releases/KRVLA-5fd5ef3e4256-extrapolated-80ms-v2')
train=set(json.loads((root/'meta/splits.json').read_text())['train'])
usable=json.loads((root/'meta/quality_policy.json').read_text())['usable_frames']
rows=[json.loads(l) for l in (root/'meta/episodes.jsonl').open()]
metrics={k:[] for k in ['speed_mps','acceleration_mps2','angular_speed_deg_s','angular_acceleration_deg_s2']}
positions=[];examples=[]
for row in rows:
 ep=row['episode_index']
 if ep not in train or not any('striped ball' in x for x in row['tasks']):continue
 indices=usable.get(str(ep),[])
 if len(indices)<3:continue
 p=root/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
 t=pq.read_table(p,columns=['action','observation.state']);a=np.array(t['action'].to_pylist());s=np.array(t['observation.state'].to_pylist());positions.append(a[indices,:3]);mask=np.zeros(len(a),bool);mask[indices]=True
 pairs=mask[1:]&mask[:-1];triples=mask[2:]&mask[1:-1]&mask[:-2]
 v=np.diff(a[:,:3],axis=0)/.04;acc=np.diff(v,axis=0)/.04
 r=Rotation.from_euler('xyz',a[:,3:6]).as_matrix();rot=Rotation.from_matrix(r[1:]@r[:-1].transpose(0,2,1)).as_rotvec()/.04
 metrics['speed_mps'].extend(np.linalg.norm(v[pairs],axis=1));metrics['acceleration_mps2'].extend(np.linalg.norm(acc[triples],axis=1))
 metrics['angular_speed_deg_s'].extend(np.rad2deg(np.linalg.norm(rot[pairs],axis=1)));metrics['angular_acceleration_deg_s2'].extend(np.rad2deg(np.linalg.norm((np.diff(rot,axis=0)/.04)[triples],axis=1)))
 if len(examples)<3:
  segments=np.split(np.array(indices),np.flatnonzero(np.diff(indices)!=1)+1);segment=max(segments,key=len)
  if len(segment)>=60:examples.append(dict(episode=ep,start_frame=int(segment[0]),actions=a[segment].tolist(),states=s[segment].tolist()))
position=np.vstack(positions)
report=dict(task='striped_ball',split='train',episodes=len(positions),dt_s=.04,
 action_position_quantiles={str(q):np.quantile(position,q,axis=0).tolist() for q in [0,.01,.05,.5,.95,.99,1]},
 metrics={k:{str(q):float(np.quantile(v,q)) for q in [.5,.9,.95,.99]} for k,v in metrics.items()},
 notes='Human commanded EE profiles. Finite-difference accelerations include measurement/teleoperation noise; quantiles are descriptive, not independently certified hardware limits.')
out=Path('scratch/real_robot/finetuning-runs/20261109T014917Z/evaluation')
(out/'striped_ball_train_ee_profile.json').write_text(json.dumps(report,indent=2)+'\n')
(out/'striped_ball_train_ee_examples.json').write_text(json.dumps(examples)+'\n')
print(json.dumps(report,indent=2))
