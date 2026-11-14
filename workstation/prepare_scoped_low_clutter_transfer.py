"""Build a bounded copy manifest and aggregate train-only DOM velocity stats."""
import json
import os
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq


def main():
    assert os.environ['SLURM_JOB_ID']=='20066627'
    split_text=sys.stdin.read();split=json.loads(split_text)['splits']
    bundle=Path('/scratch/aparame/Dynamic_VLA/low-clutter-transfer-20261112')
    bundle.mkdir(exist_ok=True)
    target='/scratch/arai/Dynamic_VLA/low-clutter-velocity-gated-20261112'
    source=Path('/scratch/aparame/Dynamic_VLA/main8-replicas-h200-20261021')
    dataset=Path('/scratch/aparame/.cache/huggingface/lerobot/hzxie/DOM')
    manifest=[]
    def add(src,name):manifest.append([str(src),name])
    refs=json.loads((source/'checkpoints.json').read_text())
    for key,ref in refs.items():
        src=Path(ref['path']);prefix='model-assets/'+key.replace('/','--')
        for f in sorted(src.rglob('*')):
            if not f.is_file():continue
            if not key.startswith('lerobot/') and (f.suffix in {'.safetensors','.bin'} or '.safetensors.' in f.name):continue
            add(f,prefix+'/'+str(f.relative_to(src)))
        ref['path']=target+'/'+prefix
    (bundle/'checkpoints.json').write_text(json.dumps(refs,indent=2)+'\n')
    for model,campaign in {'smolvla_kr':'main8-h200-only-20261022','pi05_kr':'main8-replicas-h200-20261021','vlash_kr':'main8-replicas-h200-20261021'}.items():
        src=Path('/scratch/aparame/Dynamic_VLA')/campaign/'parents'/model
        for name in ['latest.pt','migration.json','latest.json']:
            if (src/name).is_file():add(src/name,'parents/'+model+'/'+name)
    for name in ['info.json','episodes.jsonl','camera.jsonl','tasks.jsonl']:
        add(dataset/'meta'/name,'dataset/meta/'+name)
    for ep in sorted({ep for ids in split.values() for ep in ids}):
        rel=f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet';add(dataset/rel,'dataset/'+rel)
        for camera in ['observation.images.opst_cam','observation.images.wrist_cam']:
            rel=f'videos/chunk-{ep//1000:03d}/{camera}/episode_{ep:06d}.mp4';add(dataset/rel,'dataset/'+rel)
    selected=set(split['train']);count=0;total=np.zeros(3);square=np.zeros(3);found=set()
    probes={split['train'][i] for i in [0,1,100,1000,5000,10000,15000,19999]}
    verified=0
    for line in (dataset/'meta/episodes_stats.jsonl').open():
        entry=json.loads(line);ep=entry['episode_index']
        if ep not in selected:continue
        stats=entry['stats']['observation.environment_state'];n=stats['count'][0]
        mean=np.asarray(stats['mean'][6:9],dtype=np.float64);std=np.asarray(stats['std'][6:9],dtype=np.float64)
        assert np.isfinite(mean).all() and np.isfinite(std).all()
        count+=n;total+=mean*n;square+=(std**2+mean**2)*n;found.add(ep)
        if ep in probes:
            path=dataset/f'data/chunk-{ep//1000:03d}/episode_{ep:06d}.parquet'
            a=np.asarray(pq.read_table(path,columns=['observation.environment_state'])['observation.environment_state'].to_pylist(),dtype=np.float64)[:,6:9]
            assert len(a)==n
            np.testing.assert_allclose(a.mean(0),mean,atol=2e-6,rtol=2e-5)
            np.testing.assert_allclose(a.std(0),std,atol=2e-6,rtol=2e-5)
            verified+=1
    assert found==selected and count==2376682 and verified==8
    mean=total/count;std=np.sqrt(np.maximum(square/count-mean*mean,1e-12))
    conditioning={'mean':mean.tolist(),'std':std.tolist(),'frames':count,
                  'source':'Current environment_state[6:9]; training partition only; count-weighted published episode mean/variance',
                  'raw_parquet_cross_checks':verified,'split_sha256':'a3d524277b9758a9bc33a423beb2f462433794ecb5510de1fa9929c99c331522'}
    (bundle/'conditioning-statistics.json').write_text(json.dumps(conditioning,indent=2)+'\n')
    data=json.loads((source/'data-provenance.json').read_text())
    data.update(dataset_root=target+'/dataset',split_sha256=conditioning['split_sha256'],
                velocity_source='privileged current observed simulator velocity; explicit neural inputs and analytic causally gated KR',
                phase_gate='causal_rules; previous applied gripper command and current/past observations only',
                normalization='Inherited physical state/action and analytic drift centering; new train-only neural velocity statistics')
    (bundle/'data-provenance.json').write_text(json.dumps(data,indent=2)+'\n')
    (bundle/'statistics.json').write_text((source/'statistics.json').read_text())
    (bundle/'split.json').write_text(split_text)
    for name in ['checkpoints.json','parquet_hashes.json','video_hashes.json','conditioning-statistics.json','data-provenance.json','statistics.json','split.json']:
        assert (bundle/name).is_file();add(bundle/name,name)
    vlash=Path('/scratch/aparame/Dynamic_VLA/cross-vla-krfm-20261019/upstream-vlash/vlash')
    for f in sorted(vlash.rglob('*')):
        if f.is_file() and f.suffix in {'.py','.json','.yaml','.yml','.toml','.txt'}:
            add(f,'upstream-vlash/vlash/'+str(f.relative_to(vlash)))
    assert any(name.startswith('upstream-vlash/') for _,name in manifest)
    assert len({name for _,name in manifest})==len(manifest)
    (bundle/'transfer-manifest.json').write_text(json.dumps(manifest))
    print(json.dumps({'files':len(manifest),'train_velocity_statistics':conditioning}),flush=True)


if __name__=='__main__':main()
