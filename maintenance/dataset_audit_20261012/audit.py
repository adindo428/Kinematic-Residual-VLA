"""Read-only exhaustive retained-DOM audit; output is outside the frozen family."""
import argparse, hashlib, io, json, multiprocessing as mp, os, signal, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torchcodec.decoders import VideoDecoder


def digest(raw): return hashlib.sha256(raw).hexdigest()
def read_json(p): return json.loads(p.read_text())
def atomic(p, data):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2)+'\n');t.replace(p)
def expired(*_): raise TimeoutError('episode audit exceeded 180 seconds')
def check(ep):
    eid=ep['episode_index']; n=ep['length']; errors=[]; frames=0
    def require(ok,msg):
        if not ok: raise ValueError(msg)
    signal.signal(signal.SIGALRM,expired);signal.alarm(180)
    try:
        rel=INFO['data_path'].format(episode_chunk=eid//INFO['chunks_size'],episode_index=eid)
        raw=(DATA/rel).read_bytes();require(digest(raw)==PH[str(eid)],'parquet hash mismatch')
        table=pq.read_table(io.BytesIO(raw));require(table.num_rows==n,'parquet row count mismatch')
        vals={k:np.asarray(table[k].to_pylist()) for k in ['action','observation.state','observation.environment_state','timestamp','frame_index','episode_index','task_index']}
        for k,width in [('action',7),('observation.state',6),('observation.environment_state',9)]:
            require(vals[k].shape==(n,width),k+' wrong shape')
            require(np.isfinite(vals[k]).all(),k+' contains nonfinite values')
        ts=vals['timestamp'].reshape(-1).astype(np.float32)
        require(np.isfinite(ts).all() and np.all(np.diff(ts)>0),'invalid timestamps')
        require(np.array_equal(vals['frame_index'].reshape(-1),np.arange(n)),'frame indices invalid')
        require(np.all(vals['episode_index']==eid),'episode IDs mismatch')
        require(set(vals['task_index'].reshape(-1)).issubset(TASKS),'missing instruction task IDs')
        require(np.max(np.abs(ts-np.arange(n)/INFO['fps']))<1e-4,'timestamps disagree with nominal FPS')
    except Exception as ex:
        errors.append('parquet: '+str(ex));ts=np.arange(n,dtype=np.float32)/INFO['fps']
    for camera in CAMERAS:
        try:
            rel=INFO['video_path'].format(episode_chunk=eid//INFO['chunks_size'],episode_index=eid,video_key=camera)
            raw=(DATA/rel).read_bytes();require(digest(raw)==VH[rel]['sha256'],'video hash mismatch')
            decoder=VideoDecoder(raw,seek_mode='approximate',num_ffmpeg_threads=1)
            require(len(decoder)>=n,'video has fewer frames than labels')
            for start in range(0,n,16):
                query=ts[start:start+16];indices=[round(float(t)*INFO['fps']) for t in query]
                batch=decoder.get_frames_at(indices=indices)
                shape=INFO['features'][camera]['shape']
                require(tuple(batch.data.shape)==(len(query),shape[2],shape[0],shape[1]),'decoded frame shape mismatch')
                # Stronger paired timestamp check than the trainer's nearest-time check.
                require(np.max(np.abs(batch.pts_seconds.numpy().astype(np.float32)-query))<1e-4,'decoded timestamps outside training tolerance')
                frames+=len(query)
            del decoder,raw
        except Exception as ex: errors.append(camera+': '+str(ex))
    try:
        raw=(FAMILY/'phase_cache'/('%06d.npz'%eid)).read_bytes()
        require(digest(raw)==GH[str(eid)],'phase hash mismatch')
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            require(len(z['state'])==n and len(z['phase'])==n,'phase length mismatch')
            require(z['state'].shape==(n,4) and np.isfinite(z['state']).all(),'phase state shape/nonfinite')
            require(np.all(z['state'][:,3]>0),'invalid phase fade duration')
            require(np.isin(z['phase'],np.arange(7)).all(),'invalid phase IDs')
    except Exception as ex: errors.append('phase: '+str(ex))
    signal.alarm(0)
    return {'episode':eid,'frames_decoded':frames,'errors':errors}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',required=True);p.add_argument('--output',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=16);p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int)
    a=p.parse_args();FAMILY=Path(a.family);OUT=Path(a.output);OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    MAN=read_json(FAMILY/'manifest.json');DATA=Path(MAN['dataset_root']);INFO=read_json(DATA/'meta/info.json')
    CAMERAS=['observation.images.opst_cam','observation.images.wrist_cam']
    EPS=read_json(FAMILY/'episodes.json');SPLIT=read_json(FAMILY/'split.json')['splits']
    PH=read_json(FAMILY/'parquet_hashes.json');VH=read_json(FAMILY/'video_hashes.json');GH=read_json(FAMILY/'phase_hashes.json')
    TASKS={json.loads(line)['task_index'] for line in (DATA/'meta/tasks.jsonl').read_text().splitlines()}
    metadata_errors=[];warnings=[]
    for name,h in MAN['artifacts'].items():
        if digest((FAMILY/name).read_bytes())!=h: metadata_errors.append('artifact mismatch '+name)
    ids=[e['episode_index'] for e in EPS];flat=[x for v in SPLIT.values() for x in v]
    if len(ids)!=len(set(ids)) or len(flat)!=len(set(flat)) or set(flat)!=set(ids):metadata_errors.append('split coverage/uniqueness mismatch')
    byid={eid:s for s,ee in SPLIT.items() for eid in ee};duplicates={}
    for e in EPS:duplicates.setdefault(e['duplicate_id'],set()).add(byid[e['episode_index']])
    if any(len(v)>1 for v in duplicates.values()):metadata_errors.append('duplicate demonstrations cross splits')
    if MAN['episodes']!=len(EPS):warnings.append('stale manifest episodes field: %s versus %s retained'%(MAN['episodes'],len(EPS)))
    selected=EPS[a.shard::a.shards]
    if a.limit:selected=selected[:a.limit]
    report={'family':str(FAMILY),'manifest_fingerprint':MAN['fingerprint'],'split_sha256':digest((FAMILY/'split.json').read_bytes()),'audit_sha256':digest(Path(__file__).read_bytes()),'shard':a.shard,'shards':a.shards,'expected_episodes':len(selected),'total_retained_episodes':len(EPS),'metadata_errors':metadata_errors,'warnings':warnings,'completed_episodes':0,'failed_episodes':0,'frames_decoded':0,'state':'running','started_unix':time.time()}
    status=OUT/('status-%02d.json'%a.shard);atomic(status,report)
    with (OUT/('episodes-%02d.jsonl'%a.shard)).open('w',buffering=1) as log,mp.get_context('fork').Pool(a.workers) as pool:
        for result in pool.imap_unordered(check,selected,chunksize=1):
            log.write(json.dumps(result)+'\n');report['completed_episodes']+=1;report['failed_episodes']+=bool(result['errors']);report['frames_decoded']+=result['frames_decoded']
            if result['errors']:print(json.dumps(result),flush=True)
            if report['completed_episodes']%100==0:
                report['updated_unix']=time.time();atomic(status,report)
    report['state']='complete';report['updated_unix']=time.time();atomic(status,report);print(json.dumps(report),flush=True)
