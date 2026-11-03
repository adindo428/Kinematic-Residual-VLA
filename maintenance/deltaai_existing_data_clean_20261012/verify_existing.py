"""Verify existing DeltaAI bytes against the exhaustively audited Palmetto family."""
import concurrent.futures, json, os, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORK=Path('/work/nvme/bivv/aparameshwaran/dynamicvla')
FAMILY=WORK/'domv-audited-20261012'
SOURCE=Path('/u/aparameshwaran/Kinematic-Residual-VLA/training_snapshots/training-20261012-video-clean/dom100')
sys.path.insert(0,str(SOURCE))
from utils.dom_experiment import atomic_json,digest,file_digest,source_digest

def main():
    manifest=json.loads((FAMILY/'manifest.json').read_text())
    pal=json.loads((HERE/'audit-receipts/palmetto-final-manifest.json').read_text())
    assert digest({k:v for k,v in manifest.items() if k!='fingerprint'})==manifest['fingerprint']
    assert digest({k:v for k,v in pal.items() if k!='fingerprint'})==pal['fingerprint']
    assert manifest['source_sha256']==source_digest(SOURCE)
    assert manifest['dataset_revision']==pal['dataset_revision']
    assert manifest['split_sha256']==pal['split_sha256']
    assert manifest['counts']==pal['counts']=={'train':197198,'validation':2015,'test':2011}
    assert manifest['train_frames']==pal['train_frames']==22370201
    for name,expected in manifest['artifacts'].items():
        assert file_digest(FAMILY/name)==expected,'Frozen artifact changed: '+name
    # These maps bind every retained byte to the prior full decode audit.
    for name in ['split.json','episodes.json','parquet_hashes.json','video_hashes.json','phase_hashes.json',
                 'initialization/config.json','validation_cases.json','validation_anchors.json','test_cases.json']:
        assert manifest['artifacts'][name]==pal['artifacts'][name],'Paired artifact differs: '+name
    assert file_digest(Path(manifest['initialization'])/'model.safetensors')==manifest['weights_sha256']==pal['weights_sha256']
    data=Path(manifest['dataset_root'])
    info=json.loads((data/'meta/info.json').read_text())
    videos=json.loads((FAMILY/'video_hashes.json').read_text())
    parquets=json.loads((FAMILY/'parquet_hashes.json').read_text())
    phases=json.loads((FAMILY/'phase_hashes.json').read_text())
    metadata=json.loads((HERE/'audit-receipts/metadata_expected.json').read_text())
    def entries():
        for rel,value in metadata.items():
            yield ('metadata',data/rel,value['sha256'] if isinstance(value,dict) else value,None)
        for rel,value in videos.items():yield ('video',data/rel,value['sha256'],value['bytes'])
        for eid,h in parquets.items():
            rel=info['data_path'].format(episode_chunk=int(eid)//info['chunks_size'],episode_index=int(eid))
            yield ('parquet',data/rel,h,None)
        for eid,h in phases.items():yield ('phase',FAMILY/'phase_cache'/('%06d.npz'%int(eid)),h,None)
    def check(item):
        kind,path,expected,size=item
        try:
            actual_size=path.stat().st_size
            if size is not None and actual_size!=size:raise ValueError('size mismatch')
            if file_digest(path)!=expected:raise ValueError('SHA256 mismatch')
            return kind,actual_size,None
        except Exception as exc:return kind,0,{'path':str(path),'kind':kind,'error':str(exc)}
    report={'state':'running','manifest':manifest['fingerprint'],'split_sha256':manifest['split_sha256'],
            'paired_audit_manifest':pal['fingerprint'],'source_audit_job':'17119970',
            'method':'Full byte identity with completed Palmetto decode audit; no repeated decode sweep',
            'total_files':len(videos)+len(parquets)+len(phases)+len(metadata),'completed_files':0,
            'counts':{},'verified_bytes':0,'errors':[],'started_unix':time.time(),'job_id':os.environ.get('SLURM_JOB_ID')}
    atomic_json(HERE/'verification-progress.json',report)
    # Keep a bounded work queue: hundreds of thousands of Futures use excessive RAM.
    items=iter(entries())
    workers=int(os.environ.get('DOM100_VERIFY_THREADS','4'))
    if not 1<=workers<=16:raise ValueError('Verification workers must be between 1 and 16')
    report['workers']=workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending=set()
        for _ in range(64):
            try:pending.add(pool.submit(check,next(items)))
            except StopIteration:break
        while pending:
            done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                kind,size,error=future.result();report['completed_files']+=1
                report['counts'][kind]=report['counts'].get(kind,0)+1
                report['verified_bytes']+=size
                if error:report['errors'].append(error)
                try:pending.add(pool.submit(check,next(items)))
                except StopIteration:pass
            if report['completed_files']%1000 < len(done):
                report['updated_unix']=time.time();atomic_json(HERE/'verification-progress.json',report)
    report['state']='failed' if report['errors'] else 'complete';report['updated_unix']=time.time()
    atomic_json(HERE/'verification-progress.json',report)
    if report['errors']:raise RuntimeError('Existing dataset checksum failures: '+str(len(report['errors'])))
    atomic_json(HERE/'verified.json',report)
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        atomic_json(HERE/'verification-error.json',{'state':'failed','error':str(exc),'updated_unix':time.time()})
        raise
