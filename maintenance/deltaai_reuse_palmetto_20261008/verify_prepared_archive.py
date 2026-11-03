import hashlib, json, tarfile
from pathlib import Path

root=Path(__file__).resolve().parent
archive=root/'palmetto-prepared-family.tar.gz'
manifest=None; expected_phase=None; metadata={}; phase_count=0
with tarfile.open(archive,mode='r|gz') as tar:
    for member in tar:
        path=Path(member.name)
        assert not path.is_absolute() and '..' not in path.parts
        if member.isdir(): continue
        assert member.isfile(), 'Unexpected archive link: '+member.name
        raw=tar.extractfile(member).read()
        checksum=hashlib.sha256(raw).hexdigest()
        if member.name=='manifest.json':
            manifest=json.loads(raw)
            fp=manifest.pop('fingerprint')
            digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            assert digest==fp
            manifest['fingerprint']=fp
        elif member.name=='phase_hashes.json': expected_phase=json.loads(raw)
        if member.name.startswith('phase_cache/'):
            ep=str(int(path.stem))
            assert checksum==expected_phase[ep], 'Phase checksum mismatch: '+ep
            phase_count+=1
        else: metadata[member.name]=checksum
assert phase_count==len(expected_phase)==manifest['episodes']
for name,value in manifest['artifacts'].items():
    if name=='initialization/config.json': continue
    assert metadata[name]==value, 'Artifact checksum mismatch: '+name
assert metadata['split.json']==manifest['split_sha256']
report={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'verified_phase_cache_files':phase_count,'verified_metadata_files':len(metadata),'source_fingerprint':manifest['fingerprint'],'split_sha256':manifest['split_sha256']}
(root/'PREPARED_ARCHIVE_REPORT.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
